# benchmark.py - run all models on a subject PDF with and without RAG
# measures response time, CPU, RAM, tokens/s per query
# scoring via GPT (grounded_accuracy + completeness, 0-10)

import os
import re
import time
import threading
import ollama
import psutil
import pandas as pd
import json
import hashlib
import csv
import pickle
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pypdf import PdfReader
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -- config --

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
API_KEY = os.getenv("OPENAI_API_KEY")

RAW_PDF_NAME = os.getenv("PDF_FILENAME", "biology.pdf")
PDF_PATH = os.path.join("files", RAW_PDF_NAME)
CATEGORY = os.getenv("QUESTION_CATEGORY", "biology")

RESULTS_FOLDER = os.path.join("results", CATEGORY)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

SUBJECT_NAME = os.path.splitext(RAW_PDF_NAME)[0]
RESULTS_CSV   = os.path.join(RESULTS_FOLDER, f"results_{SUBJECT_NAME}.csv")
RESULTS_XLSX  = os.path.join(RESULTS_FOLDER, f"results_{SUBJECT_NAME}.xlsx")
CACHE_PATH    = os.path.join("files", f"embeddings_{SUBJECT_NAME}.pkl")
GOLD_CACHE    = os.path.join("files", f"gold_{SUBJECT_NAME}.json")
QUESTION_FILE = os.path.join("questions", f"{CATEGORY}.json")

# RAG settings
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
TOP_K         = 3           # number of chunks retrieved per question
EMBED_MODEL   = "nomic-embed-text"

# Inference settings - low temperature for reproducibility
TEMPERATURE   = 0.1
MAX_TOKENS    = 2048        # max tokens generated per answer (thinking models need extra budget)
TIMEOUT_S     = 600         # 10-minute per-query timeout (CPU is slow)

# Models to benchmark (all run via Ollama)
MODELS = [
    "qwen2.5:1.5b",
    "deepseek-r1:1.5b",
    "llama3.2:1b",
    "gemma2:2b",
    "gemma3:1b",
    "gemma4:e2b",
    "qwen3.5:0.8b",
    "qwen3:0.6b",
]

# Max retries when a model returns an empty answer (thinking models sometimes do this)
MAX_ANSWER_RETRIES = 2

CONDITIONS = ["no_rag", "rag"]

print(f"Ollama host : {OLLAMA_HOST}")
print(f"Category    : {CATEGORY}  |  Subject: {SUBJECT_NAME}")
print(f"Models      : {MODELS}")
print(f"Conditions  : {CONDITIONS}")

openai_client = OpenAI(api_key=API_KEY)
# 10-minute timeout so slow CPU inference doesn't disconnect
ollama_client = ollama.Client(host=OLLAMA_HOST, timeout=TIMEOUT_S)


# -- resource monitor --

class ResourceMonitor:

    def __init__(self, interval=0.5):
        self.interval    = interval
        self.cpu_samples = []
        self.ram_samples = []
        self._stop       = threading.Event()
        self._thread     = None
        self._proc       = psutil.Process()

    def start(self):
        self.cpu_samples.clear()
        self.ram_samples.clear()
        self._stop.clear()
        # Establish CPU baseline before spawning thread
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self.interval):
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            try:
                self.ram_samples.append(self._proc.memory_info().rss / 1024 / 1024)
            except psutil.NoSuchProcess:
                break

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def cpu_avg(self):
        return round(sum(self.cpu_samples) / len(self.cpu_samples), 1) if self.cpu_samples else 0.0

    @property
    def cpu_max(self):
        return round(max(self.cpu_samples), 1) if self.cpu_samples else 0.0

    @property
    def ram_start_mb(self):
        return round(self.ram_samples[0], 1) if self.ram_samples else 0.0

    @property
    def ram_max_mb(self):
        return round(max(self.ram_samples), 1) if self.ram_samples else 0.0


# -- embeddings --

def _file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_or_create_embeddings(pdf_path, cache_path):
    print(f"\nChecking embeddings cache...")
    current_hash = _file_hash(pdf_path)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            if data.get("hash") == current_hash and data.get("chunks"):
                print(f"Cache hit - {len(data['chunks'])} chunks loaded.")
                return data["chunks"]
            print("Cache miss (PDF changed or empty). Rebuilding...")
        except Exception:
            print("Cache corrupted. Rebuilding...")

    print(f"Reading PDF: {pdf_path}")
    reader   = PdfReader(pdf_path)
    full_text = "".join(page.extract_text() or "" for page in reader.pages)

    # split into overlapping chunks so that sentences near chunk boundaries
    # are not cut off and lost from retrieval
    raw_chunks = [
        full_text[i : i + CHUNK_SIZE]
        for i in range(0, len(full_text), CHUNK_SIZE - CHUNK_OVERLAP)
    ]

    processed = []
    print(f"Embedding {len(raw_chunks)} chunks with {EMBED_MODEL}...")
    for idx, chunk in enumerate(raw_chunks):
        if idx % 10 == 0:
            print(f"  {idx}/{len(raw_chunks)}", end="\r")
        for attempt in range(3):
            try:
                vec = ollama_client.embeddings(model=EMBED_MODEL, prompt=chunk)["embedding"]
                processed.append({"text": chunk, "vector": vec})
                break
            except Exception as e:
                if attempt == 2:
                    print(f"\n  Skipped chunk {idx}: {e}")
                time.sleep(1)

    with open(cache_path, "wb") as f:
        pickle.dump({"hash": current_hash, "chunks": processed}, f)

    print(f"\nEmbeddings saved: {cache_path}")
    return processed


def get_top_k_context(question, chunks, k=TOP_K):
    # retrieve the top-k most relevant chunks using dot-product similarity
    # (cosine without normalisation, which is fine since nomic-embed-text produces unit vectors)
    q_vec = ollama_client.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]

    scored = sorted(
        ((sum(a * b for a, b in zip(q_vec, item["vector"])), item["text"]) for item in chunks),
        key=lambda x: x[0],
        reverse=True,
    )
    return "\n\n---\n\n".join(text for _, text in scored[:k])


# -- gold references --

def _strip_answer_prefix(raw):
    return re.sub(r"^A\d+:\s*", "", raw.strip())


def load_or_create_gold_references(chunks):
    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_questions   = raw_data.get("questions", [])
    raw_answers     = raw_data.get("answers", [])   # pre-written answers array

    # Load cached auto-generated gold refs (only used as fallback)
    cached_gold: dict = {}
    if os.path.exists(GOLD_CACHE):
        try:
            with open(GOLD_CACHE, "r", encoding="utf-8") as f:
                cached_gold = json.load(f)
        except Exception:
            pass

    gold_refs = []
    changed   = False

    for i, q in enumerate(raw_questions):
        # 1. Inline dict with full gold package
        if isinstance(q, dict) and q.get("reference_answer") and q.get("key_points"):
            gold_refs.append({
                "question":         q["question"],
                "reference_answer": q["reference_answer"],
                "key_points":       q.get("key_points", []),
                "source_passages":  q.get("source_passages", []),
            })
            continue

        question_text = q if isinstance(q, str) else q.get("question", "")

        # 2. Pre-written answer from the JSON answers array
        if i < len(raw_answers) and raw_answers[i]:
            gold_refs.append({
                "question":         question_text,
                "reference_answer": _strip_answer_prefix(raw_answers[i]),
                "key_points":       [],
                "source_passages":  [],
            })
            continue

        # 3. Cached auto-generated reference
        if question_text in cached_gold:
            gold_refs.append(cached_gold[question_text])
            continue

        # 4. Auto-generate via GPT-5-mini
        print(f"  Generating gold reference for Q{i+1}: {question_text[:60]}...")
        context = get_top_k_context(question_text, chunks, k=TOP_K)

        gen_prompt = (
            "You are creating a gold reference package for a study benchmark.\n\n"
            "Source text from a course PDF:\n"
            "---\n"
            f"{context}\n"
            "---\n\n"
            f"Question: {question_text}\n\n"
            "Create a gold reference package. Return ONLY valid JSON with these keys:\n"
            '- "reference_answer": 2-4 sentence answer based strictly on the source text\n'
            '- "key_points": list of 3-5 specific facts that a complete answer must include\n'
            '- "source_passages": list of exact phrases from the text that support the answer'
        )

        try:
            resp = openai_client.chat.completions.create(
                model="gpt-5-mini",
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": gen_prompt}],
                temperature=0.2,
            )
            pkg = json.loads(resp.choices[0].message.content)
            gold = {
                "question":         question_text,
                "reference_answer": pkg.get("reference_answer", ""),
                "key_points":       pkg.get("key_points", []),
                "source_passages":  pkg.get("source_passages", []),
            }
        except Exception as e:
            print(f"  Gold reference error for Q{i+1}: {e}")
            gold = {
                "question":         question_text,
                "reference_answer": "",
                "key_points":       [],
                "source_passages":  [],
            }

        cached_gold[question_text] = gold
        gold_refs.append(gold)
        changed = True

    if changed:
        with open(GOLD_CACHE, "w", encoding="utf-8") as f:
            json.dump(cached_gold, f, indent=2, ensure_ascii=False)

    return gold_refs


# -- questions --

def load_questions(question_file):
    try:
        with open(question_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        questions = []
        for q in data.get("questions", []):
            if isinstance(q, str):
                questions.append(q)
            elif isinstance(q, dict):
                questions.append(q.get("question", ""))
        print(f"Loaded {len(questions)} questions from {question_file}")
        return questions
    except FileNotFoundError:
        print(f"Question file not found: {question_file}")
        return []
    except Exception as e:
        print(f"Error loading questions: {e}")
        return []


# -- model query --

def query_model(model_name, question, context, condition, allow_think_fallback=False):
    if condition == "rag":
        prompt = (
            "You are a study assistant. Answer the question using ONLY the information "
            "in the text below. Do not add anything not found in the text.\n\n"
            f"Text:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer (based strictly on the text above):"
        )
    else:
        prompt = (
            "You are a study assistant. Answer the following study question "
            "concisely and accurately.\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    monitor   = ResourceMonitor(interval=0.5)
    ram_before = psutil.Process().memory_info().rss / 1024 / 1024

    monitor.start()
    t_start = time.perf_counter()

    try:
        # Disable thinking for all models - keeps inference fast on CPU and
        # avoids empty output when a model exhausts its token budget in <think>.
        # Models that don't support this flag ignore it safely.
        response = ollama_client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
            think=False,
        )
        t_end = time.perf_counter()
        monitor.stop()

        raw_content = response["message"]["content"].strip()

        # Extract thinking content before stripping (used as fallback if no answer follows)
        think_match = re.search(r"<think>(.*?)</think>", raw_content, flags=re.DOTALL)
        think_content = think_match.group(1).strip() if think_match else ""

        # Strip <think>...</think> blocks produced by reasoning models (e.g. qwen3.x, deepseek-r1)
        answer = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

        # If model only produced thinking with no trailing answer, grab whatever came after </think>
        if not answer and "</think>" in raw_content:
            answer = raw_content.split("</think>")[-1].strip()

        # Handle incomplete <think> block (no closing tag - model truncated mid-think)
        if not answer and "<think>" in raw_content and "</think>" not in raw_content:
            think_content = raw_content.split("<think>", 1)[-1].strip()

        # Last-resort: use end of thinking content so we don't waste the run on an empty score
        if not answer and allow_think_fallback and think_content:
            answer = think_content[-1500:].strip()

        # Token metrics from Ollama response
        eval_count    = response.get("eval_count", 0)
        eval_dur_ns   = response.get("eval_duration", 0)
        prompt_tokens = response.get("prompt_eval_count", 0)
        tps           = eval_count / (eval_dur_ns / 1e9) if eval_dur_ns > 0 else 0.0

        metrics = {
            "wall_time_s":      round(t_end - t_start, 2),
            "tokens_generated": eval_count,
            "prompt_tokens":    prompt_tokens,
            "tokens_per_second": round(tps, 2),
            "cpu_percent_avg":  monitor.cpu_avg,
            "cpu_percent_max":  monitor.cpu_max,
            "ram_mb_before":    round(ram_before, 1),
            "ram_mb_max":       monitor.ram_max_mb,
            "ram_mb_delta":     round(monitor.ram_max_mb - ram_before, 1),
        }
        return answer, metrics

    except Exception as e:
        t_end = time.perf_counter()
        monitor.stop()
        metrics = {
            "wall_time_s":       round(t_end - t_start, 2),
            "tokens_generated":  0,
            "prompt_tokens":     0,
            "tokens_per_second": 0.0,
            "cpu_percent_avg":   monitor.cpu_avg,
            "cpu_percent_max":   monitor.cpu_max,
            "ram_mb_before":     round(ram_before, 1),
            "ram_mb_max":        round(ram_before, 1),
            "ram_mb_delta":      0.0,
        }
        return f"ERROR: {e}", metrics


# -- evaluation --

def evaluate_answer(question, answer, gold_ref, context, condition):
    # rubric-based LLM-as-a-Judge via GPT-5 mini
    # two dimensions scored 0-5 each (max 10 total): grounded_accuracy + completeness
    # for RAG runs only, document_grounding is also recorded but NOT added to the total
    # so that RAG and no-RAG scores remain directly comparable
    reference_answer = gold_ref.get("reference_answer", "")
    key_points       = gold_ref.get("key_points", [])

    key_points_str = (
        "\n".join(f"- {kp}" for kp in key_points) if key_points else "(not available)"
    )

    if condition == "no_rag":
        eval_prompt = (
            "You are a strict evaluator for a study assistant benchmark.\n\n"
            f"QUESTION:\n{question}\n\n"
            "CORRECT ANSWER (authoritative - use this as the primary grading standard):\n"
            f"{reference_answer}\n\n"
            f"REQUIRED KEY POINTS:\n{key_points_str}\n\n"
            f"ANSWER TO EVALUATE:\n{answer}\n\n"
            "Score the answer on TWO dimensions using a 0–5 scale. Use whole integers only (0, 1, 2, 3, 4, or 5).\n\n"
            "GROUNDED_ACCURACY (0-5):\n"
            "  5 = Fully accurate, consistent with the correct answer, no contradictions\n"
            "  4 = Accurate with only trivial imprecision\n"
            "  3 = Mostly accurate, one minor error\n"
            "  2 = Partially accurate, some errors or unsupported claims\n"
            "  1 = Mostly inaccurate or heavily hallucinated\n"
            "  0 = Completely wrong or off-topic\n\n"
            "COMPLETENESS (0-5):\n"
            "  5 = Covers all key ideas from the correct answer\n"
            "  4 = Covers almost all key ideas, one small gap\n"
            "  3 = Covers most key ideas, one notable omission\n"
            "  2 = Covers some key ideas, missing several\n"
            "  1 = Covers only one key idea or mostly misses the point\n"
            "  0 = Misses all key ideas\n\n"
            "ERROR_TYPE - classify the primary failure mode (use \"none\" if the answer is good):\n"
            '  "none"             - answer is correct and complete, no notable failure\n'
            '  "hallucination"    - answer contains facts not in the reference answer or source document\n'
            '  "retrieval_ignore" - RAG only: model received a relevant chunk but did not use it\n'
            '  "truncation"       - answer is cut off or incomplete, missing key points\n'
            '  "confusion"        - model mixes up two concepts or attributes facts to the wrong entity\n'
            '  "off_topic"        - answer does not address the question asked\n'
            '  "context_overflow" - answer degrades mid-response or contradicts itself, likely context window issue\n\n'
            "Return ONLY this JSON (no other text):\n"
            '{"grounded_accuracy": <0-5>, "completeness": <0-5>, "total": <0-10>, "feedback": "<one sentence>", "error_type": "<see taxonomy above>"}'
        )
    else:
        eval_prompt = (
            "You are a strict evaluator for a study assistant benchmark.\n\n"
            f"QUESTION:\n{question}\n\n"
            "CORRECT ANSWER (authoritative - use this as the primary grading standard):\n"
            f"{reference_answer}\n\n"
            f"REQUIRED KEY POINTS:\n{key_points_str}\n\n"
            "SOURCE TEXT (the document passages the model received - use only for document_grounding):\n"
            "---\n"
            f"{context[:3000]}\n"
            "---\n\n"
            f"ANSWER TO EVALUATE:\n{answer}\n\n"
            "Score the answer on TWO comparable dimensions plus ONE extra RAG-only dimension.\n"
            "All dimensions use a 0–5 scale. Use whole integers only (0, 1, 2, 3, 4, or 5).\n\n"
            "GROUNDED_ACCURACY (0-5):\n"
            "  5 = Fully accurate, consistent with the correct answer and source, no contradictions\n"
            "  4 = Accurate with only trivial imprecision\n"
            "  3 = Mostly accurate, one minor error\n"
            "  2 = Partially accurate, some errors or unsupported claims\n"
            "  1 = Mostly inaccurate or heavily hallucinated\n"
            "  0 = Completely wrong or off-topic\n\n"
            "COMPLETENESS (0-5):\n"
            "  5 = Covers all key ideas from the correct answer\n"
            "  4 = Covers almost all key ideas, one small gap\n"
            "  3 = Covers most key ideas, one notable omission\n"
            "  2 = Covers some key ideas, missing several\n"
            "  1 = Covers only one key idea or mostly misses the point\n"
            "  0 = Misses all key ideas\n\n"
            "DOCUMENT_GROUNDING (0-5) - extra RAG-only column, not added to total:\n"
            "  5 = Every claim is directly supported by the source text above\n"
            "  4 = Almost all claims supported, one minor unsupported detail\n"
            "  3 = Most claims supported, some additions not in the source\n"
            "  2 = About half the claims are unsupported by the source\n"
            "  1 = Most claims are not supported by the source\n"
            "  0 = Answer ignores the source entirely\n\n"
            "ERROR_TYPE - classify the primary failure mode (use \"none\" if the answer is good):\n"
            '  "none"             - answer is correct and complete, no notable failure\n'
            '  "hallucination"    - answer contains facts not in the reference answer or source document\n'
            '  "retrieval_ignore" - RAG only: model received a relevant chunk but did not use it\n'
            '  "truncation"       - answer is cut off or incomplete, missing key points\n'
            '  "confusion"        - model mixes up two concepts or attributes facts to the wrong entity\n'
            '  "off_topic"        - answer does not address the question asked\n'
            '  "context_overflow" - answer degrades mid-response or contradicts itself, likely context window issue\n\n'
            "Return ONLY this JSON (no other text):\n"
            '{"grounded_accuracy": <0-5>, "completeness": <0-5>, '
            '"document_grounding": <0-5>, "total": <0-10>, "feedback": "<one sentence>", "error_type": "<see taxonomy above>"}'
        )

    try:
        resp   = openai_client.chat.completions.create(
            model="gpt-5-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": eval_prompt}],
        )
        scores = json.loads(resp.choices[0].message.content)
        ga  = max(0, min(5, int(round(float(scores.get("grounded_accuracy", 0))))))
        co  = max(0, min(5, int(round(float(scores.get("completeness",       0))))))
        dg  = max(0, min(5, int(round(float(scores.get("document_grounding", 0)))))) if condition == "rag" else None
        et  = scores.get("error_type", "parse_error")
        if not isinstance(et, str) or not et.strip():
            et = "parse_error"
        return {
            "score_grounded_accuracy":  ga,
            "score_completeness":       co,
            "score_document_grounding": dg,
            "score_total":              ga + co,
            "score_feedback":           scores.get("feedback", ""),
            "score_error_type":         et,
        }
    except Exception as e:
        print(f"    Evaluation error: {e}")
        return {
            "score_grounded_accuracy":  0,
            "score_completeness":       0,
            "score_document_grounding": None if condition == "no_rag" else 0,
            "score_total":              0,
            "score_feedback":           f"ERROR: {e}",
            "score_error_type":         "parse_error",
        }


# -- save results --

def save_results(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)

    try:
        df.to_csv(RESULTS_CSV, index=False, quoting=csv.QUOTE_ALL)
    except Exception as e:
        print(f"CSV save error: {e}")

    try:
        with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Results")
            ws = writer.book["Results"]
            for idx in range(1, len(df.columns) + 1):
                ws.column_dimensions[get_column_letter(idx)].width = 35
            for row in ws.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(wrapText=False, vertical="top")
    except Exception as e:
        print(f"Excel save error: {e}")


# -- aggregation and plotting --

def aggregate_category_benchmark(category):
    print(f"\nAGGREGATING: {category.upper()}")

    cat_folder = os.path.join("results", category)
    files      = glob.glob(os.path.join(cat_folder, "results_*.csv"))
    if not files:
        print(f"No result files found in {cat_folder}")
        return

    all_dfs = []
    for f in files:
        try:
            all_dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_dfs:
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # Quality leaderboard: avg score_total per model × condition
    quality = (
        combined.groupby(["model", "condition"])["score_total"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    quality.columns = ["model", "condition", "avg_score", "std_score", "n"]
    quality["avg_score"] = quality["avg_score"].round(3)
    quality["std_score"] = quality["std_score"].round(3)
    quality = quality.sort_values("avg_score", ascending=False).reset_index(drop=True)

    print(f"\nQuality results (score out of 10 - grounded_accuracy + completeness):")
    print(quality.to_string(index=False))

    leaderboard_path = os.path.join(cat_folder, f"{category}_LEADERBOARD.csv")
    quality.to_csv(leaderboard_path, index=False)

    # Performance summary: avg resource metrics per model × condition
    perf_cols = [
        "model", "condition",
        "wall_time_s", "tokens_per_second",
        "cpu_percent_avg", "cpu_percent_max",
        "ram_mb_before", "ram_mb_max", "ram_mb_delta",
    ]
    existing_perf_cols = [c for c in perf_cols if c in combined.columns]
    if len(existing_perf_cols) > 2:
        perf = combined[existing_perf_cols].groupby(["model", "condition"]).mean().round(2).reset_index()
        perf.to_csv(os.path.join(cat_folder, f"{category}_METRICS_SUMMARY.csv"), index=False)
        print(f"\nPerformance summary:")
        print(perf.to_string(index=False))

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"{category.capitalize()} Benchmark - Answer Quality (score / 10)",
        fontsize=13, fontweight="bold",
    )

    for ax, condition in zip(axes, ["no_rag", "rag"]):
        cdata = quality[quality["condition"] == condition].sort_values("avg_score")
        colors = ["steelblue" if condition == "rag" else "coral"] * len(cdata)
        bars = ax.barh(cdata["model"], cdata["avg_score"], color=colors)
        ax.set_xlim(0, 10)
        ax.set_xlabel("Average Score (max 10)")
        ax.set_title(f'{"RAG" if condition == "rag" else "No RAG"} condition')
        for bar, score in zip(bars, cdata["avg_score"]):
            ax.text(
                bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}",
                va="center", fontsize=9,
            )

    plt.tight_layout()
    quality_plot = os.path.join(cat_folder, f"{category}_BENCHMARK_PLOT.png")
    plt.savefig(quality_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Quality plot saved: {quality_plot}")

    # Performance plots (only if we have the data)
    if len(existing_perf_cols) > 2:
        _plot_performance(combined, cat_folder, category)


def _plot_performance(df, folder, category):
    metrics_to_plot = [
        ("wall_time_s",       "Avg Response Time (s)"),
        ("tokens_per_second", "Tokens per Second"),
        ("cpu_percent_avg",   "Avg CPU Usage (%)"),
        ("ram_mb_delta",      "RAM Increase During Inference (MB)"),
    ]
    available = [(col, label) for col, label in metrics_to_plot if col in df.columns]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        f"{category.capitalize()} - Resource Usage per Model × Condition",
        fontsize=12, fontweight="bold",
    )

    for ax, (col, label) in zip(axes, available):
        pivot = df.groupby(["model", "condition"])[col].mean().unstack("condition")
        pivot.plot(kind="barh", ax=ax, color=["coral", "steelblue"])
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.legend(title="Condition")

    plt.tight_layout()
    perf_plot = os.path.join(folder, f"{category}_PERFORMANCE_PLOT.png")
    plt.savefig(perf_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Performance plot saved: {perf_plot}")


def aggregate_global_benchmark():
    print("\nGLOBAL BENCHMARK (all domains)")

    categories = ["biology", "history", "law"]
    all_rows   = []

    for cat in categories:
        lb = os.path.join("results", cat, f"{cat}_LEADERBOARD.csv")
        if os.path.exists(lb):
            try:
                df = pd.read_csv(lb)
                df["category"] = cat
                all_rows.append(df)
            except Exception as e:
                print(f"Error reading {lb}: {e}")

    if not all_rows:
        print("No leaderboard data found.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    global_lb = (
        combined.groupby(["model", "condition"])["avg_score"]
        .mean()
        .reset_index()
        .rename(columns={"avg_score": "global_avg_score"})
    )
    global_lb["global_avg_score"] = global_lb["global_avg_score"].round(3)
    global_lb = global_lb.sort_values("global_avg_score", ascending=False).reset_index(drop=True)

    print("\nGlobal leaderboard (avg score / 10 across all domains):")
    print(global_lb.to_string(index=False))

    global_lb.to_csv(os.path.join("results", "GLOBAL_LEADERBOARD.csv"), index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Global Benchmark - Average Score Across Biology, Law & History",
        fontsize=13, fontweight="bold",
    )

    for ax, condition in zip(axes, ["no_rag", "rag"]):
        cdata = global_lb[global_lb["condition"] == condition].sort_values("global_avg_score")
        color = "steelblue" if condition == "rag" else "coral"
        bars  = ax.barh(cdata["model"], cdata["global_avg_score"], color=color)
        ax.set_xlim(0, 10)
        ax.set_xlabel("Average Score (max 10)")
        ax.set_title(f'{"RAG" if condition == "rag" else "No RAG"} condition')
        for bar, score in zip(bars, cdata["global_avg_score"]):
            ax.text(
                bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}",
                va="center", fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(os.path.join("results", "GLOBAL_BENCHMARK_PLOT.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Global plot saved: results/GLOBAL_BENCHMARK_PLOT.png")


# -- main --

def main():
    print(f"\nBENCHMARK: {SUBJECT_NAME.upper()}  ({CATEGORY})")

    cached_chunks = load_or_create_embeddings(PDF_PATH, CACHE_PATH)
    questions     = load_questions(QUESTION_FILE)
    if not questions:
        print("No questions loaded. Exiting.")
        return

    print("\nPreparing gold reference packages...")
    gold_refs = load_or_create_gold_references(cached_chunks)

    # Resume support - skip already-completed (question, model, condition) combos.
    # Rows where the answer was empty or errored are excluded from done_keys so they
    # get re-run automatically on the next run.
    # resume support: if results CSV already exists, load completed rows so a crashed
    # or interrupted run continues from where it left off instead of starting over
    done_keys = set()
    rows      = []
    if os.path.exists(RESULTS_CSV):
        try:
            existing = pd.read_csv(RESULTS_CSV)
            failed_mask = (
                existing["answer"].isna()
                | (existing["answer"].astype(str).str.strip() == "")
                | existing["answer"].astype(str).str.startswith("ERROR:")
                | (existing.get("score_feedback", "").astype(str) == "Model error or empty answer - not evaluated")
            )
            good_rows = existing[~failed_mask]
            bad_count = failed_mask.sum()
            rows      = good_rows.to_dict("records")
            done_keys = set(
                zip(good_rows["question_idx"], good_rows["model"], good_rows["condition"])
            )
            skipped_msg = f"  ({bad_count} failed rows will be retried)" if bad_count else ""
            print(f"Resuming: {len(rows)} valid results loaded.{skipped_msg}")
        except Exception:
            pass

    total_runs = len(questions) * len(MODELS) * len(CONDITIONS)
    remaining  = total_runs - len(done_keys)
    print(f"\nTotal runs: {total_runs}  |  Remaining: {remaining}")
    print(f"(Sequential execution - no GPU, CPU inference only)\n")

    for q_idx, question in enumerate(questions):
        gold_ref = gold_refs[q_idx]
        context  = get_top_k_context(question, cached_chunks, k=TOP_K)

        print(f"\n[Q{q_idx+1}/{len(questions)}] {question[:80]}...")

        for model in MODELS:
            for condition in CONDITIONS:
                key = (q_idx + 1, model, condition)
                if key in done_keys:
                    continue

                print(f"  [{model}] [{condition.upper():6s}] ", end="", flush=True)

                answer, metrics = query_model(
                    model_name=model,
                    question=question,
                    context=context,
                    condition=condition,
                )

                # Retry if the answer came back empty (thinking models can produce no output)
                # On the final retry, allow falling back to thinking content so we never
                # send an empty answer to the evaluator.
                retry_num = 0
                while (
                    not answer.strip()
                    and not answer.startswith("ERROR:")
                    and retry_num < MAX_ANSWER_RETRIES
                ):
                    retry_num += 1
                    is_last_retry = (retry_num == MAX_ANSWER_RETRIES)
                    print(f"[empty→retry {retry_num}] ", end="", flush=True)
                    answer, metrics = query_model(
                        model_name=model,
                        question=question,
                        context=context,
                        condition=condition,
                        allow_think_fallback=is_last_retry,
                    )

                print(
                    f"{metrics['wall_time_s']:6.1f}s | "
                    f"{metrics['tokens_per_second']:5.1f} tok/s | "
                    f"CPU {metrics['cpu_percent_avg']:4.0f}% | "
                    f"RAM +{metrics['ram_mb_delta']:5.1f} MB",
                    end="",
                    flush=True,
                )

                is_error = answer.startswith("ERROR:") or not answer.strip()
                if is_error:
                    scores = {
                        "score_grounded_accuracy":  0,
                        "score_completeness":        0,
                        "score_document_grounding":  None if condition == "no_rag" else 0,
                        "score_total":               0,
                        "score_feedback":            "Model error or empty answer - not evaluated",
                        "score_error_type":          "parse_error",
                    }
                else:
                    scores = evaluate_answer(question, answer, gold_ref, context, condition)

                print(f" | score {scores['score_total']}/10")

                row = {
                    "question_idx":    q_idx + 1,
                    "question":        question,
                    "model":           model,
                    "condition":       condition,
                    # Answer (capped at 1500 chars to keep CSV manageable)
                    "answer":          answer[:1500],
                    "reference_answer": gold_ref.get("reference_answer", ""),
                    "key_points":      " | ".join(gold_ref.get("key_points", [])),
                    # Retrieved chunks: full context for RAG, empty string for no_rag
                    "retrieved_chunks": context if condition == "rag" else "",
                    # Scores
                    **scores,
                    # Resource metrics
                    **metrics,
                }

                rows.append(row)
                done_keys.add(key)

                # Save after every single run so progress isn't lost
                save_results(rows)

    print(f"\nBENCHMARK COMPLETE: {SUBJECT_NAME.upper()}")
    print(f"Results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
    aggregate_category_benchmark(CATEGORY)
    aggregate_global_benchmark()
    print("\nAll done.")
