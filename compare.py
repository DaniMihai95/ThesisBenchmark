# Cross-condition analysis for the thesis benchmark.
# Loads results from all three domains and generates the comparison plots.
# Run this after benchmark.py has finished for biology, history and law.

import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

# -- settings --

CATEGORIES    = ["biology", "history", "law"]
RESULTS_DIR   = "results"
SCORE_MAX     = 10
COMPONENT_MAX = 5
ALPHA         = 0.05
N_BOOT        = 2000
RNG           = np.random.default_rng(42)

# each question index maps to one of the five cognitive types (6 questions each, 30 total)
QUESTION_TYPE_MAP = {
    1: "Definition / Literal Retrieval",  2: "Definition / Literal Retrieval",
    3: "Definition / Literal Retrieval",  4: "Definition / Literal Retrieval",
    5: "Definition / Literal Retrieval",  6: "Definition / Literal Retrieval",
    7: "Reorganization / Integration",    8: "Reorganization / Integration",
    9: "Reorganization / Integration",   10: "Reorganization / Integration",
   11: "Reorganization / Integration",   12: "Reorganization / Integration",
   13: "Explanation / Causal",           14: "Explanation / Causal",
   15: "Explanation / Causal",           16: "Explanation / Causal",
   17: "Explanation / Causal",           18: "Explanation / Causal",
   19: "Comparison / Distinction",       20: "Comparison / Distinction",
   21: "Comparison / Distinction",       22: "Comparison / Distinction",
   23: "Comparison / Distinction",       24: "Comparison / Distinction",
   25: "Application / Conditional",      26: "Application / Conditional",
   27: "Application / Conditional",      28: "Application / Conditional",
   29: "Application / Conditional",      30: "Application / Conditional",
}

QUESTION_TYPE_ORDER = [
    "Definition / Literal Retrieval",
    "Reorganization / Integration",
    "Explanation / Causal",
    "Comparison / Distinction",
    "Application / Conditional",
]

CONDITION_COLORS = {"no_rag": "#E07856", "rag": "#4A7BB7"}
DOMAIN_COLORS    = {"biology": "#2E8B57", "history": "#B8860B", "law": "#6A5ACD"}
SIG_POS, SIG_NEG, SIG_NS = "#2E8B57", "#C0392B", "#95A5A6"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         13,
    "axes.labelsize":    13,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.dpi":        110,
    "savefig.bbox":      "tight",
})


# -- data loading --

def load_all_results():
    frames = []
    for cat in CATEGORIES:
        pattern = os.path.join(RESULTS_DIR, cat, f"results_{cat}.csv")
        for path in glob.glob(pattern):
            try:
                df = pd.read_csv(path)
                df["domain"] = cat
                frames.append(df)
                print(f"Loaded {len(df):4d} rows from {path}")
            except Exception as e:
                print(f"Error reading {path}: {e}")
    if not frames:
        print("No result files found. Run benchmark.py first.")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in ["score_total", "score_grounded_accuracy", "score_completeness",
                "score_document_grounding", "wall_time_s", "tokens_per_second",
                "cpu_percent_avg", "cpu_percent_max"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# -- stats helpers --

def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05):
    # resample with replacement n_boot times and take the percentile interval
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(lo), 3), round(float(hi), 3)


def cohens_d_paired(diff):
    # effect size for paired data: mean difference / SD of differences
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2 or diff.std(ddof=1) == 0:
        return float("nan")
    return float(diff.mean() / diff.std(ddof=1))


def effect_size_label(d):
    # Cohen's conventions: <0.2 negligible, <0.5 small, <0.8 medium, else large
    if pd.isna(d):
        return "n/a"
    ad = abs(d)
    if ad < 0.2: return "negligible"
    if ad < 0.5: return "small"
    if ad < 0.8: return "medium"
    return "large"


def paired_wilcoxon(rag_scores, norag_scores):
    # two-sided Wilcoxon signed-rank test on per-question score differences
    # rank-biserial correlation = (R+ - R-) / (R+ + R-), used as a non-parametric effect size
    diff = np.asarray(rag_scores) - np.asarray(norag_scores)
    nz = diff[diff != 0]
    if len(nz) < 2:
        return float("nan"), 1.0, float("nan")
    try:
        w_stat, p_val = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return float("nan"), 1.0, float("nan")
    ranks   = stats.rankdata(np.abs(nz))
    r_plus  = ranks[nz > 0].sum()
    r_minus = ranks[nz < 0].sum()
    total   = r_plus + r_minus
    rbc = (r_plus - r_minus) / total if total > 0 else float("nan")
    return float(w_stat), float(p_val), float(rbc)


# -- aggregations --

def rag_vs_norag(df):
    records = []
    for model, mdf in df.groupby("model"):
        row = {"model": model}
        for cond in ["no_rag", "rag"]:
            sub = mdf[mdf["condition"] == cond]["score_total"].dropna().values
            row[f"{cond}_mean"] = round(float(sub.mean()), 3) if len(sub) else float("nan")
            lo, hi = bootstrap_ci(sub)
            row[f"{cond}_ci_lo"] = lo
            row[f"{cond}_ci_hi"] = hi
            row[f"{cond}_n"]     = int(len(sub))
        if not pd.isna(row.get("rag_mean")) and not pd.isna(row.get("no_rag_mean")):
            row["delta"] = round(row["rag_mean"] - row["no_rag_mean"], 3)
        records.append(row)
    out = pd.DataFrame(records)
    if "delta" in out.columns:
        out = out.sort_values("delta", ascending=False)
    return out.reset_index(drop=True)


def domain_breakdown(df):
    return (
        df.groupby(["model", "condition", "domain"])["score_total"]
        .mean().round(3).reset_index()
        .rename(columns={"score_total": "avg_score"})
        .sort_values(["domain", "condition", "avg_score"], ascending=[True, True, False])
    )


def component_breakdown(df):
    cols = [c for c in ["score_grounded_accuracy", "score_completeness"] if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return df.groupby(["model", "condition"])[cols].mean().round(3).reset_index()


def document_grounding_summary(df):
    rag_df = df[df["condition"] == "rag"].copy()
    if "score_document_grounding" not in rag_df.columns:
        print("score_document_grounding column not found - skipping.")
        return pd.DataFrame()
    return (
        rag_df.groupby(["model", "domain"])["score_document_grounding"]
        .mean().round(3).reset_index()
        .rename(columns={"score_document_grounding": "avg_doc_grounding"})
        .sort_values(["domain", "avg_doc_grounding"], ascending=[True, False])
    )


def wilcoxon_rag_effect(df, by_domain=False):
    if "question_idx" not in df.columns:
        print("wilcoxon_rag_effect: question_idx missing - skipping.")
        return pd.DataFrame()
    group_cols = ["model", "domain"] if by_domain else ["model"]
    records = []
    for keys, mdf in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rag_df   = mdf[mdf["condition"] == "rag"][["question_idx", "domain", "score_total"]].rename(columns={"score_total": "rag_score"})
        norag_df = mdf[mdf["condition"] == "no_rag"][["question_idx", "domain", "score_total"]].rename(columns={"score_total": "norag_score"})
        paired   = pd.merge(rag_df, norag_df, on=["question_idx", "domain"]).dropna(subset=["rag_score", "norag_score"])
        if len(paired) < 2:
            continue
        diff       = (paired["rag_score"] - paired["norag_score"]).values
        rag_mean   = round(float(paired["rag_score"].mean()), 3)
        norag_mean = round(float(paired["norag_score"].mean()), 3)
        delta      = round(rag_mean - norag_mean, 3)
        d          = cohens_d_paired(diff)
        w_stat, p_val, rbc = paired_wilcoxon(paired["rag_score"].values, paired["norag_score"].values)
        wins        = int((diff > 0).sum())
        losses      = int((diff < 0).sum())
        ties        = int((diff == 0).sum())
        win_rate    = round(wins / len(diff), 3)
        significant = bool(p_val < ALPHA)
        if significant and delta > 0:
            verdict = "RAG helps"
        elif significant and delta < 0:
            verdict = "RAG hurts"
        else:
            verdict = "no significant effect"
        rec = dict(zip(group_cols, keys))
        rec.update({
            "n_pairs":       len(paired),
            "no_rag_mean":   norag_mean,
            "rag_mean":      rag_mean,
            "delta":         delta,
            "cohens_d":      round(d, 3) if not pd.isna(d) else float("nan"),
            "effect_size":   effect_size_label(d),
            "rank_biserial": round(rbc, 3) if not pd.isna(rbc) else float("nan"),
            "wins":          wins,
            "losses":        losses,
            "ties":          ties,
            "win_rate":      win_rate,
            "w_statistic":   round(w_stat, 4) if not pd.isna(w_stat) else float("nan"),
            "p_value":       round(p_val, 4),
            "significant":   significant,
            "verdict":       verdict,
        })
        records.append(rec)
    return pd.DataFrame(records)


def efficiency_frontier(df):
    # score per second = answer quality / wall-clock time, used to identify Pareto-optimal models
    if "wall_time_s" not in df.columns:
        print("efficiency_frontier: wall_time_s column missing - skipping.")
        return pd.DataFrame()
    agg = (
        df.groupby(["model", "condition"])
        .agg(avg_score=("score_total", "mean"),
             avg_wall_time_s=("wall_time_s", "mean"))
        .reset_index()
    )
    agg["avg_score"]        = agg["avg_score"].round(3)
    agg["avg_wall_time_s"]  = agg["avg_wall_time_s"].round(3)
    agg["score_per_second"] = (agg["avg_score"] / agg["avg_wall_time_s"].replace(0, np.nan)).round(4)
    return agg.sort_values("score_per_second", ascending=False).reset_index(drop=True)


def question_type_breakdown(df):
    if "question_idx" not in df.columns:
        print("question_type_breakdown: question_idx missing - skipping.")
        return pd.DataFrame()
    out = df.copy()
    out["question_type"] = out["question_idx"].map(QUESTION_TYPE_MAP)
    out = out.dropna(subset=["question_type"])
    agg = (
        out.groupby(["model", "condition", "domain", "question_type"])["score_total"]
        .mean().round(3).reset_index()
        .rename(columns={"score_total": "avg_score"})
    )
    agg["question_type"] = pd.Categorical(
        agg["question_type"], categories=QUESTION_TYPE_ORDER, ordered=True
    )
    return agg.sort_values(["question_type", "domain", "model", "condition"]).reset_index(drop=True)


# -- plotting --

def save_fig(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def cond_label(cond):
    return "No RAG" if cond == "no_rag" else "RAG"


def domain_panels(fig, axes, domains, get_data_fn, ylabel, ylim):
    # shared layout for all 3-domain bar figures:
    # model names under every panel, no bar annotations, legend once, domain label inside
    legend_drawn = False
    for idx, (ax, domain) in enumerate(zip(axes, domains)):
        models, bar_groups = get_data_fn(domain)
        x     = np.arange(len(models))
        width = 0.38
        for i, (cond, scores) in enumerate(bar_groups):
            ax.bar(x + (i - 0.5) * width, scores, width,
                   label=cond_label(cond), color=CONDITION_COLORS[cond], zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=40, ha="right", fontsize=12)
        ax.set_ylim(0, ylim)
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=13)
        ax.text(0.50, 0.97, domain.capitalize(), transform=ax.transAxes,
                ha="center", va="top", fontsize=14, fontweight="bold",
                color=DOMAIN_COLORS.get(domain, "black"))
        if not legend_drawn:
            ax.legend(fontsize=12, loc="upper right")
            legend_drawn = True


def plot_rag_vs_norag(comparison):
    if comparison.empty or "rag_mean" not in comparison.columns:
        return
    comp   = comparison.sort_values("rag_mean", ascending=True)
    models = comp["model"].values
    y      = np.arange(len(models))
    height = 0.38

    fig, ax = plt.subplots(figsize=(12, max(5.5, len(models) * 0.70)))

    no_rag_err = np.array([comp["no_rag_mean"] - comp["no_rag_ci_lo"],
                           comp["no_rag_ci_hi"] - comp["no_rag_mean"]])
    rag_err    = np.array([comp["rag_mean"]    - comp["rag_ci_lo"],
                           comp["rag_ci_hi"]   - comp["rag_mean"]])

    ax.barh(y - height / 2, comp["no_rag_mean"], height, xerr=no_rag_err,
            label="No RAG", color=CONDITION_COLORS["no_rag"],
            error_kw={"elinewidth": 1.5, "capsize": 4, "alpha": 0.8})
    ax.barh(y + height / 2, comp["rag_mean"], height, xerr=rag_err,
            label="RAG", color=CONDITION_COLORS["rag"],
            error_kw={"elinewidth": 1.5, "capsize": 4, "alpha": 0.8})

    # labels go after the CI whisker so they don't overlap the error bar line
    for i, (n, r, n_hi, r_hi) in enumerate(zip(
            comp["no_rag_mean"], comp["rag_mean"],
            comp["no_rag_ci_hi"], comp["rag_ci_hi"])):
        ax.text(n_hi + 0.18, i - height / 2, f"{n:.2f}", va="center", fontsize=11)
        ax.text(r_hi + 0.18, i + height / 2, f"{r:.2f}", va="center", fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=13)
    ax.set_xlim(0, SCORE_MAX + 2.2)
    ax.set_xlabel(f"Average Score (max {SCORE_MAX}), 95% bootstrap CI")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    save_fig(fig, "COMPARISON_RAG_VS_NORAG.png")


def plot_global_benchmark(comparison):
    if comparison.empty:
        return
    # same model order in both panels, sorted by RAG score
    comp   = comparison.sort_values("rag_mean", ascending=True).reset_index(drop=True)
    models = comp["model"].values
    y      = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5.5, len(comp) * 0.70)))

    for ax, cond in zip(axes, ["no_rag", "rag"]):
        col_mean, col_lo, col_hi = f"{cond}_mean", f"{cond}_ci_lo", f"{cond}_ci_hi"
        err = np.array([comp[col_mean] - comp[col_lo], comp[col_hi] - comp[col_mean]])
        ax.barh(y, comp[col_mean], 0.55, xerr=err,
                color=CONDITION_COLORS[cond],
                error_kw={"elinewidth": 1.5, "capsize": 4, "alpha": 0.8})
        for i, (s, s_hi) in enumerate(zip(comp[col_mean], comp[col_hi])):
            ax.text(s_hi + 0.18, i, f"{s:.2f}", va="center", fontsize=11)
        ax.set_xlim(0, SCORE_MAX + 2.2)
        ax.set_xlabel(f"Average Score (max {SCORE_MAX})")
        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=13)
        ax.text(0.97, 0.03, cond_label(cond), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=14, fontweight="bold",
                color=CONDITION_COLORS[cond])

    plt.tight_layout()
    save_fig(fig, "GLOBAL_BENCHMARK_PLOT.png")


def plot_paired_delta(wilcoxon_df):
    if wilcoxon_df.empty:
        return
    wdf = wilcoxon_df.sort_values("delta", ascending=True)
    colors = []
    for _, row in wdf.iterrows():
        if row["significant"] and row["delta"] > 0:   colors.append(SIG_POS)
        elif row["significant"] and row["delta"] < 0: colors.append(SIG_NEG)
        else:                                          colors.append(SIG_NS)

    fig, ax = plt.subplots(figsize=(12, max(5.5, len(wdf) * 0.70)))
    bars = ax.barh(wdf["model"], wdf["delta"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.9)

    for bar, (_, row) in zip(bars, wdf.iterrows()):
        x = bar.get_width()
        offset, ha = (0.05, "left") if x >= 0 else (-0.05, "right")
        ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                f"d={x:+.2f}   Cohen d={row['cohens_d']:+.2f}   p={row['p_value']:.3f}",
                va="center", ha=ha, fontsize=10)

    ax.set_xlabel("Score Change (RAG - No-RAG)")
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(handles=[
        Patch(color=SIG_POS, label=f"RAG helps (p < {ALPHA})"),
        Patch(color=SIG_NEG, label=f"RAG hurts (p < {ALPHA})"),
        Patch(color=SIG_NS,  label="not significant"),
    ], loc="lower right", frameon=True)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin - (xmax - xmin) * 0.42, xmax + (xmax - xmin) * 0.42)
    plt.tight_layout()
    save_fig(fig, "WILCOXON_DELTA_PLOT.png")


def plot_win_rate(wilcoxon_df):
    if wilcoxon_df.empty or "win_rate" not in wilcoxon_df.columns:
        return
    wdf   = wilcoxon_df.sort_values("win_rate", ascending=True)
    y_pos = np.arange(len(wdf))
    fig, ax = plt.subplots(figsize=(11, max(5.5, len(wdf) * 0.70)))
    ax.barh(y_pos, wdf["win_rate"].values * 100, color=CONDITION_COLORS["rag"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(wdf["model"].values, fontsize=13)
    ax.axvline(50, color="black", linestyle="--", linewidth=1.0, label="50% baseline")
    for i, (_, row) in enumerate(wdf.iterrows()):
        ax.text(row["win_rate"] * 100 + 1.2, i,
                f"{row['wins']}W / {row['losses']}L / {row['ties']}T", va="center", fontsize=10)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Questions where RAG scored higher than No-RAG (%)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_fig(fig, "RAG_WIN_RATE.png")


def plot_domain_breakdown(breakdown):
    if breakdown.empty:
        return
    domains = sorted(breakdown["domain"].unique())
    fig, axes = plt.subplots(1, len(domains),
                             figsize=(5.5 * len(domains) + 1, 7.5), sharey=True)
    if len(domains) == 1:
        axes = [axes]

    def get_data(domain):
        dom_df  = breakdown[breakdown["domain"] == domain]
        models  = sorted(dom_df["model"].unique())
        groups  = []
        for cond in ["no_rag", "rag"]:
            cdf    = dom_df[dom_df["condition"] == cond].set_index("model")
            scores = [float(cdf.loc[m, "avg_score"]) if m in cdf.index else float("nan") for m in models]
            groups.append((cond, scores))
        return models, groups

    domain_panels(fig, axes, domains, get_data, f"Avg Score (max {SCORE_MAX})", SCORE_MAX + 1.5)
    plt.tight_layout()
    save_fig(fig, "COMPARISON_BY_DOMAIN.png")


def plot_per_domain_heatmap(per_domain_wilcoxon):
    if per_domain_wilcoxon.empty:
        return
    pivot_delta = per_domain_wilcoxon.pivot(index="model", columns="domain", values="delta")
    pivot_sig   = per_domain_wilcoxon.pivot(index="model", columns="domain", values="significant")

    ncols, nrows = len(pivot_delta.columns), len(pivot_delta.index)
    fig, ax = plt.subplots(figsize=(2.2 * ncols + 4, 0.68 * nrows + 2.5))
    vmax = max(abs(pivot_delta.min().min()), abs(pivot_delta.max().max()), 0.5)
    im   = ax.imshow(pivot_delta.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(ncols))
    ax.set_xticklabels([c.capitalize() for c in pivot_delta.columns], fontsize=13)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(pivot_delta.index, fontsize=13)

    for i in range(nrows):
        for j in range(ncols):
            val = pivot_delta.values[i, j]
            sig = pivot_sig.values[i, j] if not pd.isna(pivot_sig.values[i, j]) else False
            txt = "-" if pd.isna(val) else (f"{val:+.2f} *" if sig else f"{val:+.2f}")
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=12, color="black", fontweight="bold" if sig else "normal")

    cbar = fig.colorbar(im, ax=ax, shrink=0.80)
    cbar.set_label("Score change (RAG - No-RAG)", rotation=270, labelpad=18, fontsize=12)
    ax.set_xlabel(f"* = p < {ALPHA}", fontsize=11)
    ax.grid(False)
    plt.tight_layout()
    save_fig(fig, "PER_DOMAIN_DELTA_HEATMAP.png")


def plot_component_breakdown(comp_df):
    if comp_df.empty:
        return
    metrics = [c for c in ["score_grounded_accuracy", "score_completeness"] if c in comp_df.columns]
    if not metrics:
        return

    panel_labels = {
        "score_grounded_accuracy": "Grounded Accuracy (max 5)",
        "score_completeness":      "Completeness (max 5)",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 6.5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    legend_drawn = False
    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        pivot = comp_df.pivot(index="model", columns="condition", values=metric).sort_values(
            "rag" if "rag" in comp_df["condition"].unique() else metric, ascending=True
        )
        x, width = np.arange(len(pivot.index)), 0.38
        for i, cond in enumerate(["no_rag", "rag"]):
            if cond not in pivot.columns:
                continue
            ax.bar(x + (i - 0.5) * width, pivot[cond].values, width,
                   color=CONDITION_COLORS[cond], label=cond_label(cond))
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=40, ha="right", fontsize=12)
        ax.set_ylim(0, COMPONENT_MAX + 0.5)
        if idx == 0:
            ax.set_ylabel(f"Avg score (max {COMPONENT_MAX})")
        if not legend_drawn:
            ax.legend(fontsize=12, loc="upper right")
            legend_drawn = True
        ax.text(0.50, 0.97, panel_labels.get(metric, metric),
                transform=ax.transAxes, ha="center", va="top", fontsize=13, fontweight="bold")

    plt.tight_layout()
    save_fig(fig, "COMPONENT_BREAKDOWN.png")


def plot_document_grounding(grounding):
    if grounding.empty:
        return
    domains = sorted(grounding["domain"].unique())

    # use the first domain's sort order for all panels so models stay in the same position
    first_order = (grounding[grounding["domain"] == domains[0]]
                   .sort_values("avg_doc_grounding")["model"].tolist())
    y = np.arange(len(first_order))

    fig, axes = plt.subplots(1, len(domains),
                             figsize=(5.5 * len(domains), 6.5), sharey=True)
    if len(domains) == 1:
        axes = [axes]

    for idx, (ax, domain) in enumerate(zip(axes, domains)):
        dom_df = grounding[grounding["domain"] == domain].set_index("model")
        scores = [float(dom_df.loc[m, "avg_doc_grounding"]) if m in dom_df.index else 0.0
                  for m in first_order]

        ax.barh(y, scores, color=DOMAIN_COLORS.get(domain, "#888"))
        ax.set_xlim(0, 5.8)
        ax.set_xlabel("Avg Document Grounding Score (max 5)")
        for i, s in enumerate(scores):
            ax.text(s + 0.07, i, f"{s:.2f}", va="center", fontsize=11)

        ax.set_yticks(y)
        if idx == 0:
            ax.set_yticklabels(first_order, fontsize=12)
        else:
            ax.set_yticklabels([""] * len(first_order))

        ax.text(0.97, 0.03, domain.capitalize(), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=14, fontweight="bold",
                color=DOMAIN_COLORS.get(domain, "black"))

    plt.tight_layout()
    save_fig(fig, "COMPARISON_DOCUMENT_GROUNDING.png")


def plot_efficiency_frontier(frontier):
    if frontier.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    for condition, group in frontier.groupby("condition"):
        ax.scatter(group["avg_wall_time_s"], group["avg_score"],
                   label=cond_label(condition),
                   color=CONDITION_COLORS.get(condition, "gray"),
                   s=160, edgecolor="black", linewidth=0.8, zorder=3)
        for _, row in group.iterrows():
            ax.annotate(row["model"], (row["avg_wall_time_s"], row["avg_score"]),
                        textcoords="offset points", xytext=(8, 5), fontsize=11)

    # draw the Pareto frontier: points where no other model is both faster and higher scoring
    pts = frontier[["avg_wall_time_s", "avg_score"]].dropna().values
    if len(pts) > 1:
        order  = pts[pts[:, 0].argsort()]
        pareto = [order[0]]
        for p in order[1:]:
            if p[1] > pareto[-1][1]:
                pareto.append(p)
        pareto = np.array(pareto)
        ax.plot(pareto[:, 0], pareto[:, 1], "--", color="black",
                alpha=0.5, linewidth=1.5, label="Pareto frontier")

    ax.set_xlabel("Avg Response Time (s), slower to the right")
    ax.set_ylabel(f"Avg Score (max {SCORE_MAX}), better upward")
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_fig(fig, "EFFICIENCY_FRONTIER_PLOT.png")


def plot_question_type_comparisons(qtype_df):
    if qtype_df.empty:
        return
    domains = sorted(qtype_df["domain"].unique())

    for qtype in QUESTION_TYPE_ORDER:
        type_df = qtype_df[qtype_df["question_type"] == qtype]
        if type_df.empty:
            continue

        fig, axes = plt.subplots(1, len(domains),
                                 figsize=(5.5 * len(domains) + 1, 7.5), sharey=True)
        if len(domains) == 1:
            axes = [axes]

        def get_data(domain, _df=type_df):
            dom_df = _df[_df["domain"] == domain]
            models = sorted(dom_df["model"].unique())
            groups = []
            for cond in ["no_rag", "rag"]:
                cdf    = dom_df[dom_df["condition"] == cond].set_index("model")
                scores = [float(cdf.loc[m, "avg_score"]) if m in cdf.index else float("nan")
                          for m in models]
                groups.append((cond, scores))
            return models, groups

        domain_panels(fig, axes, domains, get_data, f"Avg Score (max {SCORE_MAX})", SCORE_MAX + 1.5)
        plt.tight_layout()
        save_fig(fig, f"QTYPE_{qtype.split(' /')[0].upper().replace(' ', '_')}.png")


def plot_resource_usage(df):
    needed = ["wall_time_s", "tokens_per_second", "cpu_percent_avg"]
    if not all(c in df.columns for c in needed):
        print("resource usage columns missing - skipping.")
        return

    agg    = df.groupby(["model", "condition"])[needed].mean().reset_index()
    # sort models by avg response time (no_rag) descending so slowest is at top
    order  = (agg[agg["condition"] == "no_rag"]
              .sort_values("wall_time_s", ascending=True)["model"].tolist())
    y      = np.arange(len(order))
    height = 0.38

    fig, (ax_time, ax_tps) = plt.subplots(1, 2, figsize=(14, 6))

    for i, condition in enumerate(["no_rag", "rag"]):
        cdf    = agg[agg["condition"] == condition].set_index("model")
        offset = (i - 0.5) * height

        times  = [float(cdf.loc[m, "wall_time_s"])      if m in cdf.index else 0.0 for m in order]
        tps    = [float(cdf.loc[m, "tokens_per_second"]) if m in cdf.index else 0.0 for m in order]
        cpus   = [float(cdf.loc[m, "cpu_percent_avg"])   if m in cdf.index else 0.0 for m in order]

        # left panel: response time, CPU% as text annotation at bar end
        ax_time.barh(y + offset, times, height,
                     label=cond_label(condition), color=CONDITION_COLORS[condition])
        for j, (t, cpu) in enumerate(zip(times, cpus)):
            ax_time.text(t + 0.25, y[j] + offset, f"{t:.1f}s  ({cpu:.0f}%)",
                         va="center", fontsize=10, color="#444")

        # right panel: tokens per second
        ax_tps.barh(y + offset, tps, height,
                    label=cond_label(condition), color=CONDITION_COLORS[condition])
        for j, v in enumerate(tps):
            ax_tps.text(v + 0.2, y[j] + offset, f"{v:.1f}", va="center", fontsize=10)

    for ax, xlabel in [(ax_time, "Response Time (s)  —  CPU% shown in label"),
                       (ax_tps,  "Tokens per Second")]:
        ax.set_yticks(y)
        ax.set_xlabel(xlabel, fontsize=13)

    ax_time.set_yticklabels(order, fontsize=13)
    ax_tps.set_yticklabels([""] * len(order))

    ax_time.set_xlim(0, agg["wall_time_s"].max() * 1.45)
    ax_tps.set_xlim(0, agg["tokens_per_second"].max() * 1.25)

    ax_time.legend(fontsize=12, loc="lower right")

    plt.tight_layout()
    save_fig(fig, "RESOURCE_USAGE_PLOT.png")


# -- summary report --

def write_summary_report(df, comparison, breakdown, wilcoxon_global, wilcoxon_per_domain,
                         grounding, frontier):
    lines = []
    L = lines.append
    L("# Benchmark Summary Report\n")
    L(f"_Total rows: **{len(df)}** across {df['model'].nunique()} models, "
      f"{df['condition'].nunique()} conditions, {df['domain'].nunique()} domains._\n")

    L("## Headline numbers\n")
    if not comparison.empty and "delta" in comparison.columns:
        best         = comparison.loc[comparison["rag_mean"].idxmax()]
        biggest_gain = comparison.loc[comparison["delta"].idxmax()]
        biggest_loss = comparison.loc[comparison["delta"].idxmin()]
        L(f"- **Highest RAG score:** `{best['model']}` at {best['rag_mean']:.2f} / {SCORE_MAX} "
          f"(95% CI [{best['rag_ci_lo']:.2f}, {best['rag_ci_hi']:.2f}]).")
        L(f"- **Biggest RAG gain:** `{biggest_gain['model']}` "
          f"({biggest_gain['no_rag_mean']:.2f} -> {biggest_gain['rag_mean']:.2f}, "
          f"delta = {biggest_gain['delta']:+.2f}).")
        if biggest_loss["delta"] < 0:
            L(f"- **Biggest RAG drop:** `{biggest_loss['model']}` "
              f"({biggest_loss['no_rag_mean']:.2f} -> {biggest_loss['rag_mean']:.2f}, "
              f"delta = {biggest_loss['delta']:+.2f}).")
        L("")

    L("## Wilcoxon results\n")
    if not wilcoxon_global.empty:
        helps = wilcoxon_global[wilcoxon_global["verdict"] == "RAG helps"]["model"].tolist()
        hurts = wilcoxon_global[wilcoxon_global["verdict"] == "RAG hurts"]["model"].tolist()
        ns    = wilcoxon_global[wilcoxon_global["verdict"] == "no significant effect"]["model"].tolist()
        L(f"- RAG helps ({len(helps)}): {', '.join(f'`{m}`' for m in helps) if helps else 'none'}")
        L(f"- RAG hurts ({len(hurts)}): {', '.join(f'`{m}`' for m in hurts) if hurts else 'none'}")
        L(f"- No effect ({len(ns)}): {', '.join(f'`{m}`' for m in ns) if ns else 'none'}\n")
        for _, row in wilcoxon_global.sort_values("cohens_d", ascending=False).iterrows():
            L(f"  - `{row['model']}`: d={row['cohens_d']:+.2f} ({row['effect_size']}), "
              f"win rate {row['win_rate']*100:.0f}% ({row['wins']}W/{row['losses']}L/{row['ties']}T), "
              f"p={row['p_value']:.3f}")
        L("")

    L("## Per-domain\n")
    if not wilcoxon_per_domain.empty:
        for domain in sorted(wilcoxon_per_domain["domain"].unique()):
            sub = wilcoxon_per_domain[wilcoxon_per_domain["domain"] == domain]
            L(f"- **{domain.capitalize()}**: mean delta = {sub['delta'].mean():+.2f}, "
              f"{(sub['verdict']=='RAG helps').sum()} helped, {(sub['verdict']=='RAG hurts').sum()} hurt.")
        L("")

    if not grounding.empty:
        L("## Top grounding scores\n")
        for _, row in grounding.sort_values("avg_doc_grounding", ascending=False).head(3).iterrows():
            L(f"  - `{row['model']}` on {row['domain']}: {row['avg_doc_grounding']:.2f} / 5")
        L("")

    if not frontier.empty:
        L("## Efficiency\n")
        for _, row in frontier.head(3).iterrows():
            L(f"  - `{row['model']}` ({row['condition']}): {row['avg_score']:.2f} pts "
              f"in {row['avg_wall_time_s']:.1f}s -> {row['score_per_second']:.3f} pts/s")
        L("")

    path = os.path.join(RESULTS_DIR, "SUMMARY_REPORT.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {path}")


# -- main --

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading results...")
    df = load_all_results()
    if df.empty:
        return

    print(f"\nModels    : {sorted(df['model'].unique())}")
    print(f"Conditions: {sorted(df['condition'].unique())}")
    print(f"Domains   : {sorted(df['domain'].unique())}")

    comparison = rag_vs_norag(df)
    comparison.to_csv(os.path.join(RESULTS_DIR, "COMPARISON_RAG_VS_NORAG.csv"), index=False)

    breakdown = domain_breakdown(df)
    breakdown.to_csv(os.path.join(RESULTS_DIR, "COMPARISON_BY_DOMAIN.csv"), index=False)

    comp_df = component_breakdown(df)
    if not comp_df.empty:
        comp_df.to_csv(os.path.join(RESULTS_DIR, "COMPONENT_BREAKDOWN.csv"), index=False)

    grounding = document_grounding_summary(df)
    if not grounding.empty:
        grounding.to_csv(os.path.join(RESULTS_DIR, "COMPARISON_DOCUMENT_GROUNDING.csv"), index=False)

    wilcoxon_global = wilcoxon_rag_effect(df, by_domain=False)
    if not wilcoxon_global.empty:
        wilcoxon_global.to_csv(os.path.join(RESULTS_DIR, "WILCOXON_RAG_EFFECT.csv"), index=False)

    wilcoxon_per_domain = wilcoxon_rag_effect(df, by_domain=True)
    if not wilcoxon_per_domain.empty:
        wilcoxon_per_domain.to_csv(os.path.join(RESULTS_DIR, "WILCOXON_PER_DOMAIN.csv"), index=False)

    frontier = efficiency_frontier(df)
    if not frontier.empty:
        frontier.to_csv(os.path.join(RESULTS_DIR, "EFFICIENCY_FRONTIER.csv"), index=False)

    qtype_df = question_type_breakdown(df)
    if not qtype_df.empty:
        qtype_df.to_csv(os.path.join(RESULTS_DIR, "COMPARISON_BY_QUESTION_TYPE.csv"), index=False)

    print("\nGenerating plots...")
    plot_rag_vs_norag(comparison)
    plot_global_benchmark(comparison)
    plot_paired_delta(wilcoxon_global)
    plot_win_rate(wilcoxon_global)
    plot_domain_breakdown(breakdown)
    plot_per_domain_heatmap(wilcoxon_per_domain)
    plot_component_breakdown(comp_df)
    plot_document_grounding(grounding)
    plot_efficiency_frontier(frontier)
    plot_question_type_comparisons(qtype_df)
    plot_resource_usage(df)

    write_summary_report(df, comparison, breakdown, wilcoxon_global,
                         wilcoxon_per_domain, grounding, frontier)
    print("\nDone.")


if __name__ == "__main__":
    main()
