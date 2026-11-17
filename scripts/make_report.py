"""Phase 7: build ``reports/report.md`` and figures from files in ``results/`` only.

Every number in the report is read from a results file. The only fixed text is definitions,
the list of limitations and reproduction commands. A section whose inputs are missing says so
instead of inventing content.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from piiguard.utils import ensure_dir, get_logger, load_config  # noqa: E402

log = get_logger("make_report")

# Validated categorical palette, slots 1-4 in fixed order (see the dataviz skill).
COLORS = {
    "deberta": "#2a78d6",
    "qwen_lora": "#eb6834",
    "smollm_lora": "#1baf7a",
    "qwen_zeroshot": "#eda100",
}
NAMES = {
    "deberta": "DeBERTa-v3-small",
    "qwen_lora": "Qwen2.5-0.5B + LoRA",
    "smollm_lora": "SmolLM2-360M + LoRA",
    "qwen_zeroshot": "Qwen2.5-0.5B zero-shot",
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SET_ORDER = [
    "clean",
    "obf_email",
    "obf_phone",
    "obf_name",
    "obf_id",
    "obf_all",
    "rare",
    "cross_de",
    "cross_fr",
    "cross_it",
    "context_only",
]


def style(ax: Any) -> None:
    """Recessive grid and spines, text in ink tokens."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def new_fig(w: float = 9, h: float = 4.6) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    style(ax)
    return fig, ax


def md_table(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    """Markdown table; ``fmt`` maps a column to a format string, e.g. ``'{:.3f}'``."""
    fmt = fmt or {}
    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float | np.floating) and np.isnan(v):
                cells.append("–")
            elif c in fmt and not isinstance(v, str):
                cells.append(fmt[c].format(v))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def pm(df: pd.DataFrame, col: str, digits: int = 3) -> pd.Series:
    """``mean ± std`` strings from ``<col>_mean`` / ``<col>_std``."""
    return (
        df[f"{col}_mean"].map(lambda v: f"{v:.{digits}f}")
        + " ± "
        + df[f"{col}_std"].fillna(0).map(lambda v: f"{v:.{digits}f}")
    )


def ci_str(r: pd.Series, m: str, digits: int = 3) -> str:
    return f"{r[m]:.{digits}f} [{r[m + '_lo']:.{digits}f}, {r[m + '_hi']:.{digits}f}]"


# ---------------------------------------------------------------------------------- gate
def gate_section(res: Path, fig_dir: Path) -> tuple[str, list[str]]:
    p = res / "gate" / "summary.csv"
    if not p.exists():
        return "## Stage 1: PII gate\n\n_Gate results not found in `results/gate/`._\n", []
    g = pd.read_csv(p)
    feat = json.loads((res / "gate" / "summary_features.json").read_text())
    bits: list[str] = []
    t = g.assign(
        ACC=pm(g, "accuracy"),
        AUC=pm(g, "roc_auc"),
        PR_AUC=pm(g, "pr_auc"),
        R95=pm(g, "recall_at_95p", 2),
        train_s=g["train_s_mean"].map("{:.3g}".format),
        lat_ms=g["latency_ms_single_mean"].map("{:.3g}".format),
        params=g["params_mean"].map(lambda v: f"{v:,.0f}"),
    )[
        [
            "model",
            "setting",
            "size",
            "n_seeds",
            "ACC",
            "AUC",
            "PR_AUC",
            "R95",
            "train_s",
            "lat_ms",
            "params",
        ]
    ]
    t = t.rename(
        columns={
            "ACC": "accuracy",
            "R95": "recall@95%prec",
            "train_s": "train s",
            "lat_ms": "latency ms/sent",
        }
    )
    out = ["## Stage 1: PII gate (quantum vs classical)\n"]
    out.append(
        f"All 8-feature models see identical inputs: MiniLM-L6-v2 embeddings reduced by PCA to "
        f"{feat['n_features']} dimensions, which keeps **{100 * feat['pca_explained_variance_total']:.1f}%** "
        "of the embedding variance. Mean ± std over seeds; evaluation on the shared balanced test "
        "subset (`n2000`/`full` = training-set size; `n500` = kernel train cap). "
        "Latency and training time are simulator wall-clock numbers, not hardware numbers.\n"
    )
    out.append(md_table(t))
    key = g[(g.model == "vqc") & (g.setting == "exact")]
    ref = g[(g.model == "logreg") & (g["size"] == "n2000")]
    mlp = g[(g.model == "mlp") & (g["size"] == "n2000")]
    ft = g[(g.model == "minilm_finetuned") & (g["size"] == "full")]
    if len(key) and len(ref):
        d = key.roc_auc_mean.iloc[0] - ref.roc_auc_mean.iloc[0]
        bits.append(
            f"Gate: VQC (exact) ROC-AUC {key.roc_auc_mean.iloc[0]:.3f} vs logistic regression "
            f"{ref.roc_auc_mean.iloc[0]:.3f} (difference {d:+.3f}); VQC trains in "
            f"{key.train_s_mean.iloc[0]:.1f} s vs {ref.train_s_mean.iloc[0]:.3f} s."
        )
    ref_full = g[(g.model == "logreg") & (g["size"] == "full")]
    if len(ft) and len(ref_full):
        bits.append(
            f"Gate: fine-tuned MiniLM without the PCA bottleneck reaches ROC-AUC "
            f"{ft.roc_auc_mean.iloc[0]:.3f}, against {ref_full.roc_auc_mean.iloc[0]:.3f} for "
            "the 8-feature logistic regression (both trained on the full training pool)."
        )
    if len(mlp) and len(key):
        out.append(
            f"\nThe 8-feature models span ROC-AUC {g[g['size'].isin(['n2000', 'n500', 'full']) & (g.model != 'minilm_finetuned')].roc_auc_mean.min():.3f}"
            f"–{g[g.model != 'minilm_finetuned'].roc_auc_mean.max():.3f}; the fine-tuned MiniLM is the only model "
            "outside that range.\n"
        )
    # barren plateau
    bp = res / "gate" / "summary_barren.csv"
    if bp.exists():
        b = pd.read_csv(bp)
        out.append("### Barren-plateau check\n")
        out.append("Variance of ∂⟨Z₀⟩/∂θ over random initialisations (3 layers):\n")
        out.append(md_table(b, {"var_first_param": "{:.2e}", "var_mean_all_params": "{:.2e}"}))
        r = b.var_mean_all_params.iloc[0] / b.var_mean_all_params.iloc[-1]
        bits.append(
            f"Barren plateau: mean gradient variance falls {r:.0f}x from {int(b.n_qubits.iloc[0])} to {int(b.n_qubits.iloc[-1])} qubits."
        )
        fig, ax = new_fig(6.5, 4)
        ax.plot(
            b.n_qubits,
            b.var_mean_all_params,
            color=COLORS["deberta"],
            marker="o",
            lw=2,
            ms=7,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        ax.set_yscale("log")
        ax.set_xticks(b.n_qubits)
        ax.set_xlabel("Qubits")
        ax.set_ylabel("Gradient variance (log)")
        ax.set_title("Gradient variance shrinks with qubit count", loc="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / "barren_plateau.png", dpi=160, facecolor=SURFACE)
        plt.close(fig)
        out.append("\n![barren plateau](figures/barren_plateau.png)\n")
    # AUC vs latency: one point per model/setting, chosen to keep labels readable
    keep = {
        ("logreg", "classical", "n2000"): ("logreg", (8, -12), "left"),
        ("mlp", "classical", "n2000"): ("mlp", (8, 6), "left"),
        ("vqc", "exact", "n2000"): ("vqc exact", (8, 8), "left"),
        ("vqc", "noisy_p0.01", "n2000"): ("vqc noisy p=0.01", (-8, 8), "right"),
        ("qkernel_angle", "exact", "n500"): ("kernel (angle)", (-8, 8), "right"),
        ("qkernel_iqp", "exact", "n500"): ("kernel (IQP)", (8, -3), "left"),
        ("minilm_finetuned", "raw_text", "full"): ("MiniLM fine-tuned (no PCA)", (8, -3), "left"),
    }
    fig, ax = new_fig(9, 4.8)
    for (m, st, sz), (label, off, ha) in keep.items():
        r = g[(g.model == m) & (g.setting == st) & (g["size"] == sz)]
        if r.empty:
            continue
        r = r.iloc[0]
        q = m.startswith(("vqc", "qkernel"))
        ax.scatter(
            r.latency_ms_single_mean,
            r.roc_auc_mean,
            s=70,
            zorder=3,
            color=COLORS["qwen_lora"] if q else COLORS["deberta"],
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.annotate(
            label,
            (r.latency_ms_single_mean, r.roc_auc_mean),
            textcoords="offset points",
            xytext=off,
            ha=ha,
            fontsize=8.5,
            color=MUTED,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Inference latency, ms per sentence (log)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        "Gate accuracy vs cost (orange = quantum simulation, blue = classical)",
        loc="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "gate_auc_vs_latency.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    out.append("\n![gate accuracy vs cost](figures/gate_auc_vs_latency.png)\n")
    return "\n".join(out), bits


# ---------------------------------------------------------------------------------- extras
VARIANT_COLOR = {
    "logreg": COLORS["deberta"],
    "mlp": COLORS["smollm_lora"],
    "vqc": COLORS["qwen_lora"],
    "qkernel_angle": COLORS["qwen_zeroshot"],
}
VARIANT_NAME = {
    "logreg": "Logistic regression",
    "mlp": "MLP",
    "vqc": "VQC",
    "qkernel_angle": "Quantum kernel SVM",
}


def line_chart(d: pd.DataFrame, xlabel: str, title: str, path: Path, logx: bool) -> None:
    """ROC-AUC vs x with a ±1 std band per variant and direct end labels."""
    fig, ax = new_fig(8.6, 4.8)
    ends: list[tuple[float, float, str]] = []
    for v, name in VARIANT_NAME.items():
        s = d[d.variant == v].sort_values("x")
        if s.empty:
            continue
        sd = s.roc_auc_std.fillna(0)
        ax.fill_between(
            s.x, s.roc_auc_mean - sd, s.roc_auc_mean + sd, color=VARIANT_COLOR[v], alpha=0.15, lw=0
        )
        ax.plot(
            s.x,
            s.roc_auc_mean,
            color=VARIANT_COLOR[v],
            lw=2,
            marker="o",
            ms=6,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=name,
        )
        ends.append((s.roc_auc_mean.iloc[-1], s.x.iloc[-1], name))
    if logx:
        ax.set_xscale("log")
    # direct labels at the line ends, nudged apart so they never overlap
    lo, hi = ax.get_ylim()
    gap, prev = 0.05 * (hi - lo), -np.inf
    for y, x, name in sorted(ends):
        y_lab = max(y, prev + gap)
        prev = y_lab
        ax.annotate(
            name,
            (x, y),
            xytext=(x * 1.04 if logx else x, y_lab),
            textcoords="data",
            fontsize=8.5,
            color=MUTED,
            va="center",
            arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.8, "shrinkA": 0, "shrinkB": 3},
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ROC-AUC (shared test subset)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.margins(x=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def extras_section(res: Path, fig_dir: Path) -> tuple[str, list[str]]:
    p = res / "gate_extras" / "summary.csv"
    if not p.exists():
        return (
            "## Additional quantum scenarios\n\n_Not run (`results/gate_extras/` missing)._\n",
            [],
        )
    d = pd.read_csv(p)
    bits: list[str] = []
    out = ["## Additional quantum scenarios (exact simulation)\n"]
    out.append(
        "Mean ± std over seeds on the shared balanced test subset; bands in the figures are ±1 std.\n"
    )
    ld = d[d.scenario == "low_data"]
    if not ld.empty:
        out.append("### Low-data regime\n")
        piv = ld.assign(
            v=ld.roc_auc_mean.map("{:.3f}".format)
            + " ± "
            + ld.roc_auc_std.fillna(0).map("{:.3f}".format)
        ).pivot(index="x", columns="variant", values="v")
        piv.insert(0, "train sentences", piv.index.astype(int))
        out.append(md_table(piv.reset_index(drop=True).fillna("–")))
        line_chart(
            ld,
            "Training sentences (log)",
            "Learning curves on 8 features",
            fig_dir / "low_data.png",
            True,
        )
        out.append("![low data](figures/low_data.png)\n")
        n0 = ld.x.min()
        row = ld[ld.x == n0].set_index("variant").roc_auc_mean
        bits.append(
            "Low data (n="
            + f"{int(n0)}"
            + "): "
            + ", ".join(f"{VARIANT_NAME[k]} {v:.3f}" for k, v in row.items() if k in VARIANT_NAME)
            + " ROC-AUC."
        )
    ab = d[d.scenario == "ablation"]
    if not ab.empty:
        out.append("### VQC ablation: depth and data re-uploading\n")
        t = ab.assign(
            **{
                "ROC-AUC": ab.roc_auc_mean.map("{:.3f}".format)
                + " ± "
                + ab.roc_auc_std.fillna(0).map("{:.3f}".format),
                "recall@95%prec": ab.recall_at_95p_mean.map("{:.2f}".format),
                "train s": ab.train_s_mean.map("{:.1f}".format),
                "params": ab.params.map(lambda v: "–" if pd.isna(v) else f"{v:,.0f}"),
            }
        )
        out.append(
            md_table(t[["variant", "n_seeds", "ROC-AUC", "recall@95%prec", "train s", "params"]])
        )
        best = ab.loc[ab.roc_auc_mean.idxmax()]
        bits.append(
            f"VQC ablation: best variant `{best.variant}` with ROC-AUC {best.roc_auc_mean:.3f}."
        )
    fs = d[d.scenario == "feature_sweep"]
    if not fs.empty:
        out.append("### Feature-count sweep\n")
        out.append(
            "The VQC uses k qubits for k features; classical models can use more features. "
            "`PCA variance kept` is the share of embedding variance retained.\n"
        )
        piv = fs.assign(v=fs.roc_auc_mean.map("{:.3f}".format)).pivot(
            index="x", columns="variant", values="v"
        )
        kept = fs.groupby("x").pca_variance_kept.first().map("{:.1%}".format)
        piv.insert(0, "PCA variance kept", kept)
        piv.insert(0, "features k", piv.index.astype(int))
        out.append(md_table(piv.reset_index(drop=True).fillna("–")))
        line_chart(
            fs,
            "Number of PCA features / qubits k (log)",
            "Gate accuracy vs feature count",
            fig_dir / "feature_sweep.png",
            True,
        )
        out.append("![feature sweep](figures/feature_sweep.png)\n")
        lr = fs[fs.variant == "logreg"].sort_values("x")
        if len(lr) > 1:
            bits.append(
                f"Feature sweep: logistic regression goes from ROC-AUC {lr.roc_auc_mean.iloc[0]:.3f} at k={int(lr.x.iloc[0])} to {lr.roc_auc_mean.iloc[-1]:.3f} at k={int(lr.x.iloc[-1])}."
            )
    return "\n".join(out), bits


# ---------------------------------------------------------------------------------- masker
def masker_section(res: Path, fig_dir: Path) -> tuple[str, list[str]]:
    p = res / "eval" / "masker_ci.csv"
    if not p.exists():
        return (
            "## Stage 2: maskers\n\n_Run `make eval` first (`results/eval/masker_ci.csv` missing)._\n",
            [],
        )
    ci = pd.read_csv(p)
    bits: list[str] = []
    out = ["## Stage 2: PII maskers\n"]
    out.append(
        "Entity F1 is strict (exact span and label). **Character leak rate** is the share of gold "
        "PII characters not covered by any predicted mask, whatever its type. Values are "
        "seed-averaged with 95% bootstrap CIs over documents (seeds share the resample). "
        "`n_seeds` is the number of training seeds behind each row.\n"
    )
    for metric, title in (("entity_f1", "Entity F1"), ("char_leak_rate", "Character leak rate")):
        piv = {}
        for _, r in ci.iterrows():
            piv.setdefault(r["set"], {})[NAMES.get(r.model, r.model)] = ci_str(r, metric)
        df = pd.DataFrame(piv).T.reindex([s for s in SET_ORDER if s in piv])
        df.insert(0, "set", df.index)
        out.append(f"### {title}\n")
        out.append(md_table(df.reset_index(drop=True)))
    seeds = ci.groupby("model").n_seeds.first().to_dict()
    out.append(
        "Seeds per model: "
        + ", ".join(f"{NAMES.get(k, k)} = {v}" for k, v in seeds.items())
        + ".\n"
    )

    # F1-hides-leaks bits
    cl = ci[ci["set"] == "clean"].set_index("model")
    for m in ("deberta", "qwen_lora"):
        if m in cl.index:
            sub = ci[(ci.model == m) & (ci["set"] != "clean") & ci["set"].isin(SET_ORDER[1:10])]
            w = sub.loc[sub.entity_f1.idxmin()]
            bits.append(
                f"{NAMES[m]}: clean entity F1 {cl.loc[m].entity_f1:.3f}, character leak "
                f"{cl.loc[m].char_leak_rate:.3f}; worst stress-set F1 is {w.entity_f1:.3f} on "
                f"`{w['set']}` (leak {w.char_leak_rate:.3f})."
            )
    # figures
    trained = ["deberta", "qwen_lora", "smollm_lora"]
    sets = [s for s in SET_ORDER if s in set(ci["set"]) and s != "context_only"]
    fig, ax = new_fig(11, 4.8)
    w = 0.26
    for i, m in enumerate(trained):
        d = ci[ci.model == m].set_index("set").reindex(sets)
        x = np.arange(len(sets)) + (i - 1) * w
        err = np.array(
            [d.char_leak_rate - d.char_leak_rate_lo, d.char_leak_rate_hi - d.char_leak_rate]
        )
        ax.bar(x, d.char_leak_rate * 100, w * 0.86, color=COLORS[m], label=NAMES[m], zorder=3)
        ax.errorbar(
            x,
            d.char_leak_rate * 100,
            yerr=err * 100,
            fmt="none",
            ecolor=INK,
            elinewidth=1,
            capsize=2,
            zorder=4,
        )
    ax.set_xticks(np.arange(len(sets)))
    ax.set_xticklabels(sets, rotation=25, ha="right")
    ax.set_ylabel("Character leak rate (%)")
    ax.set_title("Share of PII characters left unmasked (95% CI)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "masker_leak.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    out.append("![masker leak](figures/masker_leak.png)\n")
    fig, ax = new_fig(8.4, 5)
    for m in trained:
        d = ci[(ci.model == m) & ci["set"].isin(sets)]
        ax.scatter(
            d.entity_f1,
            d.char_leak_rate * 100,
            s=64,
            color=COLORS[m],
            label=NAMES[m],
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
        )
    ax.set_xlabel("Entity F1 (strict)")
    ax.set_ylabel("Character leak rate (%)")
    ax.set_title("F1 and leak rate disagree: one point per evaluation set", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "f1_vs_leak.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    out.append("![F1 vs leak](figures/f1_vs_leak.png)\n")

    # stress deltas
    dp = res / "eval" / "masker_delta_vs_clean.csv"
    if dp.exists():
        d = pd.read_csv(dp)
        d = d[d.metric.isin(["entity_f1", "char_leak_rate"]) & d.model.isin(trained)]
        d["model"] = d.model.map(NAMES)
        d["diff vs clean [95% CI]"] = d.apply(
            lambda r: f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", axis=1
        )
        d = d[["model", "set", "metric", "diff vs clean [95% CI]", "p"]]
        out.append("### Change relative to the clean test set\n")
        out.append("Different documents, so the bootstrap is unpaired.\n")
        out.append(md_table(d, {"p": "{:.3g}"}))
    pp = res / "eval" / "masker_paired.csv"
    if pp.exists():
        d = pd.read_csv(pp)
        d = d[d.metric.isin(["entity_f1", "char_leak_rate"]) & (d.model_a != d.model_b)]
        d["a − b [95% CI]"] = d.apply(
            lambda r: f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", axis=1
        )
        d = d[["model_a", "model_b", "set", "metric", "a − b [95% CI]", "p", "p_holm"]]
        out.append("### Paired comparisons between maskers\n")
        out.append(
            "Same documents; two-sided paired bootstrap; `p_holm` is Holm-adjusted within each metric.\n"
        )
        out.append(md_table(d, {"p": "{:.3g}", "p_holm": "{:.3g}"}))
    cp = res / "masker" / "summary_cost.csv"
    if cp.exists():
        c = pd.read_csv(cp)
        c["model"] = c.model.map(lambda m: NAMES.get(m, m))
        t = pd.DataFrame(
            {
                "model": c.model,
                "seeds": c.n_seeds,
                "train s": c.train_s_mean.map("{:.0f}".format),
                "trainable params": c.trainable_params_mean.map(lambda v: f"{v:,.0f}"),
                "latency p50 ms": c.latency_ms_p50_mean.map("{:.1f}".format),
                "latency p95 ms": c.latency_ms_p95_mean.map("{:.1f}".format),
                "docs/s": c.throughput_docs_per_s_mean.map("{:.1f}".format),
                "peak MB": c.peak_memory_mb_mean.map(lambda v: "–" if np.isnan(v) else f"{v:,.0f}"),
            }
        )
        out.append("### Cost\n")
        out.append(
            "Latency is one document at a time; throughput is batched over all evaluation sets.\n"
        )
        out.append(md_table(t))
    return "\n".join(out), bits


# ---------------------------------------------------------------------------------- pipeline
def pipeline_section(res: Path, fig_dir: Path) -> tuple[str, list[str]]:
    p = res / "eval" / "pipeline_ci.csv"
    if not p.exists():
        return "## Two-stage pipeline\n\n_Pipeline results not found in `results/pipeline/`._\n", []
    ci = pd.read_csv(p)
    summ = pd.read_csv(res / "pipeline" / "summary.csv")
    bits: list[str] = []
    out = ["## Two-stage pipeline (gate → masker)\n"]
    out.append(
        "Arms: `masker_doc` = masker on the whole document; `masker_sent` = masker on every "
        "sentence (isolates the effect of sentence splitting); `gate_*` = masker only on the "
        "sentences the gate flags (threshold chosen on validation data to keep ≥ the configured "
        "share of PII sentences); `gate_oracle` = only sentences that truly contain PII.\n"
    )
    m = summ.set_index(["set", "arm"])
    rows = []
    for _, r in ci.iterrows():
        s = m.loc[(r["set"], r.arm)]
        rows.append(
            {
                "set": r["set"],
                "arm": r.arm,
                "F1 [95% CI]": ci_str(r, "entity_f1"),
                "char leak [95% CI]": ci_str(r, "char_leak_rate", 4),
                "sentences skipped": "–"
                if pd.isna(s.skipped_fraction)
                else f"{s.skipped_fraction:.2f}",
                "gate recall": "–" if pd.isna(s.gate_recall) else f"{s.gate_recall:.3f}",
                "latency p50 ms": "–" if pd.isna(s.latency_ms_p50) else f"{s.latency_ms_p50:.1f}",
            }
        )
    out.append(md_table(pd.DataFrame(rows)))
    pp = res / "eval" / "pipeline_paired.csv"
    if pp.exists():
        d = pd.read_csv(pp)
        d = d[
            d.metric.isin(["char_leak_rate", "entity_f1"])
            & d.arm_b.isin(["masker_sent", "masker_doc"])
        ]
        d["a − b [95% CI]"] = d.apply(
            lambda r: f"{r['diff']:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]", axis=1
        )
        out.append("### Paired differences against the baselines\n")
        out.append(
            md_table(
                d[["arm_a", "arm_b", "set", "metric", "a − b [95% CI]", "p", "p_holm"]],
                {"p": "{:.3g}", "p_holm": "{:.3g}"},
            )
        )
    clean = summ[summ["set"] == "clean"].set_index("arm")
    if {"masker_doc", "masker_sent"} <= set(clean.index):
        bits.append(
            f"Pipeline (clean): masker on whole documents leaks {clean.loc['masker_doc'].char_leak_rate:.4f}; "
            f"on sentences {clean.loc['masker_sent'].char_leak_rate:.4f}."
        )
    for g in [a for a in clean.index if a.startswith("gate_") and a != "gate_oracle"]:
        bits.append(
            f"Pipeline (clean) {g}: skips {clean.loc[g].skipped_fraction:.2f} of sentences, leak "
            f"{clean.loc[g].char_leak_rate:.4f}, latency p50 {clean.loc[g].latency_ms_p50:.1f} ms "
            f"vs {clean.loc['masker_doc'].latency_ms_p50:.1f} ms masker-only."
        )
    d = clean.reset_index()
    d = d[d.latency_ms_p50.notna()]
    fig, ax = new_fig(8.6, 4.8)
    for _, r in d.iterrows():
        ax.scatter(
            r.latency_ms_p50,
            r.char_leak_rate * 100,
            s=70,
            color=COLORS["deberta"],
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
        )
        ax.annotate(
            r.arm,
            (r.latency_ms_p50, r.char_leak_rate * 100),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color=MUTED,
        )
    ax.set_xlabel("End-to-end latency p50, ms per document")
    ax.set_ylabel("Character leak rate (%)")
    ax.set_title("Pipeline arms on the clean set", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "pipeline_tradeoff.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    out.append("\n![pipeline trade-off](figures/pipeline_tradeoff.png)\n")
    return "\n".join(out), bits


LIMITS = """## Limitations and what is not claimed

- **Simulator only.** All quantum numbers come from classical simulation (`default.qubit`, plus a
  depolarizing-noise model). No real quantum hardware was used, so noise results are model
  predictions, not hardware measurements, and simulator latency says nothing about hardware latency.
- **Feature bottleneck.** The 8 gate features are a PCA projection of MiniLM embeddings; models that
  see only these features share that ceiling. Comparisons between them say little about the models
  on richer features.
- **Product-state kernel.** The angle-embedding fidelity kernel equals a closed-form product kernel
  (tested in `tests/test_gate.py`), so it is classically trivial by construction.
- **Dataset.** `ai4privacy/pii-masking-300k` is synthetic and templated, and some source spans are
  corrupted; documents with corrupted spans were excluded (counts in the data card). Results may not
  transfer to real-world text.
- **LLM masking.** LLMs return PII *text*; spans are recovered by exact string match, which masks
  every occurrence of a returned string and misses altered strings. Invalid JSON counts as no
  detections.
- **Context-only set.** The 50 synthetic examples use a label unseen in training, so strict F1 is 0
  by construction; the leak rate is the meaningful number. They were written by the project author
  and have not been independently reviewed.
- **Obfuscation sets** are rule-based transformations and cover only a few evasion styles.
- **Seeds and CIs.** Intervals capture document-sampling noise (and seed averaging where several seeds
  exist); they do not capture variation across datasets.
"""

REPRO = """## Reproduce

```
make setup
make PROFILE=full data stress gate masker eval pipeline report   # needs a CUDA GPU
make smoke                                                       # quick check of every stage
```

Run records (config, seed, git commit, timings, metrics) are in `results/`; this report is
generated from them by `scripts/make_report.py`.
"""


README_START = (
    "<!-- RESULTS:START (generated by scripts/make_report.py from results/; do not edit) -->"
)
README_END = "<!-- RESULTS:END -->"
GATE_ROWS = [
    ("logreg", "classical", "n2000", "Logistic regression"),
    ("mlp", "classical", "n2000", "MLP"),
    ("vqc", "exact", "n2000", "VQC (quantum, exact)"),
    ("vqc", "noisy_p0.01", "n2000", "VQC (noise p=0.01)"),
    ("qkernel_angle", "exact", "n500", "Quantum kernel (angle)"),
    ("qkernel_iqp", "exact", "n500", "Quantum kernel (IQP)"),
    ("minilm_finetuned", "raw_text", "full", "MiniLM fine-tuned (upper bound)"),
]


def summary_figure(ci: pd.DataFrame, cost: pd.DataFrame, path: Path) -> None:
    """Entity F1 on three sets (with CI) next to per-document latency, for the four maskers."""
    order = ["deberta", "qwen_lora", "smollm_lora", "qwen_zeroshot"]
    sets = ["clean", "obf_all", "rare"]
    fig, ax = plt.subplots(
        1, 2, figsize=(12, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.7, 1]}
    )
    for a in ax:
        style(a)
    w = 0.2
    for i, m in enumerate(order):
        v = [ci[(ci.model == m) & (ci["set"] == s)].iloc[0] for s in sets]
        f = [x.entity_f1 for x in v]
        err = [[x.entity_f1 - x.entity_f1_lo for x in v], [x.entity_f1_hi - x.entity_f1 for x in v]]
        xs = np.arange(3) + (i - 1.5) * w
        ax[0].bar(xs, f, w * 0.88, color=COLORS[m], label=NAMES[m], zorder=3)
        ax[0].errorbar(xs, f, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=4)
        for x, y in zip(xs, f, strict=True):
            ax[0].text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=7.5, color=MUTED)
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels(["clean", "obfuscated (all)", "rare types"])
    ax[0].set_ylabel("Entity F1 (strict)")
    ax[0].set_ylim(0, 1.12)
    ax[0].set_title("Entity F1 by evaluation set (95% CI)", loc="left", fontsize=11)
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    lat = [float(cost.set_index("model").loc[m, "latency_ms_p50_mean"]) for m in order]
    ax[1].bar(range(4), lat, 0.6, color=[COLORS[m] for m in order], zorder=3)
    for x, y in enumerate(lat):
        ax[1].text(
            x,
            y * 1.15,
            f"{y:.0f} ms" if y >= 10 else f"{y:.1f} ms",
            ha="center",
            fontsize=8.5,
            color=MUTED,
        )
    ax[1].set_yscale("log")
    ax[1].set_ylim(1, 2500)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels([NAMES[m] for m in order], rotation=18, ha="right", fontsize=8.5)
    ax[1].set_ylabel("Latency p50, ms per document (log)")
    ax[1].set_title("Latency", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def readme_block(res: Path, bits: list[str]) -> str:
    """Compact 'results at a glance' section built only from files in ``results/``."""
    out = [README_START, "### Key findings", ""]
    out.append(
        "Generated from `results/` by `make report` (full profile). See "
        "[`reports/report.md`](reports/report.md) for confidence intervals, paired tests, "
        "per-set tables and limitations.\n"
    )
    out += [f"- {b}" for b in bits] + [""]
    g = pd.read_csv(res / "gate" / "summary.csv")
    rows = []
    for m, st, sz, label in GATE_ROWS:
        r = g[(g.model == m) & (g.setting == st) & (g["size"] == sz)]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append(
            {
                "model": label,
                "accuracy": f"{r.accuracy_mean:.3f}",
                "F1": f"{r.f1_mean:.3f}",
                "ROC-AUC": f"{r.roc_auc_mean:.3f}",
                "latency ms/sentence": f"{r.latency_ms_single_mean:.3g}",
                "seeds": int(r.n_seeds),
            }
        )
    out += [
        "### Stage 1: gate (simulated quantum vs classical, 8 shared features)",
        "",
        md_table(pd.DataFrame(rows)),
    ]
    ci = pd.read_csv(res / "eval" / "masker_ci.csv")
    cost = pd.read_csv(res / "masker" / "summary_cost.csv").set_index("model")
    rows = []
    for m in ["deberta", "qwen_lora", "smollm_lora", "qwen_zeroshot"]:
        c = {
            s: ci[(ci.model == m) & (ci["set"] == s)].iloc[0] for s in ("clean", "obf_all", "rare")
        }
        rows.append(
            {
                "masker": NAMES[m],
                "seeds": int(c["clean"].n_seeds),
                "F1 clean": f"{c['clean'].entity_f1:.3f}",
                "F1 obfuscated": f"{c['obf_all'].entity_f1:.3f}",
                "F1 rare": f"{c['rare'].entity_f1:.3f}",
                "leak % clean": f"{100 * c['clean'].char_leak_rate:.1f}",
                "leak % obfuscated": f"{100 * c['obf_all'].char_leak_rate:.1f}",
                "latency p50 ms": f"{cost.loc[m, 'latency_ms_p50_mean']:.1f}",
            }
        )
    out += [
        "### Stage 2: maskers (character leak = share of PII characters left unmasked)",
        "",
        md_table(pd.DataFrame(rows)),
        "![F1 and latency](reports/figures/summary_f1_latency.png)",
        "",
    ]
    pl = res / "pipeline" / "summary.csv"
    if pl.exists():
        ps = pd.read_csv(pl)
        ps = ps[ps["set"] == "clean"].set_index("arm")
        rows = [
            {
                "arm": a,
                "F1": f"{r.entity_f1:.3f}",
                "leak %": f"{100 * r.char_leak_rate:.2f}",
                "sentences skipped": "–"
                if pd.isna(r.skipped_fraction)
                else f"{r.skipped_fraction:.2f}",
                "latency p50 ms": "–" if pd.isna(r.latency_ms_p50) else f"{r.latency_ms_p50:.1f}",
            }
            for a, r in ps.iterrows()
        ]
        out += ["### Two-stage pipeline (clean documents)", "", md_table(pd.DataFrame(rows))]
    out.append(README_END)
    return "\n".join(out) + "\n"


def update_readme(block: str) -> None:
    """Replace the marked results block in README.md (insert it before 'Quick start' if absent)."""
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text()
    if README_START in text and README_END in text:
        a, b = text.index(README_START), text.index(README_END) + len(README_END)
        text = text[:a] + block.rstrip("\n") + text[b:]
    else:
        text = text.replace("## Quick start", block + "\n## Quick start", 1)
    readme.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = Path(cfg["paths"]["results_dir"])
    rep = ensure_dir(Path(cfg["paths"]["reports_dir"]))
    fig_dir = ensure_dir(rep / "figures")
    gate, gbits = gate_section(res, fig_dir)
    extra, xbits = extras_section(res, fig_dir)
    mask, mbits = masker_section(res, fig_dir)
    pipe, pbits = pipeline_section(res, fig_dir)
    summary = "\n".join(f"- {b}" for b in [*gbits, *xbits, *mbits, *pbits]) or "_No results found._"
    text = (
        "# Clean F1 hides leaks: stress-testing small-language-model PII maskers, "
        "with a quantum-vs-classical benchmark for the PII gate\n\n"
        f"_Generated from `{res}` by `scripts/make_report.py`; every number below is read from "
        "a results file._\n\n## Key results\n\n"
        + summary
        + "\n\n"
        + gate
        + "\n"
        + extra
        + "\n"
        + mask
        + "\n"
        + pipe
        + "\n"
        + LIMITS
        + "\n"
        + REPRO
    )
    (rep / "report.md").write_text(text)
    if (res / "eval" / "masker_ci.csv").exists():
        ci = pd.read_csv(res / "eval" / "masker_ci.csv")
        cost = pd.read_csv(res / "masker" / "summary_cost.csv")
        summary_figure(ci, cost, fig_dir / "summary_f1_latency.png")
        if cfg.get("profile") == "full":
            update_readme(readme_block(res, [*gbits, *xbits, *mbits, *pbits]))
    log.info("wrote %s (%d chars) and figures in %s", rep / "report.md", len(text), fig_dir)


if __name__ == "__main__":
    main()
