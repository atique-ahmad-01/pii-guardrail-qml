"""Phase 6: bootstrap confidence intervals and significance tests from saved per-document counts.

Reads ``<results_dir>/masker/counts`` and ``<results_dir>/pipeline/counts``; writes CSVs to
``<results_dir>/eval/``. Nothing is recomputed from models, so this is cheap and reproducible.
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Any

import pandas as pd

from piiguard.eval.bootstrap import METRICS, bootstrap_ci, compare, holm
from piiguard.utils import ensure_dir, get_logger, load_config

log = get_logger("run_eval")
MASKER_PAIRS = [
    ("deberta", "qwen_lora"),
    ("deberta", "smollm_lora"),
    ("qwen_lora", "smollm_lora"),
    ("qwen_lora", "qwen_zeroshot"),
]
SEED_RE = re.compile(r"^(?P<model>.+?)__(?P<set>.+?)__seed(?P<seed>\d+)\.parquet$")


def load_masker_counts(d: Path) -> dict[tuple[str, str], list[pd.DataFrame]]:
    """``{(model, set): [counts per seed]}``, documents aligned by id."""
    out: dict[tuple[str, str], list[tuple[int, pd.DataFrame]]] = {}
    for p in sorted((d / "counts").glob("*.parquet")):
        m = SEED_RE.match(p.name)
        if m:
            out.setdefault((m["model"], m["set"]), []).append((int(m["seed"]), pd.read_parquet(p)))
    return {k: [f for _, f in sorted(v, key=lambda t: t[0])] for k, v in out.items()}


def load_pipeline_counts(d: Path) -> dict[tuple[str, str], list[pd.DataFrame]]:
    """``{(arm, set): [counts]}`` (the pipeline runs one seed)."""
    out = {}
    for p in sorted((d / "counts").glob("*.parquet")):
        arm, set_ = p.stem.split("__", 1)
        out[(arm, set_)] = [pd.read_parquet(p)]
    return out


def ci_table(
    data: dict[tuple[str, str], list[pd.DataFrame]], ev: dict[str, Any], key: str
) -> pd.DataFrame:
    """One row per (system, set) with point estimate and CI for every metric."""
    rows = []
    for (system, set_), frames in data.items():
        r = bootstrap_ci(frames, ev["bootstrap_resamples"], ev["ci"], seed=0)
        row = {key: system, "set": set_, "n_seeds": len(frames), "n_docs": len(frames[0])}
        for m in METRICS:
            row.update({f"{m}": r[m]["point"], f"{m}_lo": r[m]["lo"], f"{m}_hi": r[m]["hi"]})
        rows.append(row)
    return pd.DataFrame(rows).sort_values([key, "set"]).reset_index(drop=True)


def paired_table(
    data: dict[tuple[str, str], list[pd.DataFrame]],
    pairs: list[tuple[str, str]],
    ev: dict[str, Any],
    key: str,
) -> pd.DataFrame:
    """Paired bootstrap differences ``a - b`` per set; Holm-adjusted p within each metric."""
    rows = []
    for a, b in pairs:
        for set_ in sorted({s for (_, s) in data}):
            if (a, set_) in data and (b, set_) in data:
                fa, fb = data[(a, set_)], data[(b, set_)]
                n = min(len(fa), len(fb))  # compare seed-by-seed pools of equal size
                r = compare(fa[:n], fb[:n], True, ev["bootstrap_resamples"], ev["ci"], seed=0)
                for m in METRICS:
                    rows.append({f"{key}_a": a, f"{key}_b": b, "set": set_, "metric": m, **r[m]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_holm"] = float("nan")
    for idx in df.groupby("metric").groups.values():
        df.loc[idx, "p_holm"] = holm(df.loc[idx, "p"].tolist())
    return df


def delta_vs_clean(
    data: dict[tuple[str, str], list[pd.DataFrame]], ev: dict[str, Any]
) -> pd.DataFrame:
    """Each stress set minus the clean set per model (different documents, so unpaired)."""
    rows = []
    for (model, set_), frames in data.items():
        if set_ == "clean" or (model, "clean") not in data:
            continue
        r = compare(frames, data[(model, "clean")], False, ev["bootstrap_resamples"], ev["ci"], 0)
        for m in METRICS:
            rows.append({"model": model, "set": set_, "metric": m, **r[m]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ev = cfg["eval"]
    res = Path(cfg["paths"]["results_dir"])
    out = ensure_dir(res / "eval")
    mk = load_masker_counts(res / "masker")
    if mk:
        ci_table(mk, ev, "model").to_csv(out / "masker_ci.csv", index=False)
        paired_table(mk, MASKER_PAIRS, ev, "model").to_csv(out / "masker_paired.csv", index=False)
        delta_vs_clean(mk, ev).to_csv(out / "masker_delta_vs_clean.csv", index=False)
        log.info("masker: %d (model, set) cells", len(mk))
    pl = load_pipeline_counts(res / "pipeline")
    if pl:
        ci_table(pl, ev, "arm").to_csv(out / "pipeline_ci.csv", index=False)
        arms = sorted({a for a, _ in pl})
        pairs = [
            (a, base)
            for base in ("masker_doc", "masker_sent")
            for a in arms
            if a.startswith("gate_") and a != base
        ]
        pairs += list(
            itertools.combinations(
                [a for a in arms if a.startswith("gate_") and a != "gate_oracle"], 2
            )
        )
        paired_table(pl, pairs, ev, "arm").to_csv(out / "pipeline_paired.csv", index=False)
        log.info("pipeline: %d (arm, set) cells", len(pl))


if __name__ == "__main__":
    main()
