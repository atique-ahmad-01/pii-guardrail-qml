"""Aggregate per-run gate records in ``results/gate`` into mean +/- std summary tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

METRICS = [
    "roc_auc",
    "pr_auc",
    "recall_at_95p",
    "f1",
    "train_s",
    "latency_ms_single",
    "params",
    "n_eval",
    "auc_exact_same_subset",
]


def load_records(gate_dir: str | Path) -> list[dict]:
    """All JSON run records in ``gate_dir``."""
    return [json.loads(p.read_text()) for p in sorted(Path(gate_dir).glob("*.json"))]


def summarize(gate_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(performance table, barren-plateau table)`` built only from the records on disk."""
    rows, barren = [], []
    for rec in load_records(gate_dir):
        m = rec["metrics"]
        if m.get("kind") == "barren":
            barren.append(
                {
                    **{k: m[k] for k in ("n_qubits", "n_layers", "n_samples")},
                    "var_first_param": m["var_first_param"],
                    "var_mean_all_params": m["var_mean_all_params"],
                }
            )
        elif m.get("kind") == "gate_model":
            rows.append(
                {
                    "model": m["model"],
                    "setting": m["setting"],
                    "size": m["size"],
                    "seed": rec["seed"],
                    **{k: m.get(k) for k in METRICS if k != "auc_exact_same_subset"},
                    # noisy rows are scored on a smaller subset: compare with this paired value
                    "auc_exact_same_subset": (m.get("exact_same_subset") or {}).get("roc_auc"),
                }
            )
    perf = pd.DataFrame(rows)
    if not perf.empty:
        g = perf.groupby(["model", "setting", "size"])
        agg = g[METRICS].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg["n_seeds"] = g.size()
        perf = agg.reset_index()
    return perf, pd.DataFrame(barren)
