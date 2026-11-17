"""Aggregate masker run records into mean +/- std tables (built only from files on disk)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EVAL_COLS = [
    "entity_f1",
    "char_leak_rate",
    "entity_leak_rate",
    "type_agnostic_recall",
    "valid_json_rate",
]
COST_COLS = [
    "train_s",
    "trainable_params",
    "latency_ms_p50",
    "latency_ms_p95",
    "throughput_docs_per_s",
    "peak_memory_mb",
]


def summarize(masker_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(per model x set metrics, per model cost)`` aggregated over seeds."""
    ev, cost = [], []
    for p in sorted(Path(masker_dir).glob("*.json")):
        rec = json.loads(p.read_text())
        m = rec["metrics"]
        if m.get("kind") == "masker_eval":
            ev.append(
                {
                    "model": m["model"],
                    "set": m["set"],
                    "seed": rec["seed"],
                    **{k: m.get(k) for k in EVAL_COLS},
                    "n_docs": m["n_docs"],
                }
            )
        elif m.get("kind") == "masker_cost":
            cost.append(
                {"model": m["model"], "seed": rec["seed"], **{k: m.get(k) for k in COST_COLS}}
            )

    def agg(rows: list[dict], keys: list[str], cols: list[str]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        g = df.groupby(keys)
        out = g[cols].agg(["mean", "std"])
        out.columns = [f"{a}_{b}" for a, b in out.columns]
        out["n_seeds"] = g.size()
        return out.reset_index()

    return agg(ev, ["model", "set"], EVAL_COLS + ["n_docs"]), agg(cost, ["model"], COST_COLS)
