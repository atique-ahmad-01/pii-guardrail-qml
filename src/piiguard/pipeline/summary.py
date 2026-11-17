"""Collect pipeline run records into one table (only values found in the records)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

COLS = [
    "entity_f1",
    "entity_precision",
    "entity_recall",
    "char_leak_rate",
    "entity_leak_rate",
    "gate_recall",
    "gate_precision",
    "skipped_fraction",
    "docs_skipped_fraction",
    "n_docs",
]


def summarize(pipeline_dir: str | Path) -> pd.DataFrame:
    """One row per (set, arm); latency columns are joined on arm."""
    ev, lat = [], {}
    for p in sorted(Path(pipeline_dir).glob("pipeline__*.json")):
        m = json.loads(p.read_text())["metrics"]
        if m.get("kind") == "pipeline_eval":
            ev.append({"set": m["set"], "arm": m["arm"], **{k: m.get(k) for k in COLS}})
        elif m.get("kind") == "pipeline_latency":
            lat[m["arm"]] = {k: m[k] for k in ("latency_ms_p50", "latency_ms_p95")}
    df = pd.DataFrame(ev)
    if df.empty:
        return df
    for k in ("latency_ms_p50", "latency_ms_p95"):
        df[k] = df["arm"].map(lambda a, k=k: lat.get(a, {}).get(k))
    return df.sort_values(["set", "arm"]).reset_index(drop=True)
