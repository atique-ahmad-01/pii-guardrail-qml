"""Load evaluation sets (clean test, cross-lingual, stress) with prompts and eval-label filters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from piiguard.data.load import frame_to_docs

SET_FILES = {
    "clean": "test.parquet",
    "cross_de": "cross_de.parquet",
    "cross_fr": "cross_fr.parquet",
    "cross_it": "cross_it.parquet",
}


def load_sets(
    data_dir: str | Path, names: list[str], cap: int | None = None
) -> dict[str, dict[str, Any]]:
    """``{name: {"docs", "prompts", "eval_labels"}}``; ``cap`` keeps the first N documents."""
    d = Path(data_dir)
    sets = {}
    for name in names:
        df = pd.read_parquet(d / SET_FILES.get(name, f"stress/{name}.parquet"))
        if cap:
            df = df.head(cap)
        el = (
            [json.loads(x) if x else None for x in df["eval_labels"]]
            if "eval_labels" in df
            else [None] * len(df)
        )
        sets[name] = {
            "docs": frame_to_docs(df),
            "prompts": df["prompt"].tolist(),
            "eval_labels": el,
        }
    return sets
