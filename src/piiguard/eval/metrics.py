"""Masker metrics, computed from per-document counts so they can be bootstrapped later.

Definitions (all spans are character offsets):

* **Entity F1**: strict. A predicted span is correct only if ``(start, end, label)`` equals a gold
  span.
* **Character leak rate** (headline privacy metric): share of gold PII characters not covered by
  *any* predicted mask. Whitespace inside a gold span is not counted. Type is ignored: a mask of the
  wrong type still hides the characters.
* **Entity leak rate**: share of gold entities with at least one uncovered character.
* **Type-agnostic recall**: share of gold entities whose exact ``(start, end)`` was predicted,
  whatever the label.
* **Recall per type**: strict (exact span and label) and coverage-based (entity fully covered).

``eval_labels`` restricts the *gold* spans scored (rare-type and context-only sets). Strict
precision also ignores predicted spans of other labels; leaks always look at all predicted masks.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
import pandas as pd

Span = dict[str, Any]
BASE_COLS = [
    "tp",
    "fp",
    "fn",
    "agn_tp",
    "gold_ents",
    "leaked_ents",
    "gold_chars",
    "leaked_chars",
]


def doc_counts(
    text: str,
    gold: Sequence[Span],
    pred: Sequence[Span],
    eval_labels: Collection[str] | None = None,
) -> dict[str, int]:
    """Counts for one document: ``BASE_COLS`` plus ``n__L``, ``tp__L``, ``cov__L`` per label."""
    keep = set(eval_labels) if eval_labels is not None else None
    g = [s for s in gold if keep is None or s["label"] in keep]
    p = [s for s in pred if keep is None or s["label"] in keep]
    gset = {(s["start"], s["end"], s["label"]) for s in g}
    pset = {(s["start"], s["end"], s["label"]) for s in p}
    tp = len(gset & pset)
    agn = len({(a, b) for a, b, _ in gset} & {(a, b) for a, b, _ in pset})
    covered = np.zeros(len(text), dtype=bool)
    for s in pred:  # leaks consider every predicted mask
        covered[s["start"] : s["end"]] = True
    out = {
        "tp": tp,
        "fp": len(pset) - tp,
        "fn": len(gset) - tp,
        "agn_tp": agn,
        "gold_ents": len(g),
        "leaked_ents": 0,
        "gold_chars": 0,
        "leaked_chars": 0,
    }
    for s in g:
        idx = [i for i in range(s["start"], s["end"]) if not text[i].isspace()]
        miss = sum(1 for i in idx if not covered[i])
        out["gold_chars"] += len(idx)
        out["leaked_chars"] += miss
        out["leaked_ents"] += int(miss > 0)
        lab = s["label"]
        out[f"n__{lab}"] = out.get(f"n__{lab}", 0) + 1
        out[f"tp__{lab}"] = out.get(f"tp__{lab}", 0) + int((s["start"], s["end"], lab) in pset)
        out[f"cov__{lab}"] = out.get(f"cov__{lab}", 0) + int(miss == 0)
    return out


def counts_frame(
    texts: Sequence[str],
    golds: Sequence[Sequence[Span]],
    preds: Sequence[Sequence[Span]],
    eval_labels: Sequence[Collection[str] | None] | None = None,
) -> pd.DataFrame:
    """One row of counts per document (columns missing for a label are filled with 0)."""
    eval_labels = eval_labels or [None] * len(texts)
    rows = [
        doc_counts(t, g, p, el)
        for t, g, p, el in zip(texts, golds, preds, eval_labels, strict=True)
    ]
    df = pd.DataFrame(rows).fillna(0).astype(int)
    for c in BASE_COLS:
        if c not in df:
            df[c] = 0
    return df


def _div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def summarize_frame(df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate a counts frame (or a bootstrap resample of it) into the headline metrics."""
    s = df.sum()
    tp, fp, fn = s["tp"], s["fp"], s["fn"]
    prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = _div(2 * prec * rec, prec + rec) if tp else (0.0 if (fp or fn) else float("nan"))
    labels = sorted(c[3:] for c in df.columns if c.startswith("n__"))
    return {
        "entity_precision": prec,
        "entity_recall": rec,
        "entity_f1": f1,
        "char_leak_rate": _div(s["leaked_chars"], s["gold_chars"]),
        "entity_leak_rate": _div(s["leaked_ents"], s["gold_ents"]),
        "type_agnostic_recall": _div(s["agn_tp"], s["gold_ents"]),
        "n_docs": int(len(df)),
        "n_gold_entities": int(s["gold_ents"]),
        "recall_strict_by_type": {lab: _div(s[f"tp__{lab}"], s[f"n__{lab}"]) for lab in labels},
        "coverage_recall_by_type": {lab: _div(s[f"cov__{lab}"], s[f"n__{lab}"]) for lab in labels},
        "n_by_type": {lab: int(s[f"n__{lab}"]) for lab in labels},
    }
