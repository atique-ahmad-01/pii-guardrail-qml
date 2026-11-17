"""Pure helpers for the gate -> masker pipeline (no models, easy to test)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

Span = dict[str, Any]


def shift_spans(spans: Sequence[Span], offset: int) -> list[Span]:
    """Move sentence-relative spans to document offsets."""
    return [{**s, "start": s["start"] + offset, "end": s["end"] + offset} for s in spans]


def gated_spans(
    sent_offsets: Sequence[tuple[int, int]],
    sent_spans: Sequence[Sequence[Span]],
    flags: Sequence[bool],
) -> list[Span]:
    """Document-level spans from the sentences the gate let through.

    ``sent_spans[i]`` are sentence-relative spans predicted by the masker for sentence ``i``.
    Sentences with ``flags[i]`` False were never shown to the masker, so they contribute nothing.
    """
    out: list[Span] = []
    for (a, _), sp, f in zip(sent_offsets, sent_spans, flags, strict=True):
        if f:
            out += shift_spans(sp, a)
    return out


def pick_threshold(y_val: np.ndarray, p_val: np.ndarray, target_recall: float = 0.99) -> float:
    """Highest threshold that still flags at least ``target_recall`` of validation positives.

    A missed sentence is a leak, so the gate is run conservatively. Returns 0.5 when the
    validation set has no positives.
    """
    pos = np.sort(np.asarray(p_val, dtype=float)[np.asarray(y_val) == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1.0 - target_recall) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def gate_sentence_stats(y: np.ndarray, flags: np.ndarray) -> dict[str, float]:
    """Sentence-level recall/precision of the flags and the share of sentences skipped."""
    y, flags = np.asarray(y).astype(bool), np.asarray(flags).astype(bool)
    tp = int((y & flags).sum())
    return {
        "gate_recall": float(tp / y.sum()) if y.sum() else float("nan"),
        "gate_precision": float(tp / flags.sum()) if flags.sum() else float("nan"),
        "skipped_fraction": float(1.0 - flags.mean()) if len(flags) else float("nan"),
        "n_sentences": int(len(y)),
    }
