"""Sentence splitting and sentence-level gate labels (PII present: yes/no)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

_LINE = re.compile(r"[^\n]+")
_SENT = re.compile(r".+?(?:[.!?](?=\s|$)|$)")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character offsets of sentences, trimmed of whitespace.

    A newline always ends a sentence; within a line, ``. ! ?`` followed by whitespace does too.
    This is a deliberately simple rule-based splitter that keeps offsets exact.
    """
    out: list[tuple[int, int]] = []
    for line in _LINE.finditer(text):
        for m in _SENT.finditer(line.group()):
            seg = m.group()
            lead = len(seg) - len(seg.lstrip())
            body = seg.strip()
            if body:
                s = line.start() + m.start() + lead
                out.append((s, s + len(body)))
    return out


def label_sentences(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Sentences of one document; ``label`` is 1 if any PII span overlaps the sentence."""
    rows = []
    for k, (a, b) in enumerate(split_sentences(doc["text"])):
        hit = [sp for sp in doc["spans"] if sp["start"] < b and sp["end"] > a]
        rows.append(
            {
                "doc_id": doc["id"],
                "sent_idx": k,
                "start": a,
                "end": b,
                "text": doc["text"][a:b],
                "label": int(bool(hit)),
                "n_spans": len(hit),
            }
        )
    return rows


def sentence_table(docs: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """All labelled sentences of ``docs`` as a DataFrame."""
    rows = [r for d in docs for r in label_sentences(d)]
    cols = ["doc_id", "sent_idx", "start", "end", "text", "label", "n_spans"]
    return pd.DataFrame(rows, columns=cols)


def balance(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Subsample the larger class so both classes have equal size (seeded, order preserved)."""
    pos, neg = df[df.label == 1], df[df.label == 0]
    n = min(len(pos), len(neg))
    rng = np.random.default_rng(seed)
    keep = np.concatenate(
        [
            rng.choice(pos.index.to_numpy(), n, replace=False),
            rng.choice(neg.index.to_numpy(), n, replace=False),
        ]
    )
    return df.loc[np.sort(keep)].reset_index(drop=True)
