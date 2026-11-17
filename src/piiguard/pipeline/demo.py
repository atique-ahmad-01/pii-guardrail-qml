"""Core of the demo: mask a text with the gate -> masker pipeline (no UI dependencies)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from piiguard.data.sentences import split_sentences
from piiguard.masker.spans import mask_text
from piiguard.pipeline.assemble import gated_spans


@dataclass
class DemoResult:
    """Masked text plus what the gate decided for each sentence."""

    masked: str
    spans: list[dict[str, Any]]
    sentences: list[tuple[str, float, bool]]  # (sentence, gate score, passed to masker)


def run_demo(text: str, gate: Any, masker: Any, use_gate: bool = True) -> DemoResult:
    """Mask ``text``. ``gate`` has ``score(name, texts)`` and ``thresholds``; ``masker.predict``.

    With ``use_gate`` False every sentence goes to the masker (the masker-only baseline).
    The gate used is the first one in ``gate.thresholds``.
    """
    offs = split_sentences(text)
    sents = [text[a:b] for a, b in offs]
    if not sents:
        return DemoResult(text, [], [])
    name = next(iter(gate.thresholds))
    scores = gate.score(name, sents) if use_gate else np.ones(len(sents))
    flags = scores >= gate.thresholds[name] if use_gate else np.ones(len(sents), bool)
    sent_spans = masker.predict([s for s, f in zip(sents, flags, strict=True) if f])
    it = iter(sent_spans)
    per_sent = [next(it) if f else [] for f in flags]
    spans = gated_spans(offs, per_sent, [bool(f) for f in flags])
    return DemoResult(
        mask_text(text, spans),
        spans,
        [(s, float(p), bool(f)) for s, p, f in zip(sents, scores, flags, strict=True)],
    )
