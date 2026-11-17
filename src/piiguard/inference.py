"""Use the trained guardrail on new text (real-time or batch).

    from piiguard.inference import PiiGuardrail
    guard = PiiGuardrail.from_dir("checkpoints/deberta_seed0/best", "checkpoints/gates")
    print(guard.mask("Call me on 555-0142").masked)

Modes (see the README for what was measured for each):
  document   the masker reads the whole text (best measured privacy; no gate needed)
  doc_gate   the gate may skip the masker, but only when NO sentence is flagged
  sentence   only sentences the gate flags are shown to the masker (leaks more, measured)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from piiguard.data.sentences import split_sentences
from piiguard.data.sft import build_prompt
from piiguard.masker.spans import mask_text
from piiguard.pipeline.assemble import gated_spans
from piiguard.utils import get_logger

log = get_logger("inference")
MODES = ("document", "doc_gate", "sentence")


@dataclass
class GuardResult:
    """Masked text, the spans that were hidden, and what the gate did."""

    masked: str
    spans: list[dict[str, Any]]
    mode: str
    masker_ran: bool
    sentences: list[tuple[str, float, bool]] = field(default_factory=list)  # (text, score, flagged)


class _GateView:
    """One gate out of a ``Gates`` object, with its threshold."""

    def __init__(self, gates: Any, name: str) -> None:
        if name not in gates.models:
            raise ValueError(f"gate {name!r} not loaded; available: {sorted(gates.models)}")
        self.gates, self.name, self.threshold = gates, name, gates.thresholds[name]

    def flags(self, sents: list[str]) -> tuple[np.ndarray, np.ndarray]:
        scores = self.gates.score(self.name, sents)
        return scores, scores >= self.threshold


class ChunkedEncoderMasker:
    """DeBERTa masker that splits texts longer than the model limit into sentence groups.

    Without this, text past the model's token limit would be silently ignored, which is a leak.
    """

    def __init__(self, masker: Any, max_tokens: int = 512) -> None:
        self.m, self.budget = masker, max_tokens - 8

    def _n_tokens(self, text: str) -> int:
        return len(self.m.tok(text, add_special_tokens=False)["input_ids"])

    def _chunks(self, text: str) -> list[tuple[int, int]]:
        spans = split_sentences(text) or [(0, len(text))]
        out, start, end = [], spans[0][0], spans[0][1]
        for a, b in spans[1:]:
            if self._n_tokens(text[start:b]) <= self.budget:
                end = b
            else:
                out.append((start, end))
                start, end = a, b
        out.append((start, end))
        return out

    def predict(self, texts: list[str]) -> list[list[dict[str, Any]]]:
        result = []
        for text in texts:
            if self._n_tokens(text) <= self.budget:
                result.append(self.m.predict([text])[0])
                continue
            log.info("long text (%d chars) processed in chunks", len(text))
            spans: list[dict[str, Any]] = []
            for a, b in self._chunks(text):
                for s in self.m.predict([text[a:b]])[0]:
                    spans.append({**s, "start": s["start"] + a, "end": s["end"] + a})
            result.append(spans)
        return result


class LLMAdapterMasker:
    """LoRA language-model masker with the same ``predict(texts)`` interface."""

    def __init__(self, base: str, adapter: Path, max_new_tokens: int = 512) -> None:
        from piiguard.masker.llm_infer import LLMMasker

        self.m = LLMMasker(base, adapter, max_new_tokens, batch_size=8)

    def predict(self, texts: list[str]) -> list[list[dict[str, Any]]]:
        return self.m.predict([build_prompt(t) for t in texts], texts)[0]


class PiiGuardrail:
    """The gate (optional) plus a masker behind one ``mask`` call."""

    def __init__(self, masker: Any, gates: Any | None = None, gate: str | None = None) -> None:
        self.masker = masker
        self.gate = _GateView(gates, gate) if gates is not None and gate else None

    @classmethod
    def from_dir(
        cls,
        masker_path: str | Path,
        gates_dir: str | Path | None = None,
        gate: str = "logreg",
        max_length: int = 512,
    ) -> PiiGuardrail:
        """Load a masker (DeBERTa folder or LoRA adapter folder) and, optionally, saved gates."""
        mp = Path(masker_path)
        if (mp / "adapter_config.json").exists():
            base = json.loads((mp / "adapter_config.json").read_text())["base_model_name_or_path"]
            masker: Any = LLMAdapterMasker(base, mp)
        else:
            from piiguard.masker.token_clf import DebertaMasker

            masker = ChunkedEncoderMasker(DebertaMasker(mp, max_length), max_length)
        gates = None
        if gates_dir is not None:
            from piiguard.pipeline.gates import Gates

            gates = Gates.load(gates_dir, only=[gate])
        return cls(masker, gates, gate if gates is not None else None)

    def mask(self, text: str, mode: str = "document") -> GuardResult:
        """Hide the PII in ``text``. See the module docstring for the modes."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not text.strip():
            return GuardResult(text, [], mode, False)
        if mode == "document":
            spans = self.masker.predict([text])[0]
            return GuardResult(mask_text(text, spans), spans, mode, True)
        if self.gate is None:
            raise ValueError(f"mode {mode!r} needs a gate; load one with gates_dir=...")
        offs = split_sentences(text)
        sents = [text[a:b] for a, b in offs]
        scores, flags = self.gate.flags(sents)
        info = [(s, float(p), bool(f)) for s, p, f in zip(sents, scores, flags, strict=True)]
        if mode == "doc_gate":
            if not flags.any():
                return GuardResult(text, [], mode, False, info)
            spans = self.masker.predict([text])[0]
            return GuardResult(mask_text(text, spans), spans, mode, True, info)
        passed = [s for s, f in zip(sents, flags, strict=True) if f]
        it = iter(self.masker.predict(passed) if passed else [])
        per = [next(it) if f else [] for f in flags]
        spans = gated_spans(offs, per, [bool(f) for f in flags])
        return GuardResult(mask_text(text, spans), spans, mode, bool(passed), info)

    def mask_many(self, texts: list[str], mode: str = "document") -> list[GuardResult]:
        """``mask`` for several texts."""
        return [self.mask(t, mode) for t in texts]
