"""Supervised fine-tuning pairs for the LLM maskers: instruction + text -> JSON list of PII."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

INSTRUCTION = (
    "Extract all personally identifiable information (PII) from the text below. "
    'Return a JSON list of objects with the keys "type" and "text", in order of appearance, '
    'where "text" is the exact substring from the input. Return [] if there is no PII.'
)


def build_prompt(text: str) -> str:
    """Prompt = instruction + the document text."""
    return f"{INSTRUCTION}\n\nText:\n{text}\n\nJSON:"


def build_target(text: str, spans: Sequence[dict[str, Any]]) -> str:
    """JSON list ``[{"type": LABEL, "text": exact substring}]`` in order of appearance."""
    items = [
        {"type": sp["label"], "text": text[sp["start"] : sp["end"]]}
        for sp in sorted(spans, key=lambda sp: (sp["start"], sp["end"]))
    ]
    return json.dumps(items, ensure_ascii=False)
