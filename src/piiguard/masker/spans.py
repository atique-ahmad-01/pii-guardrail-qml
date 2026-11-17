"""Parse LLM output into PII spans and apply masks.

Known limitation (also stated in the report): the LLM returns only the PII *text*, so spans are
recovered by exact string match. Every occurrence of a returned string is masked, which can
over-mask a string that is PII in one place and not in another, and a hallucinated or slightly
altered string matches nothing, which becomes a leak. Invalid JSON counts as a failure: no spans.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

Span = dict[str, Any]
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_llm_output(raw: str) -> tuple[list[dict[str, str]], bool]:
    """Return ``(items, valid)`` where items are ``{"type", "text"}`` dicts.

    Tolerates a markdown code fence and text around the list (everything from the first ``[`` to
    the last ``]`` is parsed). The output is *valid* only if that slice is a JSON list whose every
    element is an object with string ``type`` and ``text``. Anything else returns ``([], False)``.
    """
    s = raw.strip()
    m = _FENCE.search(s)
    if m:
        s = m.group(1).strip()
    a, b = s.find("["), s.rfind("]")
    if a == -1 or b <= a:
        return [], False
    try:
        data = json.loads(s[a : b + 1])
    except json.JSONDecodeError:
        return [], False
    if not isinstance(data, list):
        return [], False
    items = []
    for it in data:
        if not (
            isinstance(it, dict)
            and isinstance(it.get("text"), str)
            and isinstance(it.get("type"), str)
        ):
            return [], False
        if it["text"]:
            items.append({"type": it["type"], "text": it["text"]})
    return items, True


def map_items_to_spans(text: str, items: Sequence[dict[str, str]]) -> list[Span]:
    """Exact-match every occurrence of each item text; keep disjoint spans (earlier, longer)."""
    cand: list[Span] = []
    for it in items:
        v, i = it["text"], text.find(it["text"])
        while i != -1:
            cand.append({"start": i, "end": i + len(v), "label": it["type"], "value": v})
            i = text.find(v, i + 1)
    cand.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
    out: list[Span] = []
    for s in cand:
        if out and s["start"] < out[-1]["end"]:
            continue
        out.append(s)
    return out


def mask_text(text: str, spans: Sequence[Span]) -> str:
    """Replace each span with ``[LABEL]``."""
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: s["start"]):
        out += [text[pos : s["start"]], f"[{s['label']}]"]
        pos = s["end"]
    out.append(text[pos:])
    return "".join(out)
