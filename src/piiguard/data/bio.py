"""BIO token labels aligned to a tokenizer via character offset mappings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

IGNORE = -100


def build_label_list(labels: Sequence[str]) -> list[str]:
    """``["O", "B-X", "I-X", ...]`` with labels sorted for a stable id assignment."""
    out = ["O"]
    for lab in sorted(set(labels)):
        out += [f"B-{lab}", f"I-{lab}"]
    return out


def spans_to_bio(
    offsets: Sequence[tuple[int, int]],
    spans: Sequence[dict[str, Any]],
    label2id: dict[str, int],
) -> list[int]:
    """Label each token B-/I-/O from character spans.

    A token overlapping a span gets that span's label: ``B-`` for the first token of the span,
    ``I-`` for the rest. Special tokens (offset ``(0, 0)``) get ``IGNORE`` so the loss skips them.
    """
    ordered = sorted(spans, key=lambda sp: sp["start"])
    tags: list[int] = []
    prev_span = -1
    for s, e in offsets:
        if s == e:
            tags.append(IGNORE)
            continue
        hit = -1
        for k, sp in enumerate(ordered):
            if sp["start"] < e and sp["end"] > s:
                hit = k
                break
        if hit < 0:
            tags.append(label2id["O"])
            prev_span = -1
            continue
        prefix = "I" if hit == prev_span else "B"
        tags.append(label2id[f"{prefix}-{ordered[hit]['label']}"])
        prev_span = hit
    return tags


def bio_to_spans(
    text: str,
    offsets: Sequence[tuple[int, int]],
    tag_ids: Sequence[int],
    id2label: dict[int, str],
) -> list[dict[str, Any]]:
    """Decode token tags back into character spans (whitespace trimmed from both edges)."""
    spans: list[dict[str, Any]] = []
    cur: list[Any] | None = None  # [label, start, end]

    def close() -> None:
        nonlocal cur
        if cur is not None:
            s, e = cur[1], cur[2]
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if e > s:
                spans.append({"start": s, "end": e, "label": cur[0], "value": text[s:e]})
        cur = None

    for (s, e), t in zip(offsets, tag_ids, strict=True):
        if t == IGNORE or s == e:
            continue
        tag = id2label[int(t)]
        if tag == "O":
            close()
            continue
        prefix, lab = tag.split("-", 1)
        if prefix == "B" or cur is None or cur[0] != lab:
            close()
            cur = [lab, s, e]
        else:
            cur[2] = e
    close()
    return spans


def roundtrip_ok(
    text: str,
    spans: Sequence[dict[str, Any]],
    offsets: Sequence[tuple[int, int]],
    label2id: dict[str, int],
) -> list[bool]:
    """Per gold span: does BIO encode then decode reproduce it exactly (start, end, label)?"""
    id2label = {v: k for k, v in label2id.items()}
    decoded = bio_to_spans(text, offsets, spans_to_bio(offsets, spans, label2id), id2label)
    got = {(d["start"], d["end"], d["label"]) for d in decoded}
    return [(sp["start"], sp["end"], sp["label"]) in got for sp in spans]
