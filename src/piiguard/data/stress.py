"""Stress-test sets built from the test split, with gold spans kept correct after each change.

Every transform works through :func:`apply_edits`, which rewrites the text and remaps all span
offsets, so a span's ``value`` always equals ``text[start:end]`` afterwards (checked by
:func:`validate_doc`).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

Span = dict[str, Any]
Doc = dict[str, Any]
Edit = tuple[int, int, str]  # (start, end, replacement) in original-text coordinates

NAME_LABELS = {"GIVENNAME1", "GIVENNAME2", "LASTNAME1", "LASTNAME2", "LASTNAME3"}
ID_LABELS = {"IDCARD", "SOCIALNUMBER", "PASSPORT", "DRIVERLICENSE"}
DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


# --------------------------------------------------------------------------- core machinery
def validate_doc(doc: Doc) -> list[str]:
    """Return a list of problems (empty if every span is in bounds, sorted, disjoint, exact)."""
    problems = []
    text, prev_end = doc["text"], 0
    for sp in doc["spans"]:
        s, e = sp["start"], sp["end"]
        if not (0 <= s < e <= len(text)):
            problems.append(f"out of bounds {s}:{e}")
        elif text[s:e] != sp["value"]:
            problems.append(f"value mismatch at {s}:{e}: {text[s:e]!r} != {sp['value']!r}")
        if s < prev_end:
            problems.append(f"overlap or unsorted at {s}")
        prev_end = max(prev_end, e)
    return problems


def apply_edits(text: str, spans: Sequence[Span], edits: Sequence[Edit]) -> tuple[str, list[Span]]:
    """Apply non-overlapping text edits and remap span offsets.

    Each edit must either equal one span exactly (that span's text is replaced) or lie entirely
    outside every span (e.g. the gap between two names). Anything else raises ``ValueError``.
    """
    edits = sorted(edits)
    for (_s1, e1, _), (s2, _e2, _) in zip(edits, edits[1:], strict=False):
        if s2 < e1:
            raise ValueError("overlapping edits")
    exact = {(sp["start"], sp["end"]) for sp in spans}
    for s, e, _ in edits:
        if (s, e) in exact:
            continue
        if any(s < sp["end"] and e > sp["start"] for sp in spans):
            raise ValueError(f"edit {s}:{e} partially overlaps a span")

    def shift(p: int) -> int:
        return sum(len(n) - (e - s) for s, e, n in edits if e <= p)

    replacement = {(s, e): n for s, e, n in edits}
    out, pos = [], 0
    for s, e, n in edits:
        out += [text[pos:s], n]
        pos = e
    out.append(text[pos:])
    new_text = "".join(out)
    new_spans = []
    for sp in spans:
        ns = sp["start"] + shift(sp["start"])
        key = (sp["start"], sp["end"])
        ne = ns + len(replacement[key]) if key in replacement else sp["end"] + shift(sp["end"])
        new_spans.append({"start": ns, "end": ne, "label": sp["label"], "value": new_text[ns:ne]})
    return new_text, new_spans


# --------------------------------------------------------------------------- string transforms
def obfuscate_email(value: str) -> str:
    """``john.smith@mail.com`` -> ``john.smith at mail dot com``."""
    if "@" not in value:
        return value
    local, domain = value.split("@", 1)
    return f"{local} at {domain.replace('.', ' dot ')}"


def _chunks(chars: str, rng: np.random.Generator, lo: int, hi: int) -> list[str]:
    out, i = [], 0
    while i < len(chars):
        k = int(rng.integers(lo, hi + 1))
        out.append(chars[i : i + k])
        i += k
    return out


def obfuscate_phone(value: str, rng: np.random.Generator) -> str:
    """Insert spaces or dashes between digit groups, or spell the digits out as words."""
    digits = re.sub(r"\D", "", value)
    if len(digits) < 4:
        return value
    plus = value.strip().startswith("+")
    mode = str(rng.choice(["spaces", "dashes", "words"]))
    if mode == "words":
        words = " ".join(DIGIT_WORDS[int(d)] for d in digits)
        return ("plus " if plus else "") + words
    sep = " " if mode == "spaces" else "-"
    return ("+" if plus else "") + sep.join(_chunks(digits, rng, 2, 3))


def obfuscate_id(value: str, rng: np.random.Generator) -> str:
    """Re-group the alphanumeric characters of an ID with a separator (``-``, space or ``.``)."""
    core = re.sub(r"[^0-9A-Za-z]", "", value)
    if len(core) < 4:
        return value
    sep = str(rng.choice(["-", " ", "."]))
    return sep.join(_chunks(core, rng, 2, 4))


# --------------------------------------------------------------------------- edit builders
def _span_edits(doc: Doc, labels: set[str], fn: Callable[[str], str]) -> list[Edit]:
    edits = []
    for sp in doc["spans"]:
        if sp["label"] in labels:
            new = fn(sp["value"])
            if new != sp["value"]:
                edits.append((sp["start"], sp["end"], new))
    return edits


def email_edits(doc: Doc, rng: np.random.Generator) -> list[Edit]:
    """Edits that rewrite every EMAIL span as ``name at domain dot com``."""
    return _span_edits(doc, {"EMAIL"}, obfuscate_email)


def phone_edits(doc: Doc, rng: np.random.Generator) -> list[Edit]:
    """Edits that regroup or spell out every TEL span."""
    return _span_edits(doc, {"TEL"}, lambda v: obfuscate_phone(v, rng))


def id_edits(doc: Doc, rng: np.random.Generator) -> list[Edit]:
    """Edits that insert separators into ID-like spans."""
    return _span_edits(doc, ID_LABELS, lambda v: obfuscate_id(v, rng))


def name_edits(doc: Doc, rng: np.random.Generator) -> list[Edit]:
    """Lowercase all name spans, or widen the single space between adjacent name spans.

    The mode is drawn per document; when no adjacent pair with a one-space gap exists, the
    "extra spaces" mode falls back to lowercasing.
    """
    mode = str(rng.choice(["lower", "spaces"]))
    if mode == "spaces":
        edits: list[Edit] = []
        names = [sp for sp in doc["spans"] if sp["label"] in NAME_LABELS]
        for a, b in zip(names, names[1:], strict=False):
            if doc["text"][a["end"] : b["start"]] == " ":
                edits.append((a["end"], b["start"], " " * int(rng.integers(2, 5))))
        if edits:
            return edits
    return _span_edits(doc, NAME_LABELS, str.lower)


EDIT_BUILDERS: dict[str, Callable[[Doc, np.random.Generator], list[Edit]]] = {
    "email": email_edits,
    "phone": phone_edits,
    "name": name_edits,
    "id": id_edits,
}


# --------------------------------------------------------------------------- set builders
def build_obfuscated(docs: Sequence[Doc], kind: str, seed: int) -> list[Doc]:
    """Obfuscate ``docs``; ``kind`` is one of ``email, phone, name, id`` or ``all``.

    Only documents that actually change are kept. Each result records its source document id,
    the transform name, and how many edits were applied.
    """
    kinds = list(EDIT_BUILDERS) if kind == "all" else [kind]
    rng = np.random.default_rng(seed)
    out: list[Doc] = []
    for doc in docs:
        edits: list[Edit] = []
        for k in kinds:
            edits += EDIT_BUILDERS[k](doc, rng)
        if not edits:
            continue
        text, spans = apply_edits(doc["text"], doc["spans"], edits)
        new = {
            **doc,
            "id": f"{doc['id']}~{kind}",
            "text": text,
            "spans": spans,
            "source_id": doc["id"],
            "transform": kind,
            "n_changed": len(edits),
        }
        problems = validate_doc(new)
        if problems:
            raise AssertionError(f"{new['id']}: {problems[:3]}")
        out.append(new)
    return out


def rare_labels(
    train_counts: Counter, test_counts: Counter, fraction: float, min_test_support: int
) -> tuple[list[str], list[str], list[str]]:
    """Pick the least frequent ``fraction`` of labels by train frequency.

    Returns ``(rare, testable, untestable)``; a rare label is untestable if the test split has
    fewer than ``min_test_support`` examples of it.
    """
    ranked = sorted(train_counts, key=lambda lab: (train_counts[lab], lab))
    rare = ranked[: math.ceil(fraction * len(ranked))]
    testable = [lab for lab in rare if test_counts.get(lab, 0) >= min_test_support]
    return rare, testable, [lab for lab in rare if lab not in testable]


def build_rare(docs: Sequence[Doc], testable: Sequence[str]) -> list[Doc]:
    """Test documents containing at least one testable rare label.

    All spans stay in the document so the masker sees realistic text; ``eval_labels`` lists the
    labels the evaluation should score.
    """
    keep = set(testable)
    out = []
    for doc in docs:
        if any(sp["label"] in keep for sp in doc["spans"]):
            out.append(
                {**doc, "transform": "rare", "source_id": doc["id"], "eval_labels": sorted(keep)}
            )
    return out
