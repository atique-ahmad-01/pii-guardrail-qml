from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from piiguard.data.context_examples import EXAMPLES, build_context_docs
from piiguard.data.load import frame_to_docs
from piiguard.data.stress import (
    apply_edits,
    build_obfuscated,
    build_rare,
    obfuscate_email,
    obfuscate_id,
    obfuscate_phone,
    rare_labels,
    validate_doc,
)

TEXT = "Hi John Smith, mail john.smith@mail.com or call +49 151 234 5678. ID: AB123456 ok."


def sp(text, value, label):
    s = text.index(value)
    return {"start": s, "end": s + len(value), "label": label, "value": value}


SPANS = [
    sp(TEXT, "John", "GIVENNAME1"),
    sp(TEXT, "Smith", "LASTNAME1"),
    sp(TEXT, "john.smith@mail.com", "EMAIL"),
    sp(TEXT, "+49 151 234 5678", "TEL"),
    sp(TEXT, "AB123456", "IDCARD"),
]
DOC = {"id": "d1", "language": "English", "text": TEXT, "spans": SPANS}


# ------------------------------------------------------------------ core edit machinery
def test_validate_doc_accepts_good_and_flags_bad():
    assert validate_doc(DOC) == []
    bad = {**DOC, "spans": [{**SPANS[0], "start": 4, "end": 7}]}
    assert validate_doc(bad)


def test_apply_edits_grow_shrink_and_gap_edit():
    edits = [
        (SPANS[2]["start"], SPANS[2]["end"], "x"),  # shrinks the email
        (SPANS[0]["end"], SPANS[1]["start"], "    "),  # widens the gap between the names
        (SPANS[4]["start"], SPANS[4]["end"], "AB-12-34-56"),  # grows the id
    ]
    text, spans = apply_edits(TEXT, SPANS, edits)
    doc = {**DOC, "text": text, "spans": spans}
    assert validate_doc(doc) == []
    assert [s["value"] for s in spans] == ["John", "Smith", "x", "+49 151 234 5678", "AB-12-34-56"]
    assert "John    Smith" in text


def test_apply_edits_rejects_partial_overlap_and_overlapping_edits():
    with pytest.raises(ValueError):
        apply_edits(TEXT, SPANS, [(SPANS[2]["start"] + 1, SPANS[2]["end"], "z")])
    with pytest.raises(ValueError):
        apply_edits(TEXT, SPANS, [(0, 5, "a"), (3, 8, "b")])


# ------------------------------------------------------------------ string transforms
def test_email_obfuscation_format():
    assert (
        obfuscate_email("john.smith@mail.example.com") == "john.smith at mail dot example dot com"
    )
    assert obfuscate_email("no-at-sign") == "no-at-sign"


def test_phone_obfuscation_modes_keep_digits():
    seen = set()
    for seed in range(30):
        out = obfuscate_phone("+49 151 234 5678", np.random.default_rng(seed))
        seen.add("words" if out.startswith("plus ") else "digits")
        if out.startswith("plus "):
            assert out.split()[1:3] == ["four", "nine"]
        else:
            assert "".join(c for c in out if c.isdigit()) == "491512345678"
            assert out.startswith("+")
    assert seen == {"words", "digits"}


def test_id_obfuscation_keeps_alnum_content():
    out = obfuscate_id("AB123456", np.random.default_rng(0))
    assert "".join(c for c in out if c.isalnum()) == "AB123456"
    assert not out.isalnum()  # a separator was added


# ------------------------------------------------------------------ set builders
@pytest.mark.parametrize("kind", ["email", "phone", "name", "id", "all"])
def test_every_transform_keeps_offsets_correct(kind):
    docs = build_obfuscated([DOC], kind, seed=3)
    assert len(docs) == 1
    d = docs[0]
    assert validate_doc(d) == []
    assert d["text"] != TEXT and d["source_id"] == "d1" and d["n_changed"] >= 1
    assert [s["label"] for s in d["spans"]] == [s["label"] for s in SPANS]  # labels untouched


def test_transforms_are_deterministic_and_seed_sensitive():
    a = build_obfuscated([DOC], "all", seed=5)[0]["text"]
    assert a == build_obfuscated([DOC], "all", seed=5)[0]["text"]
    outs = {build_obfuscated([DOC], "all", seed=s)[0]["text"] for s in range(10)}
    assert len(outs) > 1


def test_docs_without_target_labels_are_dropped():
    plain = {"id": "p", "language": "English", "text": "hello world", "spans": []}
    assert build_obfuscated([plain, DOC], "email", seed=1)[0]["source_id"] == "d1"
    assert len(build_obfuscated([plain], "all", seed=1)) == 0


def test_name_lowercase_or_spacing_both_valid():
    seen_lower = seen_space = False
    for seed in range(20):
        d = build_obfuscated([DOC], "name", seed=seed)[0]
        assert validate_doc(d) == []
        if d["spans"][0]["value"] == "john":
            seen_lower = True
        if "John  " in d["text"]:
            seen_space = True
    assert seen_lower and seen_space


def test_rare_labels_selection_and_untestable():
    train = Counter(
        {"A": 1, "B": 5, "C": 50, "D": 60, "E": 70, "F": 80, "G": 90, "H": 100, "I": 110, "J": 120}
    )
    test = Counter({"A": 0, "B": 9, "C": 12})
    rare, testable, untestable = rare_labels(train, test, 0.30, min_test_support=5)
    assert rare == ["A", "B", "C"]
    assert testable == ["B", "C"] and untestable == ["A"]
    docs = [
        {
            "id": "1",
            "language": "English",
            "text": "x",
            "spans": [{"start": 0, "end": 1, "label": "B", "value": "x"}],
        },
        {
            "id": "2",
            "language": "English",
            "text": "x",
            "spans": [{"start": 0, "end": 1, "label": "H", "value": "x"}],
        },
    ]
    out = build_rare(docs, testable)
    assert [d["id"] for d in out] == ["1"] and out[0]["eval_labels"] == ["B", "C"]


# ------------------------------------------------------------------ context-only examples
def test_context_examples_are_valid():
    assert len(EXAMPLES) == 50
    assert len({t for t, _ in EXAMPLES}) == 50
    docs = build_context_docs()
    assert all(validate_doc(d) == [] for d in docs)
    assert all(d["spans"][0]["label"] == "CONTEXT_ID" for d in docs)


# ------------------------------------------------------------------ on the real processed data
@pytest.mark.skipif(
    not (Path(__file__).parents[1] / "data/processed_smoke/test.parquet").exists(),
    reason="run scripts/prepare_data.py --config configs/smoke.yaml first",
)
def test_all_transforms_valid_on_real_test_split():
    test = frame_to_docs(
        pd.read_parquet(Path(__file__).parents[1] / "data/processed_smoke/test.parquet")
    )
    for kind in ["email", "phone", "name", "id", "all"]:
        docs = build_obfuscated(test, kind, seed=11)
        assert docs, kind
        assert all(validate_doc(d) == [] for d in docs)
        for d in docs:
            src = next(x for x in test if x["id"] == d["source_id"])
            assert len(d["spans"]) == len(src["spans"])
