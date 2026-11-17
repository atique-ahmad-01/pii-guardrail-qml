import json
import re

import pytest

from piiguard.data.bio import IGNORE, bio_to_spans, build_label_list, roundtrip_ok, spans_to_bio
from piiguard.data.load import (
    fix_spans,
    rarest_label_key,
    split_docs,
    stratified_sample,
    to_docs,
)
from piiguard.data.sentences import balance, label_sentences, sentence_table, split_sentences
from piiguard.data.sft import build_prompt, build_target


def regex_offsets(text):
    """Offsets of a toy tokenizer, with one special token (0, 0) at each end."""
    return [(0, 0)] + [m.span() for m in re.finditer(r"\w+|[^\w\s]", text)] + [(0, 0)]


def span(text, value, label):
    s = text.index(value)
    return {"start": s, "end": s + len(value), "label": label, "value": value}


TEXT = "Hi, I am John Smith. Mail john.smith@mail.com or call 555-0142 today."
SPANS = [
    span(TEXT, "John", "GIVENNAME1"),
    span(TEXT, "Smith", "LASTNAME1"),
    span(TEXT, "john.smith@mail.com", "EMAIL"),
    span(TEXT, "555-0142", "TEL"),
]


# ------------------------------------------------------------------ BIO round trip
def test_bio_roundtrip_every_gold_span():
    labels = build_label_list(sp["label"] for sp in SPANS)
    label2id = {lab: i for i, lab in enumerate(labels)}
    assert all(roundtrip_ok(TEXT, SPANS, regex_offsets(TEXT), label2id))


def test_bio_tags_b_then_i_and_ignore_special():
    labels = build_label_list(["EMAIL"])
    label2id = {lab: i for i, lab in enumerate(labels)}
    text = "mail a.b@c.com now"
    sp = [span(text, "a.b@c.com", "EMAIL")]
    tags = spans_to_bio(regex_offsets(text), sp, label2id)
    names = [None if t == IGNORE else labels[t] for t in tags]
    assert names[0] is None and names[-1] is None
    assert names[1] == "O" and names[2] == "B-EMAIL"
    assert all(n == "I-EMAIL" for n in names[3:9])
    assert names[9] == "O"


def test_bio_decode_trims_whitespace_and_splits_adjacent_entities():
    labels = build_label_list(["A", "B"])
    label2id = {lab: i for i, lab in enumerate(labels)}
    text = "xx yy"
    sp = [
        {"start": 0, "end": 2, "label": "A", "value": "xx"},
        {"start": 3, "end": 5, "label": "B", "value": "yy"},
    ]
    off = regex_offsets(text)
    out = bio_to_spans(text, off, spans_to_bio(off, sp, label2id), dict(enumerate(labels)))
    assert [(o["start"], o["end"], o["label"]) for o in out] == [(0, 2, "A"), (3, 5, "B")]


def test_roundtrip_flags_span_starting_mid_token():
    labels = build_label_list(["X"])
    label2id = {lab: i for i, lab in enumerate(labels)}
    text = "abcdef gh"
    sp = [{"start": 3, "end": 6, "label": "X", "value": "def"}]  # inside one token
    assert roundtrip_ok(text, sp, regex_offsets(text), label2id) == [False]


# ------------------------------------------------------------------ span repair
def test_fix_spans_keeps_ok_relocates_and_drops():
    text = "call 555-0142 now, or 555-0142 later"
    raw = [
        {"start": 5, "end": 13, "label": "TEL", "value": "555-0142"},  # correct
        {"start": 20, "end": 28, "label": "TEL", "value": "555-0142"},  # off by two
        {"start": 0, "end": 4, "label": "NAME", "value": "ghost"},  # not in text
    ]
    spans, stats = fix_spans(text, raw)
    assert [(s["start"], s["end"]) for s in spans] == [(5, 13), (22, 30)]
    assert stats["ok"] == 1 and stats["relocated"] == 1 and stats["dropped_value_not_found"] == 1
    assert all(text[s["start"] : s["end"]] == s["value"] for s in spans)


def test_fix_spans_drops_overlap():
    text = "abcdefgh"
    raw = [
        {"start": 0, "end": 5, "label": "A", "value": "abcde"},
        {"start": 3, "end": 8, "label": "B", "value": "defgh"},
    ]
    spans, stats = fix_spans(text, raw)
    assert len(spans) == 1 and spans[0]["label"] == "A" and stats["dropped_overlap"] == 1


# ------------------------------------------------------------------ sentences
def test_split_sentences_offsets_exact():
    sents = split_sentences(TEXT)
    assert [TEXT[a:b] for a, b in sents] == [
        "Hi, I am John Smith.",
        "Mail john.smith@mail.com or call 555-0142 today.",
    ]


def test_split_sentences_newlines_and_decimals():
    text = "Line one\n\n  Pi is 3.14 ok. Next"
    assert [text[a:b] for a, b in split_sentences(text)] == ["Line one", "Pi is 3.14 ok.", "Next"]


def test_sentence_labels_and_overlap_rule():
    doc = {"id": "d1", "text": TEXT, "spans": SPANS}
    rows = label_sentences(doc)
    assert [r["label"] for r in rows] == [1, 1]
    neg = {"id": "d2", "text": "Nothing here. Still nothing.", "spans": []}
    assert [r["label"] for r in label_sentences(neg)] == [0, 0]
    # a span that only touches the boundary of the next sentence does not make it positive
    t = "Ab cd. Ef gh."
    sp = [{"start": 3, "end": 5, "label": "X", "value": "cd"}]
    assert [r["label"] for r in label_sentences({"id": "d", "text": t, "spans": sp})] == [1, 0]


def test_balance_equal_classes_and_seeded():
    docs = [
        {
            "id": f"d{i}",
            "text": "Pos a. Neg b. Neg c.",
            "spans": [{"start": 0, "end": 3, "label": "X", "value": "Pos"}],
        }
        for i in range(10)
    ]
    df = sentence_table(docs)
    b1, b2 = balance(df, 1), balance(df, 1)
    assert (b1.label == 1).sum() == (b1.label == 0).sum() == 10
    assert b1.equals(b2)


# ------------------------------------------------------------------ sampling
def test_stratified_sample_deterministic_proportional():
    keys = ["a"] * 80 + ["b"] * 20
    i1, i2 = stratified_sample(keys, 10, 7), stratified_sample(keys, 10, 7)
    assert i1 == i2 and len(i1) == 10
    assert sum(1 for i in i1 if keys[i] == "b") == 2
    assert stratified_sample(keys, 10, 8) != i1


def make_docs(n, tag):
    docs = []
    for i in range(n):
        sp = [] if i % 5 == 0 else [{"start": 0, "end": 1, "label": f"L{i % 3}", "value": "x"}]
        docs.append({"id": f"{tag}{i}", "language": "English", "text": "x y", "spans": sp})
    return docs


def test_split_docs_disjoint_and_sized():
    s = split_docs(make_docs(200, "tr"), make_docs(100, "te"), 60, 20, 30, seed=3)
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (60, 20, 30)
    tr, va = {d["id"] for d in s["train"]}, {d["id"] for d in s["val"]}
    assert not tr & va
    assert all(d["id"].startswith("te") for d in s["test"])
    assert len(rarest_label_key(s["train"])) == 60


def test_to_docs_applies_repair():
    rows = [
        {
            "id": "1",
            "language": "English",
            "source_text": "hi bob",
            "privacy_mask": [{"start": 0, "end": 3, "label": "N", "value": "bob"}],
        }
    ]
    docs, stats = to_docs(rows)
    assert docs[0]["spans"][0]["start"] == 3 and stats["relocated"] == 1


# ------------------------------------------------------------------ SFT pairs
def test_sft_target_is_valid_json_in_order_with_exact_substrings():
    shuffled = [SPANS[3], SPANS[0], SPANS[2], SPANS[1]]
    items = json.loads(build_target(TEXT, shuffled))
    assert [i["text"] for i in items] == ["John", "Smith", "john.smith@mail.com", "555-0142"]
    assert all(i["text"] in TEXT for i in items)
    assert json.loads(build_target("clean text", [])) == []
    assert TEXT in build_prompt(TEXT)


@pytest.mark.parametrize("n", [0, 1])
def test_stratified_sample_edge_sizes(n):
    assert len(stratified_sample(["a", "b", "a"], n, 0)) == n


def test_to_docs_excludes_documents_with_corrupted_spans():
    good = {
        "id": "g",
        "language": "English",
        "source_text": "hi bob",
        "privacy_mask": [{"start": 3, "end": 6, "label": "N", "value": "bob"}],
    }
    bad = {
        "id": "b",
        "language": "English",
        "source_text": "hi bob and ann",
        "privacy_mask": [
            {"start": 3, "end": 6, "label": "N", "value": "bob"},
            {"start": 0, "end": 3, "label": "N", "value": "[ann_X("},
        ],
    }
    docs, stats = to_docs([good, bad])
    assert [d["id"] for d in docs] == ["g"]
    assert stats["docs_excluded"] == 1 and stats["docs_kept"] == 1
