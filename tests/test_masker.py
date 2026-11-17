import json

import torch

from piiguard.data.sft import build_prompt, build_target
from piiguard.masker.llm_sft import IGNORE, PadCollator, build_sft_example, chat_text
from piiguard.masker.spans import map_items_to_spans, mask_text, parse_llm_output


# ------------------------------------------------------------------ JSON parsing
def test_parse_valid_list():
    items, ok = parse_llm_output(
        '[{"type": "EMAIL", "text": "a@b.com"}, {"type": "TEL", "text": "555"}]'
    )
    assert ok and [i["text"] for i in items] == ["a@b.com", "555"]


def test_parse_empty_list_is_valid():
    assert parse_llm_output("[]") == ([], True)


def test_parse_tolerates_fences_and_surrounding_text():
    raw = (
        'Sure! Here you go:\n```json\n[{"type": "TEL", "text": "555-0142"}]\n```\nHope that helps.'
    )
    items, ok = parse_llm_output(raw)
    assert ok and items == [{"type": "TEL", "text": "555-0142"}]
    items, ok = parse_llm_output('Result: [{"type":"EMAIL","text":"x@y.z"}] done')
    assert ok and len(items) == 1


def test_parse_invalid_counts_as_failure_with_no_spans():
    for bad in [
        "",
        "no json here",
        '[{"type": "EMAIL", "text": "a@b.com"',  # truncated: no closing bracket
        '[{"type": "EMAIL", "text": "a@b.com"}, {"type": "TEL", "text":}]',  # broken json
        '{"type": "EMAIL", "text": "a"}',  # not a list
        '[{"type": "EMAIL"}]',  # missing text
        '[{"type": "EMAIL", "text": 5}]',  # wrong value type
        '["a@b.com"]',  # list of strings
    ]:
        assert parse_llm_output(bad) == ([], False), bad


def test_parse_drops_empty_text_items():
    items, ok = parse_llm_output('[{"type": "TEL", "text": ""}, {"type": "TEL", "text": "1"}]')
    assert ok and [i["text"] for i in items] == ["1"]


def test_parse_round_trips_the_sft_target():
    text = "Mail a@b.com or John"
    spans = [
        {"start": 5, "end": 12, "label": "EMAIL", "value": "a@b.com"},
        {"start": 16, "end": 20, "label": "GIVENNAME1", "value": "John"},
    ]
    items, ok = parse_llm_output(build_target(text, spans))
    assert ok
    out = map_items_to_spans(text, items)
    assert [(s["start"], s["end"], s["label"]) for s in out] == [
        (5, 12, "EMAIL"),
        (16, 20, "GIVENNAME1"),
    ]


# ------------------------------------------------------------------ text -> span mapping
def test_map_all_occurrences_and_masking():
    text = "Call 555 now. Again 555 later."
    spans = map_items_to_spans(text, [{"type": "TEL", "text": "555"}])
    assert [(s["start"], s["end"]) for s in spans] == [(5, 8), (20, 23)]
    assert mask_text(text, spans) == "Call [TEL] now. Again [TEL] later."


def test_map_hallucinated_text_matches_nothing():
    assert map_items_to_spans("hello world", [{"type": "X", "text": "absent"}]) == []


def test_map_overlaps_prefer_earlier_then_longer():
    text = "John Smith"
    items = [
        {"type": "A", "text": "Smith"},
        {"type": "B", "text": "John Smith"},
        {"type": "C", "text": "John"},
    ]
    spans = map_items_to_spans(text, items)
    assert [(s["start"], s["end"], s["label"]) for s in spans] == [(0, 10, "B")]


def test_known_limitation_same_string_is_masked_everywhere():
    text = "Mr May met May in May."  # only the first is a name, but all three are masked
    spans = map_items_to_spans(text, [{"type": "LASTNAME1", "text": "May"}])
    assert len(spans) == 3


# ------------------------------------------------------------------ SFT examples
class StubTok:
    """Whitespace tokenizer with a trivial chat template, enough to test label masking offline."""

    eos_token = "<eos>"
    pad_token_id = 0
    vocab: dict[str, int] = {}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "<user> " + msgs[0]["content"] + " <assistant>"

    def __call__(self, text, add_special_tokens=False):
        words = text.replace("<eos>", " <eos>").split()
        ids = [self.vocab.setdefault(w, len(self.vocab) + 1) for w in words]
        return {"input_ids": ids}


def test_sft_labels_cover_only_the_target_and_eos():
    tok = StubTok()
    ex = build_sft_example(tok, "find pii", '[{"type": "EMAIL", "text": "a"}]', 100)
    n_prompt = len(tok(chat_text(tok, "find pii"))["input_ids"])
    assert ex["labels"][:n_prompt] == [IGNORE] * n_prompt
    assert all(x != IGNORE for x in ex["labels"][n_prompt:])
    assert ex["labels"][-1] == tok.vocab["<eos>"]
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])


def test_sft_truncates_target_never_prompt():
    tok = StubTok()
    long_target = " ".join(f"w{i}" for i in range(50))
    ex = build_sft_example(tok, "a b c", long_target, 12)
    assert len(ex["input_ids"]) == 12
    n_prompt = len(tok(chat_text(tok, "a b c"))["input_ids"])
    assert ex["labels"][:n_prompt] == [IGNORE] * n_prompt


def test_collator_pads_right_and_masks_padding_in_loss():
    batch = [
        {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [IGNORE, 6, 7]},
        {"input_ids": [8], "attention_mask": [1], "labels": [8]},
    ]
    out = PadCollator(pad_id=0)(batch)
    assert out["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert out["labels"].tolist() == [[IGNORE, 6, 7], [8, IGNORE, IGNORE]]
    assert out["input_ids"].dtype == torch.long


def test_prompt_contains_instruction_and_text():
    p = build_prompt("secret text")
    assert "secret text" in p and "JSON" in p
    assert json.loads(build_target("x", [])) == []
