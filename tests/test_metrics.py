import math

import pytest

from piiguard.eval.metrics import counts_frame, doc_counts, summarize_frame


def sp(start, end, label, text):
    return {"start": start, "end": end, "label": label, "value": text[start:end]}


TEXT = "Hi John Smith, mail a@b.com now"
J = sp(3, 7, "GIVENNAME1", TEXT)  # John
S = sp(8, 13, "LASTNAME1", TEXT)  # Smith
E = sp(20, 27, "EMAIL", TEXT)  # a@b.com
GOLD = [J, S, E]


def summ(preds, gold=GOLD, text=TEXT, el=None):
    return summarize_frame(counts_frame([text], [gold], [preds], [el]))


def test_perfect_prediction():
    m = summ(GOLD)
    assert m["entity_f1"] == 1.0 and m["char_leak_rate"] == 0.0 and m["entity_leak_rate"] == 0.0
    assert m["type_agnostic_recall"] == 1.0


def test_nothing_predicted_everything_leaks():
    m = summ([])
    assert m["entity_f1"] == 0.0
    assert m["char_leak_rate"] == 1.0 and m["entity_leak_rate"] == 1.0
    assert m["type_agnostic_recall"] == 0.0


def test_wrong_type_is_f1_miss_but_not_a_leak():
    pred = [sp(3, 7, "LASTNAME1", TEXT), S, E]  # John masked with the wrong label
    m = summ(pred)
    assert m["entity_recall"] == pytest.approx(2 / 3) and m["entity_precision"] == pytest.approx(
        2 / 3
    )
    assert m["char_leak_rate"] == 0.0 and m["entity_leak_rate"] == 0.0
    assert m["type_agnostic_recall"] == 1.0  # exact span found, label ignored


def test_partial_cover_leaks_the_remaining_characters():
    # the mask for the email stops before ".com": 3 of its 7 characters leak
    pred = [J, S, sp(20, 24, "EMAIL", TEXT)]
    m = summ(pred)
    gold_chars = 4 + 5 + 7
    assert m["char_leak_rate"] == pytest.approx(3 / gold_chars)
    assert m["entity_leak_rate"] == pytest.approx(1 / 3)
    assert m["entity_f1"] < 1.0  # strict match fails for the truncated span


def test_oversized_mask_covers_but_costs_precision():
    pred = [sp(3, 13, "GIVENNAME1", TEXT), E]  # one big span over "John Smith"
    m = summ(pred)
    assert m["char_leak_rate"] == 0.0  # everything hidden
    assert m["entity_precision"] == pytest.approx(1 / 2)  # only the email is exact
    assert m["entity_recall"] == pytest.approx(1 / 3)


def test_whitespace_inside_gold_span_is_not_counted():
    text = "ID 12 34 end"
    gold = [sp(3, 8, "IDCARD", text)]  # "12 34": 4 non-space chars
    c = doc_counts(text, gold, [sp(3, 5, "IDCARD", text)])
    assert c["gold_chars"] == 4 and c["leaked_chars"] == 2


def test_eval_labels_restrict_gold_and_strict_precision_only():
    # scoring EMAIL only: a wrong-label guess on a name does not hurt precision
    pred = [sp(3, 7, "LASTNAME1", TEXT), E]
    m = summ(pred, el=["EMAIL"])
    assert m["n_gold_entities"] == 1 and m["entity_f1"] == 1.0 and m["char_leak_rate"] == 0.0
    m2 = summ([], el=["EMAIL"])
    assert m2["char_leak_rate"] == 1.0


def test_recall_by_type_strict_and_coverage():
    pred = [sp(3, 7, "LASTNAME1", TEXT), S]  # John: covered but wrong label; Smith ok; email missed
    m = summ(pred)
    assert m["recall_strict_by_type"]["GIVENNAME1"] == 0.0
    assert m["coverage_recall_by_type"]["GIVENNAME1"] == 1.0
    assert m["recall_strict_by_type"]["LASTNAME1"] == 1.0
    assert m["coverage_recall_by_type"]["EMAIL"] == 0.0


def test_aggregation_is_over_all_documents_not_a_mean_of_means():
    t2 = "x y@z.com"
    g2 = [sp(2, 9, "EMAIL", t2)]
    df = counts_frame([TEXT, t2], [GOLD, g2], [GOLD, []])  # doc1 perfect, doc2 fully leaked
    m = summarize_frame(df)
    assert m["char_leak_rate"] == pytest.approx(7 / (16 + 7))
    assert m["n_docs"] == 2
    boot = summarize_frame(df.iloc[[0, 0]])  # a resample is just another frame
    assert boot["char_leak_rate"] == 0.0


def test_no_gold_gives_nan_rates_not_a_crash():
    m = summarize_frame(counts_frame(["hello"], [[]], [[]]))
    assert math.isnan(m["char_leak_rate"]) and math.isnan(m["entity_f1"])
