import numpy as np
import pytest

from piiguard.pipeline.assemble import (
    gate_sentence_stats,
    gated_spans,
    pick_threshold,
    shift_spans,
)


def sp(a, b, label="EMAIL"):
    return {"start": a, "end": b, "label": label, "value": "x"}


def test_shift_spans_moves_offsets_only():
    out = shift_spans([sp(2, 5)], 10)
    assert (out[0]["start"], out[0]["end"], out[0]["label"]) == (12, 15, "EMAIL")


def test_gated_spans_only_keeps_flagged_sentences():
    offs = [(0, 10), (11, 20), (21, 30)]
    spans = [[sp(1, 3)], [sp(0, 4)], [sp(2, 5)]]
    out = gated_spans(offs, spans, [True, False, True])
    assert [(s["start"], s["end"]) for s in out] == [(1, 3), (23, 26)]
    assert gated_spans(offs, spans, [False] * 3) == []


def test_pick_threshold_keeps_target_recall_of_positives():
    y = np.array([1] * 100 + [0] * 100)
    p = np.concatenate([np.linspace(0.01, 1.0, 100), np.linspace(0.0, 0.5, 100)])
    thr = pick_threshold(y, p, 0.99)
    assert (p[y == 1] >= thr).mean() >= 0.99
    assert (p[y == 1] >= thr + 1e-9).mean() < 1.0 or thr == p[y == 1].max()


def test_pick_threshold_target_one_never_drops_a_positive():
    y = np.array([1, 1, 1, 0])
    p = np.array([0.2, 0.9, 0.5, 0.1])
    assert pick_threshold(y, p, 1.0) == pytest.approx(0.2)


def test_pick_threshold_without_positives_is_default():
    assert pick_threshold(np.zeros(5), np.random.rand(5)) == 0.5


def test_gate_stats():
    y = np.array([1, 1, 0, 0])
    f = np.array([True, False, True, False])
    s = gate_sentence_stats(y, f)
    assert s["gate_recall"] == 0.5 and s["gate_precision"] == 0.5
    assert s["skipped_fraction"] == 0.5 and s["n_sentences"] == 4
