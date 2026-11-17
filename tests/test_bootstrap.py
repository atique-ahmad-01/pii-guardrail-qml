import numpy as np
import pytest

from piiguard.eval.bootstrap import (
    bootstrap_ci,
    compare,
    holm,
    metrics_from_sums,
    point_estimate,
)
from piiguard.eval.metrics import counts_frame, summarize_frame


def make_frames(n=200, leak_p=0.1, seed=0):
    rng = np.random.default_rng(seed)
    texts = ["abcdefghij"] * n
    gold = [[{"start": 0, "end": 5, "label": "X", "value": "abcde"}] for _ in range(n)]
    pred = [
        [] if rng.random() < leak_p else [{"start": 0, "end": 5, "label": "X", "value": "abcde"}]
        for _ in range(n)
    ]
    df = counts_frame(texts, gold, pred)
    df.insert(0, "id", [str(i) for i in range(n)])
    return df


def test_point_estimate_matches_summarize_frame():
    df = make_frames()
    pt = point_estimate([df])
    ref = summarize_frame(df.drop(columns="id"))
    for k in ("entity_f1", "char_leak_rate", "entity_leak_rate", "type_agnostic_recall"):
        assert pt[k] == pytest.approx(ref[k])


def test_ci_contains_point_and_is_reproducible():
    df = make_frames(300, 0.2)
    a = bootstrap_ci([df], n_boot=400, seed=1)
    b = bootstrap_ci([df], n_boot=400, seed=1)
    assert a == b
    r = a["char_leak_rate"]
    assert r["lo"] <= r["point"] <= r["hi"]
    assert 0.1 < r["point"] < 0.3


def test_ci_narrows_with_more_documents():
    small = bootstrap_ci([make_frames(50, 0.2)], n_boot=400)["char_leak_rate"]
    big = bootstrap_ci([make_frames(2000, 0.2)], n_boot=400)["char_leak_rate"]
    assert (big["hi"] - big["lo"]) < (small["hi"] - small["lo"])


def test_paired_detects_a_real_difference_and_not_a_null_one():
    a, b = make_frames(400, 0.30, seed=1), make_frames(400, 0.05, seed=2)
    r = compare([a], [b], paired=True, n_boot=500)["char_leak_rate"]
    assert r["diff"] > 0.15 and r["lo"] > 0 and r["p"] < 0.01
    same = compare([a], [a], paired=True, n_boot=300)["char_leak_rate"]
    assert same["diff"] == 0 and same["lo"] == 0 == same["hi"]


def test_paired_requires_identical_documents():
    a, b = make_frames(10), make_frames(10)
    b["id"] = [f"z{i}" for i in range(10)]
    with pytest.raises(ValueError):
        compare([a], [b], paired=True, n_boot=10)


def test_unpaired_allows_different_documents():
    a, b = make_frames(300, 0.3, seed=1), make_frames(250, 0.05, seed=2)
    b["id"] = [f"z{i}" for i in range(len(b))]
    r = compare([a], [b], paired=False, n_boot=400)["char_leak_rate"]
    assert r["diff"] > 0 and r["p"] < 0.05


def test_multiple_seeds_are_averaged():
    f1, f2 = make_frames(100, 0.0, seed=1), make_frames(100, 0.4, seed=2)
    pt = point_estimate([f1, f2])
    assert pt["char_leak_rate"] == pytest.approx(
        (point_estimate([f1])["char_leak_rate"] + point_estimate([f2])["char_leak_rate"]) / 2
    )


def test_metrics_from_sums_handles_zero_denominators():
    out = metrics_from_sums(np.zeros(8))
    assert all(np.isnan(v) for v in out.values())


def test_holm_adjustment():
    adj = holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
