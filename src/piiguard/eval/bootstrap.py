"""Bootstrap confidence intervals and significance tests over documents.

Inputs are per-document count frames from ``piiguard.eval.metrics.counts_frame``. Documents are
the resampling unit. When several seeds are given (same documents, one frame per seed), each
bootstrap draw uses the *same* document resample for every seed and averages the metric over
seeds, so the interval reflects document sampling noise on the seed-averaged metric.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

COLS = ["tp", "fp", "fn", "agn_tp", "gold_ents", "leaked_ents", "gold_chars", "leaked_chars"]
METRICS = ["entity_f1", "char_leak_rate", "entity_leak_rate", "type_agnostic_recall"]


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def metrics_from_sums(s: np.ndarray) -> dict[str, np.ndarray]:
    """Headline metrics from summed counts of shape ``(..., 8)`` (column order = ``COLS``)."""
    tp, fp, fn, agn, ge, le, gc, lc = (s[..., i] for i in range(8))
    return {
        "entity_f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "char_leak_rate": _ratio(lc, gc),
        "entity_leak_rate": _ratio(le, ge),
        "type_agnostic_recall": _ratio(agn, ge),
    }


def _matrix(frames: Sequence[pd.DataFrame]) -> np.ndarray:
    """Array ``(n_seeds, n_docs, 8)``; frames must hold the same documents in the same order."""
    ids = [tuple(f["id"]) for f in frames]
    if any(i != ids[0] for i in ids[1:]):
        raise ValueError("frames must contain the same documents in the same order")
    return np.stack([f[COLS].to_numpy(dtype=float) for f in frames])


def _draws(
    m: np.ndarray, n_boot: int, rng: np.random.Generator, chunk: int = 200
) -> dict[str, np.ndarray]:
    """Bootstrap distribution of each metric (mean over seeds), shape ``(n_boot,)``."""
    n = m.shape[1]
    out: dict[str, list[np.ndarray]] = {k: [] for k in METRICS}
    for i in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - i), n))
        sums = m[:, idx, :].sum(axis=2)  # (seeds, draws, 8)
        for k, v in metrics_from_sums(sums).items():
            out[k].append(np.nanmean(v, axis=0))
    return {k: np.concatenate(v) for k, v in out.items()}


def point_estimate(frames: Sequence[pd.DataFrame]) -> dict[str, float]:
    """Seed-averaged metrics on the full document set."""
    sums = _matrix(frames).sum(axis=1)  # (seeds, 8)
    return {k: float(np.nanmean(v)) for k, v in metrics_from_sums(sums).items()}


def bootstrap_ci(
    frames: Sequence[pd.DataFrame], n_boot: int = 1000, ci: float = 0.95, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Percentile CI for each metric: ``{metric: {"point", "lo", "hi"}}``."""
    rng = np.random.default_rng(seed)
    pt, d = point_estimate(frames), _draws(_matrix(frames), n_boot, rng)
    a = (1 - ci) / 2
    return {
        k: {
            "point": pt[k],
            "lo": float(np.nanquantile(d[k], a)),
            "hi": float(np.nanquantile(d[k], 1 - a)),
        }
        for k in METRICS
    }


def _p_two_sided(diff: np.ndarray) -> float:
    d = diff[~np.isnan(diff)]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(min(1.0, max(p, 1.0 / (len(d) + 1))))


def compare(
    a: Sequence[pd.DataFrame],
    b: Sequence[pd.DataFrame],
    paired: bool,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Difference ``a - b`` per metric with CI and a two-sided bootstrap p-value.

    ``paired=True`` resamples the same documents for both systems (they must hold identical
    documents). ``paired=False`` resamples each independently (different document sets).
    """
    rng = np.random.default_rng(seed)
    ma, mb = _matrix(a), _matrix(b)
    if paired:
        if tuple(a[0]["id"]) != tuple(b[0]["id"]):
            raise ValueError("paired comparison needs identical documents")
        both = np.concatenate([ma, mb], axis=0)
        d = _draws(both, n_boot, rng)  # placeholder shape check, replaced below
        n = ma.shape[1]
        da: dict[str, list[np.ndarray]] = {k: [] for k in METRICS}
        db: dict[str, list[np.ndarray]] = {k: [] for k in METRICS}
        for i in range(0, n_boot, 200):
            idx = rng.integers(0, n, size=(min(200, n_boot - i), n))
            for src, dst in ((ma, da), (mb, db)):
                for k, v in metrics_from_sums(src[:, idx, :].sum(axis=2)).items():
                    dst[k].append(np.nanmean(v, axis=0))
        del d
        draws_a = {k: np.concatenate(v) for k, v in da.items()}
        draws_b = {k: np.concatenate(v) for k, v in db.items()}
    else:
        draws_a, draws_b = _draws(ma, n_boot, rng), _draws(mb, n_boot, rng)
    pa, pb = point_estimate(a), point_estimate(b)
    q = (1 - ci) / 2
    out = {}
    for k in METRICS:
        diff = draws_a[k] - draws_b[k]
        out[k] = {
            "diff": pa[k] - pb[k],
            "lo": float(np.nanquantile(diff, q)),
            "hi": float(np.nanquantile(diff, 1 - q)),
            "p": _p_two_sided(diff),
        }
    return out


def holm(p: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p_arr = np.asarray(p, dtype=float)
    order = np.argsort(p_arr)
    adj = np.empty_like(p_arr)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p_arr) - rank) * p_arr[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


Compare = Callable[..., dict[str, dict[str, float]]]
