"""Binary-classification metrics for the PII gate."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def recall_at_precision(y_true: np.ndarray, scores: np.ndarray, precision: float = 0.95) -> float:
    """Highest recall reachable while precision stays >= ``precision`` (0.0 if never reached)."""
    p, r, _ = precision_recall_curve(y_true, scores)
    ok = p[:-1] >= precision
    return float(r[:-1][ok].max()) if ok.any() else 0.0


def gate_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float | None = 0.5
) -> dict[str, float]:
    """ROC-AUC, PR-AUC, recall at 95% precision, F1 and accuracy.

    ``scores`` are probabilities (threshold 0.5) or decision values (pass the matching
    ``threshold``, e.g. 0.0 for an SVM).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "recall_at_95p": recall_at_precision(y_true, scores, 0.95),
        "f1": float(f1_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def latency_ms(
    fn: Callable[[np.ndarray], object], X: np.ndarray, n_single: int = 50, warmup: int = 2
) -> dict[str, float]:
    """Time ``fn`` on single rows (mean ms per sentence) and on the whole batch (ms per row)."""
    n_single = min(n_single, len(X))
    for i in range(min(warmup, len(X))):
        fn(X[i : i + 1])
    t = time.perf_counter()
    for i in range(n_single):
        fn(X[i : i + 1])
    single = (time.perf_counter() - t) / n_single * 1000
    t = time.perf_counter()
    fn(X)
    batch = (time.perf_counter() - t) / len(X) * 1000
    return {"latency_ms_single": single, "latency_ms_batch_per_sentence": batch}
