"""Classical gate baselines on the same 8 angle features: logistic regression and a tiny MLP."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from piiguard.gate.gate_metrics import gate_metrics, latency_ms
from piiguard.utils import timed


def make_model(kind: str, seed: int, hidden: int = 7, max_iter: int = 300) -> Any:
    """``logreg`` or ``mlp`` (one hidden layer; hidden=7 gives 71 params, close to the VQC's 74)."""
    if kind == "logreg":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if kind == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(hidden,), max_iter=max_iter, random_state=seed, early_stopping=False
        )
    raise ValueError(kind)


def count_params(model: Any) -> int:
    """Number of trainable parameters of a fitted sklearn model."""
    if isinstance(model, LogisticRegression):
        return int(model.coef_.size + model.intercept_.size)
    return int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_))


def fit_eval(
    kind: str,
    seed: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_ev: np.ndarray,
    y_ev: np.ndarray,
    X_full: np.ndarray,
    y_full: np.ndarray,
    **kw: Any,
) -> dict[str, Any]:
    """Train on ``(X_tr, y_tr)``; score on the common eval subset and the full test set."""
    model = make_model(kind, seed, **kw)
    with timed() as t:
        model.fit(X_tr, y_tr)
    proba = lambda X: model.predict_proba(X)[:, 1]  # noqa: E731
    return {
        **gate_metrics(y_ev, proba(X_ev)),
        "params": count_params(model),
        "train_s": t.elapsed,
        **latency_ms(proba, X_ev),
        "test_full": gate_metrics(y_full, proba(X_full)),
    }
