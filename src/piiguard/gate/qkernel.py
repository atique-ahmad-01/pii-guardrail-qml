"""Quantum fidelity-kernel SVM on the 8 angle features.

Kernel ``k(x, y) = |<phi(x)|phi(y)>|^2``. States come from PennyLane (``default.qubit``), then the
Gram matrix is a matrix product of the state vectors, which is cheap in simulation. On hardware
each entry would be one compute-uncompute circuit, so ``n_train`` circuits per test sentence.

Two feature maps:
* ``angle``: the same ``AngleEmbedding`` (RX) as the VQC. This is a *product* state, so the kernel
  equals ``prod_i cos^2((x_i - y_i)/2)`` exactly (checked in the tests). No entanglement is used,
  so it is classically trivial. It is the brief's specified kernel.
* ``iqp``: PennyLane's ``IQPEmbedding`` (entangling ZZ terms), included as an extra scenario.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pennylane as qml
from sklearn.svm import SVC

from piiguard.gate.gate_metrics import gate_metrics

warnings.filterwarnings("ignore", category=UserWarning, module="pennylane")


def _embed(x: Any, n: int, feature_map: str) -> None:
    if feature_map == "angle":
        qml.AngleEmbedding(x, wires=range(n), rotation="X")
    elif feature_map == "iqp":
        qml.IQPEmbedding(x, wires=range(n), n_repeats=2)
    else:
        raise ValueError(feature_map)


def feature_states(X: np.ndarray, feature_map: str = "angle", batch: int = 256) -> np.ndarray:
    """State vectors ``|phi(x)>`` for each row of ``X`` (shape ``(N, 2**n)``, complex128)."""
    n = X.shape[1]
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def state(x: Any) -> Any:
        _embed(x, n, feature_map)
        return qml.state()

    out = []
    for i in range(0, len(X), batch):
        chunk = np.asarray(X[i : i + batch], dtype=np.float64)
        try:
            out.append(np.asarray(state(chunk)))
        except Exception:  # embedding without broadcasting support: loop
            out.append(np.stack([np.asarray(state(r)) for r in chunk]))
    return np.concatenate(out)


def fidelity_kernel(SA: np.ndarray, SB: np.ndarray) -> np.ndarray:
    """``K[i, j] = |<a_i|b_j>|^2`` from state matrices."""
    return np.abs(SA.conj() @ SB.T) ** 2


def closed_form_angle_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``prod_i cos^2((a_i - b_j,i) / 2)``: what the angle-embedding kernel equals analytically."""
    return np.prod(np.cos((A[:, None, :] - B[None, :, :]) / 2) ** 2, axis=-1)


def sample_kernel(
    K: np.ndarray, shots: int, rng: np.random.Generator, symmetric: bool
) -> np.ndarray:
    """Estimate each entry from ``shots`` measurements of the compute-uncompute circuit.

    The probability of reading all zeros equals ``K``, so the estimate is
    Binomial(shots, K) / shots.
    A symmetric Gram matrix is sampled once per pair and mirrored, with ones on the diagonal.
    """
    est = rng.binomial(shots, np.clip(K, 0.0, 1.0)) / shots
    if symmetric:
        up = np.triu(est, 1)
        est = up + up.T + np.eye(len(K))
    return est


class QuantumKernelSVM:
    """SVC on a precomputed fidelity kernel; ``C`` chosen on a validation subset."""

    def __init__(self, feature_map: str = "angle", c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)):
        self.feature_map, self.c_grid = feature_map, c_grid
        self.svc: SVC | None = None
        self.S_train: np.ndarray | None = None
        self.best_c: float | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        shots: int | None = None,
        seed: int = 0,
    ) -> QuantumKernelSVM:
        """Fit on ``X``; pick ``C`` by validation ROC-AUC. With ``shots`` the Gram is sampled."""
        rng = np.random.default_rng(seed)
        self.S_train = feature_states(X, self.feature_map)
        K = fidelity_kernel(self.S_train, self.S_train)
        Kv = fidelity_kernel(feature_states(X_val, self.feature_map), self.S_train)
        if shots:
            K = sample_kernel(K, shots, rng, symmetric=True)
            Kv = sample_kernel(Kv, shots, rng, symmetric=False)
        best = (-1.0, self.c_grid[0])
        for c in self.c_grid:
            auc = gate_metrics(
                y_val, SVC(kernel="precomputed", C=c).fit(K, y).decision_function(Kv)
            )["roc_auc"]
            best = max(best, (auc, c))
        self.best_c = best[1]
        self.svc = SVC(kernel="precomputed", C=self.best_c).fit(K, y)
        return self

    def decision_function(
        self, X: np.ndarray, shots: int | None = None, seed: int = 0
    ) -> np.ndarray:
        """Signed distance to the SVM boundary (> 0 means PII)."""
        K = fidelity_kernel(feature_states(X, self.feature_map), self.S_train)
        if shots:
            K = sample_kernel(K, shots, np.random.default_rng(seed + 7), symmetric=False)
        return self.svc.decision_function(K)

    @property
    def n_support(self) -> int:
        """Number of support vectors."""
        return int(self.svc.n_support_.sum())
