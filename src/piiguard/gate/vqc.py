"""Variational quantum circuit gate (PennyLane): AngleEmbedding + StronglyEntanglingLayers.

Circuit: ``RX(x_i)`` on each qubit, ``L`` StronglyEntanglingLayers, readout ``<Z_0>``.
Probability = ``sigmoid(scale * <Z_0> + bias)``; ``scale`` and ``bias`` are two extra trainable
classical numbers so the output can be confident (``<Z_0>`` alone only spans [-1, 1]).

Training uses ``default.qubit`` with backprop on broadcast batches. On this project's hardware this
was ~10x faster than ``lightning.qubit`` + adjoint, which handles a batch one sample at a time.
Evaluation settings: exact, finite shots, and depolarizing noise (``default.mixed``) applied
after every gate of a hand-built copy of the same circuit (checked equal at p = 0).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pennylane as qml
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="pennylane")


@dataclass(frozen=True)
class Setting:
    """How the trained circuit is executed at evaluation time."""

    name: str
    shots: int | None = None
    noise_p: float | None = None


def exact_setting() -> Setting:
    """Noise-free analytic simulation."""
    return Setting("exact")


def shots_setting(shots: int) -> Setting:
    """Finite-shot sampling."""
    return Setting(f"shots{shots}", shots=shots)


def noisy_setting(p: float) -> Setting:
    """Depolarizing noise with probability ``p`` after every gate (analytic, density matrix)."""
    return Setting(f"noisy_p{p}", noise_p=p)


def _manual_circuit(x: Any, w: Any, n: int, p: float | None) -> Any:
    """Same unitary as AngleEmbedding + StronglyEntanglingLayers, built gate by gate.

    ``p`` adds a depolarizing channel on every wire a gate touched.
    """

    def noise(wires: list[int]) -> None:
        if p:
            for q in wires:
                qml.DepolarizingChannel(p, wires=q)

    for q in range(n):
        qml.RX(x[:, q], wires=q)
        noise([q])
    for layer in range(w.shape[0]):
        r = (layer % (n - 1)) + 1 if n > 1 else 0
        for i in range(n):
            qml.Rot(w[layer, i, 0], w[layer, i, 1], w[layer, i, 2], wires=i)
            noise([i])
        if n > 1:
            for i in range(n):
                qml.CNOT(wires=[i, (i + r) % n])
                noise([i, (i + r) % n])
    return qml.expval(qml.PauliZ(0))


class VQC:
    """8-qubit (configurable) variational classifier trained with Adam on binary cross-entropy."""

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        init_std: float = 0.5,
        seed: int = 0,
        reupload: bool = False,
    ) -> None:
        self.n, self.L, self.reupload = n_qubits, n_layers, reupload
        g = torch.Generator().manual_seed(seed)
        self.weights = (
            torch.randn(n_layers, n_qubits, 3, generator=g, dtype=torch.float64) * init_std
        ).requires_grad_()
        self.scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        self.bias = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        self.seed = seed
        self._dev = qml.device("default.qubit", wires=n_qubits)
        self._qnode = qml.QNode(self._circuit, self._dev, interface="torch", diff_method="backprop")
        self._shots_qnode = qml.QNode(self._circuit, self._dev, interface="torch", diff_method=None)
        self._noisy: dict[float, qml.QNode] = {}

    # ------------------------------------------------------------------ circuit
    def _circuit(self, x: Any, w: Any) -> Any:
        if self.reupload:  # data re-uploading: re-encode x before every trainable layer
            for layer in range(self.L):
                qml.AngleEmbedding(x, wires=range(self.n), rotation="X")
                qml.StronglyEntanglingLayers(w[layer : layer + 1], wires=range(self.n))
        else:
            qml.AngleEmbedding(x, wires=range(self.n), rotation="X")
            qml.StronglyEntanglingLayers(w, wires=range(self.n))
        return qml.expval(qml.PauliZ(0))

    @property
    def n_params(self) -> int:
        """Trainable parameters: circuit weights plus the scale and bias."""
        return int(self.weights.numel() + 2)

    def _noisy_qnode(self, p: float) -> qml.QNode:
        if self.reupload:
            raise NotImplementedError("noise model is only implemented without re-uploading")
        if p not in self._noisy:
            dev = qml.device("default.mixed", wires=self.n)
            self._noisy[p] = qml.QNode(
                lambda x, w: _manual_circuit(x, w, self.n, p), dev, interface="torch"
            )
        return self._noisy[p]

    def expval(self, X: torch.Tensor, setting: Setting | None = None) -> torch.Tensor:
        """``<Z_0>`` for each row of ``X`` under ``setting`` (default exact)."""
        setting = setting or exact_setting()
        if setting.noise_p is not None:
            out = []
            for i in range(0, len(X), 64):
                out.append(self._noisy_qnode(setting.noise_p)(X[i : i + 64], self.weights))
            return torch.cat(out)
        if setting.shots is not None:
            return qml.set_shots(self._shots_qnode, shots=setting.shots)(X, self.weights)
        return self._qnode(X, self.weights)

    def logits(self, X: torch.Tensor, setting: Setting | None = None) -> torch.Tensor:
        """Pre-sigmoid score."""
        return self.scale * self.expval(X, setting) + self.bias

    # ------------------------------------------------------------------ training / inference
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 0.05,
    ) -> list[float]:
        """Train with Adam on BCE; returns the mean loss of each epoch."""
        Xt = torch.as_tensor(X, dtype=torch.float64)
        yt = torch.as_tensor(y, dtype=torch.float64)
        opt = torch.optim.Adam([self.weights, self.scale, self.bias], lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(self.seed + 1)
        history = []
        for _ in range(epochs):
            perm = torch.randperm(len(Xt), generator=g)
            total = 0.0
            for i in range(0, len(Xt), batch_size):
                idx = perm[i : i + batch_size]
                opt.zero_grad()
                loss = loss_fn(self.logits(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += float(loss) * len(idx)
            history.append(total / len(Xt))
        return history

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, setting: Setting | None = None) -> np.ndarray:
        """P(contains PII) under ``setting``."""
        Xt = torch.as_tensor(X, dtype=torch.float64)
        return torch.sigmoid(self.logits(Xt, setting)).numpy()


# ---------------------------------------------------------------------- barren plateau check
def gradient_variance(
    n_qubits: int, n_layers: int, n_samples: int, X_pool: np.ndarray, seed: int
) -> dict[str, float]:
    """Variance over random initialisations of ``d<Z_0>/d theta``, for ``n_qubits`` qubits.

    Weights are drawn uniformly from [0, 2 pi); each sample pairs them with a random data row
    (features cycled to fill the qubits). Returns the variance of the first weight and the mean
    variance over all weights.
    """
    rng = np.random.default_rng(seed)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x: Any, w: Any) -> Any:
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="X")
        qml.StronglyEntanglingLayers(w, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    cols = [i % X_pool.shape[1] for i in range(n_qubits)]
    grads = []
    for _ in range(n_samples):
        w = torch.as_tensor(
            rng.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3)), dtype=torch.float64
        ).requires_grad_()
        x = torch.as_tensor(X_pool[rng.integers(len(X_pool))][cols], dtype=torch.float64)
        circuit(x, w).backward()
        grads.append(w.grad.numpy().copy())
    g = np.stack(grads)
    var = g.var(axis=0)
    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_samples": n_samples,
        "var_first_param": float(var.flat[0]),
        "var_mean_all_params": float(var.mean()),
    }
