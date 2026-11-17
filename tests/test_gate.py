import numpy as np
import pennylane as qml
import pytest
import torch

from piiguard.gate import classical
from piiguard.gate.features import AngleScaler, balanced_subset
from piiguard.gate.gate_metrics import gate_metrics, recall_at_precision
from piiguard.gate.qkernel import (
    QuantumKernelSVM,
    closed_form_angle_kernel,
    feature_states,
    fidelity_kernel,
    sample_kernel,
)
from piiguard.gate.vqc import (
    VQC,
    _manual_circuit,
    exact_setting,
    gradient_variance,
    noisy_setting,
    shots_setting,
)


def toy_data(n=120, d=4, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    X = rng.uniform(0.2, 1.0, (n, d)) + y[:, None] * rng.uniform(0.5, 1.2, (n, d))
    return np.clip(X, 0, np.pi), y


# ------------------------------------------------------------------ metrics
def test_recall_at_precision_hand_example():
    y = np.array([1, 1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1])
    # precision 1.0 up to score 0.8 (recall 2/3); at 0.7 precision drops to 2/3 < 0.95
    assert recall_at_precision(y, s, 0.95) == pytest.approx(2 / 3)
    assert recall_at_precision(np.array([0, 1]), np.array([0.9, 0.1]), 0.95) == 0.0


def test_gate_metrics_perfect_and_threshold():
    y = np.array([0, 0, 1, 1])
    m = gate_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == m["pr_auc"] == m["f1"] == m["recall_at_95p"] == 1.0
    m0 = gate_metrics(y, np.array([-2.0, -1.0, 1.0, 3.0]), threshold=0.0)
    assert m0["accuracy"] == 1.0


# ------------------------------------------------------------------ features
def test_angle_scaler_range_fit_on_train_only():
    rng = np.random.default_rng(0)
    tr, te = rng.normal(size=(200, 16)), rng.normal(size=(50, 16)) * 3
    sc = AngleScaler(8).fit(tr)
    a, b = sc.transform(tr), sc.transform(te)
    assert a.shape == (200, 8) and a.min() >= 0 and a.max() <= np.pi + 1e-12
    assert b.min() >= 0 and b.max() <= np.pi + 1e-12  # clipped, never outside [0, pi]


def test_balanced_subset_equal_classes_deterministic():
    y = np.array([0] * 50 + [1] * 30)
    i = balanced_subset(y, 40, 1)
    assert (y[i] == 1).sum() == (y[i] == 0).sum() == 20
    assert np.array_equal(i, balanced_subset(y, 40, 1))
    assert len(balanced_subset(y, 200, 1)) == 60  # capped by the smaller class


# ------------------------------------------------------------------ classical
def test_classical_param_counts_match_design():
    X, y = toy_data(d=8)
    assert classical.count_params(classical.make_model("logreg", 0).fit(X, y)) == 9
    assert classical.count_params(classical.make_model("mlp", 0, hidden=7).fit(X, y)) == 71
    r = classical.fit_eval("logreg", 0, X, y, X, y, X, y)
    assert r["roc_auc"] > 0.9 and r["params"] == 9


# ------------------------------------------------------------------ VQC
def test_vqc_params_and_probabilities():
    m = VQC(n_qubits=4, n_layers=3, seed=0)
    assert m.n_params == 3 * 4 * 3 + 2
    X, _ = toy_data(10, 4)
    p = m.predict_proba(X)
    assert p.shape == (10,) and ((p > 0) & (p < 1)).all()
    assert np.allclose(p, VQC(4, 3, seed=0).predict_proba(X))  # same seed -> same model


def test_vqc_training_reduces_loss():
    X, y = toy_data(60, 4)
    m = VQC(n_qubits=4, n_layers=2, seed=1)
    hist = m.fit(X, y, epochs=8, batch_size=20, lr=0.1)
    assert hist[-1] < hist[0]


def test_manual_noisy_circuit_equals_strongly_entangling_layers_at_p0():
    n = 4
    x = torch.rand(3, n, dtype=torch.float64) * np.pi
    w = torch.randn(3, n, 3, dtype=torch.float64)

    @qml.qnode(qml.device("default.qubit", wires=n), interface="torch")
    def exact(x, w):
        qml.AngleEmbedding(x, wires=range(n), rotation="X")
        qml.StronglyEntanglingLayers(w, wires=range(n))
        return qml.expval(qml.PauliZ(0))

    manual0 = qml.QNode(
        lambda x, w: _manual_circuit(x, w, n, None),
        qml.device("default.qubit", wires=n),
        interface="torch",
    )
    mixed0 = qml.QNode(
        lambda x, w: _manual_circuit(x, w, n, 0.0),
        qml.device("default.mixed", wires=n),
        interface="torch",
    )
    assert torch.allclose(exact(x, w), manual0(x, w), atol=1e-8)
    assert torch.allclose(exact(x, w), mixed0(x, w), atol=1e-8)


def test_noise_shrinks_expectation_and_shots_are_unbiased():
    X, y = toy_data(40, 4)
    m = VQC(4, 3, seed=2)
    m.fit(X, y, epochs=3, batch_size=20, lr=0.1)
    Xt = torch.as_tensor(X[:16], dtype=torch.float64)
    with torch.no_grad():
        e = m.expval(Xt, exact_setting())
        n_hi = m.expval(Xt, noisy_setting(0.2))
        sh = torch.stack([m.expval(Xt, shots_setting(2000)) for _ in range(8)]).mean(0)
    assert n_hi.abs().mean() < 0.5 * e.abs().mean()  # strong noise pulls <Z> toward 0
    assert float((sh - e).abs().max()) < 0.06  # mean of 8 x 2000 shots is close to exact


def test_gradient_variance_runs_and_decreases_with_qubits():
    pool = np.random.default_rng(0).uniform(0, np.pi, (50, 8))
    v2 = gradient_variance(2, 3, 60, pool, 0)["var_mean_all_params"]
    v6 = gradient_variance(6, 3, 60, pool, 0)["var_mean_all_params"]
    assert v2 > 0 and v6 > 0 and v6 < v2


# ------------------------------------------------------------------ quantum kernel
def test_angle_kernel_equals_closed_form_and_is_valid():
    X, _ = toy_data(12, 5)
    S = feature_states(X, "angle")
    K = fidelity_kernel(S, S)
    assert np.allclose(K, closed_form_angle_kernel(X, X), atol=1e-8)
    assert np.allclose(np.diag(K), 1) and np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-8  # positive semi-definite


def test_iqp_kernel_differs_from_product_kernel():
    X, _ = toy_data(8, 4)
    K = fidelity_kernel(feature_states(X, "iqp"), feature_states(X, "iqp"))
    assert np.allclose(np.diag(K), 1)
    assert not np.allclose(K, closed_form_angle_kernel(X, X), atol=1e-3)


def test_sample_kernel_unbiased_and_symmetric():
    K = np.full((30, 30), 0.4)
    np.fill_diagonal(K, 1.0)
    est = sample_kernel(K, 1024, np.random.default_rng(0), symmetric=True)
    assert np.allclose(est, est.T) and np.allclose(np.diag(est), 1)
    off = est[~np.eye(30, dtype=bool)]
    assert abs(off.mean() - 0.4) < 0.01


def test_kernel_svm_learns_toy_task():
    X, y = toy_data(100, 4)
    Xv, yv = toy_data(40, 4, seed=5)
    svm = QuantumKernelSVM("angle").fit(X, y, Xv, yv)
    m = gate_metrics(yv, svm.decision_function(Xv), threshold=0.0)
    assert m["roc_auc"] > 0.85 and svm.n_support > 0
