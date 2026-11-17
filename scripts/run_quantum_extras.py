"""Extra quantum scenarios on the gate (all exact simulation, same features as Phase 3).

  low_data       learning curves: ROC-AUC vs number of training sentences
  ablation       VQC depth (layers) and data re-uploading
  feature_sweep  number of PCA features / qubits; classical models also get more features

Records go to ``<results_dir>/gate_extras/``; ``summary.csv`` aggregates them over seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from piiguard.gate import classical
from piiguard.gate.features import AngleScaler, balanced_subset, build_features, embed
from piiguard.gate.gate_metrics import gate_metrics
from piiguard.gate.qkernel import QuantumKernelSVM
from piiguard.gate.vqc import VQC
from piiguard.utils import (
    ensure_dir,
    get_device,
    get_logger,
    load_config,
    set_seed,
    timed,
    write_run_record,
)

log = get_logger("quantum_extras")


class Run:
    """Shared evaluation subset and record writer."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg, self.g = cfg, cfg["gate"]
        self.out = ensure_dir(Path(cfg["paths"]["results_dir"]) / "gate_extras")
        self.f = build_features(cfg)
        self.ev = balanced_subset(self.f["y_test"], self.g["eval_n"], seed=0)

    def save(self, scenario: str, variant: str, x: float, seed: int, m: dict, t: float) -> None:
        rec = {"kind": "gate_extra", "scenario": scenario, "variant": variant, "x": x, **m}
        write_run_record(
            self.out,
            f"extra__{scenario}__{variant}__x{x:g}",
            config=self.cfg,
            seed=seed,
            metrics=rec,
            timings={"train_s": t},
        )
        log.info(
            "%s %-14s x=%-6g seed=%d auc=%.3f (%.1fs)", scenario, variant, x, seed, m["roc_auc"], t
        )


def fit_vqc(X: np.ndarray, y: np.ndarray, g: dict[str, Any], seed: int, **kw: Any) -> VQC:
    v = g["vqc"]
    set_seed(seed)
    n_q = kw.pop("n_qubits", X.shape[1])
    model = VQC(n_q, kw.pop("n_layers", v["n_layers"]), v["init_std"], seed, **kw)
    model.fit(X, y, v["epochs"], v["batch_size"], v["lr"])
    return model


def low_data(r: Run, c: dict[str, Any]) -> None:
    """Learning curves for logreg, MLP, VQC and the angle-kernel SVM."""
    f, g = r.f, r.g
    X_ev, y_ev = f["X_test"][r.ev], f["y_test"][r.ev]
    for n in c["sizes"]:
        for seed in c["seeds"]:
            idx = balanced_subset(f["y_train"], n, seed)
            X, y = f["X_train"][idx], f["y_train"][idx]
            for kind in ("logreg", "mlp"):
                kw = {"hidden": g["mlp"]["hidden"], "max_iter": g["mlp"]["max_iter"]}
                with timed() as t:
                    m = classical.make_model(kind, seed, **(kw if kind == "mlp" else {})).fit(X, y)
                r.save(
                    "low_data",
                    kind,
                    n,
                    seed,
                    gate_metrics(y_ev, m.predict_proba(X_ev)[:, 1]),
                    t.elapsed,
                )
            with timed() as t:
                v = fit_vqc(X, y, g, seed)
            r.save("low_data", "vqc", n, seed, gate_metrics(y_ev, v.predict_proba(X_ev)), t.elapsed)
            if n <= c["kernel_max"]:
                vidx = balanced_subset(f["y_val"], g["kernel"]["val_n"], seed)
                svm = QuantumKernelSVM("angle", tuple(g["kernel"]["c_grid"]))
                with timed() as t:
                    svm.fit(X, y, f["X_val"][vidx], f["y_val"][vidx], shots=None, seed=seed)
                r.save(
                    "low_data",
                    "qkernel_angle",
                    n,
                    seed,
                    gate_metrics(y_ev, svm.decision_function(X_ev), threshold=0.0),
                    t.elapsed,
                )


def ablation(r: Run, c: dict[str, Any]) -> None:
    """VQC depth, and data re-uploading at the default depth."""
    f, g = r.f, r.g
    X_ev, y_ev = f["X_test"][r.ev], f["y_test"][r.ev]
    for seed in c["seeds"]:
        idx = balanced_subset(f["y_train"], g["n_train"], seed)
        X, y = f["X_train"][idx], f["y_train"][idx]
        variants = [(f"layers{L}", {"n_layers": L}) for L in c["layers"]]
        variants.append((f"reupload_L{g['vqc']['n_layers']}", {"reupload": True}))
        for name, kw in variants:
            with timed() as t:
                v = fit_vqc(X, y, g, seed, **kw)
            r.save(
                "ablation",
                name,
                kw.get("n_layers", g["vqc"]["n_layers"]),
                seed,
                {**gate_metrics(y_ev, v.predict_proba(X_ev)), "params": int(v.n_params)},
                t.elapsed,
            )


def feature_sweep(r: Run, c: dict[str, Any]) -> None:
    """Re-fit PCA with k components; VQC uses k qubits, classical models may use more features."""
    f, g = r.f, r.g
    frames = f["frames"]
    pool = balanced_subset(f["y_train"], c["pool"], seed=0)
    dev = get_device()
    emb_pool = embed(
        frames["train"].text.to_numpy()[pool].tolist(), g["embed_model"], dev, g["embed_batch_size"]
    )
    emb_ev = embed(
        frames["test"].text.to_numpy()[r.ev].tolist(), g["embed_model"], dev, g["embed_batch_size"]
    )
    y_pool, y_ev = f["y_train"][pool], f["y_test"][r.ev]
    ks = sorted(set(c["classical_k"]) | set(c["quantum_k"]))
    for k in ks:
        sc = AngleScaler(k, 0).fit(emb_pool)
        Xp, Xe = sc.transform(emb_pool), sc.transform(emb_ev)
        kept = float(sc.pca.explained_variance_ratio_.sum())
        for seed in c["seeds"]:
            idx = balanced_subset(y_pool, g["n_train"], seed)
            X, y = Xp[idx], y_pool[idx]
            extra = {"pca_variance_kept": kept}
            if k in c["classical_k"]:
                for kind in ("logreg", "mlp"):
                    kw = {"hidden": g["mlp"]["hidden"], "max_iter": g["mlp"]["max_iter"]}
                    with timed() as t:
                        m = classical.make_model(kind, seed, **(kw if kind == "mlp" else {}))
                        m.fit(X, y)
                    r.save(
                        "feature_sweep",
                        kind,
                        k,
                        seed,
                        {**gate_metrics(y_ev, m.predict_proba(Xe)[:, 1]), **extra},
                        t.elapsed,
                    )
            if k in c["quantum_k"]:
                with timed() as t:
                    v = fit_vqc(X, y, g, seed, n_qubits=k)
                r.save(
                    "feature_sweep",
                    "vqc",
                    k,
                    seed,
                    {**gate_metrics(y_ev, v.predict_proba(Xe)), **extra},
                    t.elapsed,
                )


def summarize(out: Path) -> pd.DataFrame:
    """Mean and std over seeds for each (scenario, variant, x)."""
    rows = []
    for p in sorted(out.glob("extra__*.json")):
        rec = json.loads(p.read_text())
        m = rec["metrics"]
        rows.append(
            {
                "scenario": m["scenario"],
                "variant": m["variant"],
                "x": m["x"],
                "seed": rec["seed"],
                "roc_auc": m["roc_auc"],
                "recall_at_95p": m["recall_at_95p"],
                "train_s": rec["timings"]["train_s"],
                "params": m.get("params"),
                "pca_variance_kept": m.get("pca_variance_kept"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = df.groupby(["scenario", "variant", "x"])
    agg = g.agg(
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        recall_at_95p_mean=("recall_at_95p", "mean"),
        train_s_mean=("train_s", "mean"),
        params=("params", "first"),
        pca_variance_kept=("pca_variance_kept", "first"),
        n_seeds=("seed", "nunique"),
    )
    return agg.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--scenario",
        default="all",
        choices=["low_data", "ablation", "feature_sweep", "all", "summary"],
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    ex = cfg["gate_extras"]
    r = Run(cfg)
    todo = {"low_data": low_data, "ablation": ablation, "feature_sweep": feature_sweep}
    if args.scenario != "summary":
        for name, fn in todo.items():
            if args.scenario in (name, "all"):
                fn(r, ex[name])
    df = summarize(r.out)
    df.to_csv(r.out / "summary.csv", index=False)
    log.info("wrote %s (%d rows)", r.out / "summary.csv", len(df))


if __name__ == "__main__":
    main()
