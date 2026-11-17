"""Phase 3: the PII gate, classical vs quantum, on identical 8-dimensional features.

Usage: python scripts/run_gate.py --config configs/smoke.yaml --stage all
Stages: features, classical, vqc, kernel, finetune, barren, summary, all
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from piiguard.gate import classical
from piiguard.gate.features import balanced_subset, build_features
from piiguard.gate.finetune import finetune_and_eval
from piiguard.gate.gate_metrics import gate_metrics, latency_ms
from piiguard.gate.qkernel import QuantumKernelSVM
from piiguard.gate.summary import summarize
from piiguard.gate.vqc import VQC, exact_setting, gradient_variance, noisy_setting, shots_setting
from piiguard.utils import ensure_dir, get_logger, load_config, set_seed, timed, write_run_record

log = get_logger("run_gate")


class Ctx:
    """Config, cached features and the fixed evaluation subsets shared by every model."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg, self.g = cfg, cfg["gate"]
        self.out = ensure_dir(Path(cfg["paths"]["results_dir"]) / "gate")
        self.f = build_features(cfg)
        self.ev = balanced_subset(self.f["y_test"], self.g["eval_n"], seed=0)
        self.X_ev, self.y_ev = self.f["X_test"][self.ev], self.f["y_test"][self.ev]
        sub = balanced_subset(self.y_ev, self.g["noisy_eval_n"], seed=1)
        self.noisy_pos = sub  # positions inside the eval subset

    def train_idx(self, seed: int, n: int | None = None) -> np.ndarray:
        """Balanced training subset of the train pool (all of it when ``n`` is None)."""
        y = self.f["y_train"]
        return np.arange(len(y)) if n is None else balanced_subset(y, n, seed)

    def save(
        self,
        model: str,
        setting: str,
        size: str,
        seed: int,
        metrics: dict,
        timings: dict,
        **extra: Any,
    ) -> None:
        """Write one run record."""
        m = {
            "kind": "gate_model",
            "model": model,
            "setting": setting,
            "size": size,
            **metrics,
            **extra,
        }
        write_run_record(
            self.out,
            f"{model}__{setting}__{size}",
            config=self.cfg,
            seed=seed,
            metrics=m,
            timings=timings,
        )
        log.info(
            "%s %s %s seed=%d auc=%.3f", model, setting, size, seed, m.get("roc_auc", float("nan"))
        )


def stage_features(c: Ctx) -> None:
    """Embed, reduce and cache the features; record PCA variance and front-end latency."""
    meta = {
        k: c.f[k]
        for k in (
            "sizes",
            "model",
            "n_features",
            "pca_explained_variance",
            "pca_explained_variance_total",
            "embed_total_s",
            "front_end_ms_per_sentence_incl_model_load",
            "device",
        )
    }
    write_run_record(
        c.out, "features", config=c.cfg, seed=c.cfg["seed"], metrics={"kind": "features", **meta}
    )
    log.info(
        "features: PCA keeps %.1f%% of embedding variance",
        100 * meta["pca_explained_variance_total"],
    )


def stage_classical(c: Ctx) -> None:
    """Logistic regression and MLP at the gate size, and at the full train-pool size."""
    g, f = c.g, c.f
    sizes = [(f"n{g['n_train']}", g["n_train"])] + (
        [("full", None)] if g["classical_full_size"] else []
    )
    for seed in g["seeds_classical"]:
        for label, n in sizes:
            idx = c.train_idx(seed, n)
            for kind in ("logreg", "mlp"):
                with timed() as t:
                    r = classical.fit_eval(
                        kind,
                        seed,
                        f["X_train"][idx],
                        f["y_train"][idx],
                        c.X_ev,
                        c.y_ev,
                        f["X_test"],
                        f["y_test"],
                        **(
                            {"hidden": g["mlp"]["hidden"], "max_iter": g["mlp"]["max_iter"]}
                            if kind == "mlp"
                            else {}
                        ),
                    )
                c.save(
                    kind,
                    "classical",
                    label,
                    seed,
                    {**r, "n_train": len(idx), "n_eval": len(c.ev)},
                    {"total_s": t.elapsed},
                )


def stage_vqc(c: Ctx) -> None:
    """Train the VQC once per seed, then evaluate it under every execution setting."""
    g, f, v = c.g, c.f, c.g["vqc"]
    settings = [exact_setting(), shots_setting(g["shots"]), *map(noisy_setting, g["noise_probs"])]
    for seed in g["seeds_classical"]:
        set_seed(seed)
        idx = c.train_idx(seed, g["n_train"])
        model = VQC(v["n_qubits"], v["n_layers"], v["init_std"], seed)
        with timed() as tt:
            hist = model.fit(
                f["X_train"][idx], f["y_train"][idx], v["epochs"], v["batch_size"], v["lr"]
            )
        exact_on_noisy = gate_metrics(c.y_ev[c.noisy_pos], model.predict_proba(c.X_ev[c.noisy_pos]))
        for s in settings:
            if s.noise_p is not None:
                X, y, n_single = c.X_ev[c.noisy_pos], c.y_ev[c.noisy_pos], 10
            else:
                X, y, n_single = c.X_ev, c.y_ev, 50
            with timed() as te:
                p = model.predict_proba(X, s)
            lat = latency_ms(partial(model.predict_proba, setting=s), X, n_single=n_single)
            extra = {"exact_same_subset": exact_on_noisy} if s.noise_p is not None else {}
            c.save(
                "vqc",
                s.name,
                f"n{g['n_train']}",
                seed,
                {
                    **gate_metrics(y, p),
                    "params": model.n_params,
                    "train_s": tt.elapsed,
                    **lat,
                    "n_train": len(idx),
                    "n_eval": len(y),
                    "backend": "default.qubit",
                    "n_qubits": v["n_qubits"],
                    "n_layers": v["n_layers"],
                    "loss_first": hist[0],
                    "loss_last": hist[-1],
                    "loss_history": hist,
                    **extra,
                },
                {"eval_s": te.elapsed},
            )


def stage_kernel(c: Ctx) -> None:
    """Fidelity-kernel SVM (exact and sampled-shots), for each configured feature map."""
    g, f, k = c.g, c.f, c.g["kernel"]
    for fmap in k["feature_maps"]:
        for seed in g["seeds_kernel"]:
            idx = c.train_idx(seed, k["train_cap"])
            vidx = balanced_subset(f["y_val"], k["val_n"], seed)
            log.info(
                "kernel[%s] seed %d: train cap %d -> using %d", fmap, seed, k["train_cap"], len(idx)
            )
            for setting, shots in (("exact", None), (f"shots{g['shots']}", g["shots"])):
                svm = QuantumKernelSVM(fmap, tuple(k["c_grid"]))
                with timed() as tt:
                    svm.fit(
                        f["X_train"][idx],
                        f["y_train"][idx],
                        f["X_val"][vidx],
                        f["y_val"][vidx],
                        shots=shots,
                        seed=seed,
                    )
                score = partial(svm.decision_function, shots=shots, seed=seed)
                m = gate_metrics(c.y_ev, score(c.X_ev), threshold=0.0)
                full = gate_metrics(f["y_test"], score(f["X_test"]), threshold=0.0)
                lat = latency_ms(score, c.X_ev, n_single=20)
                c.save(
                    f"qkernel_{fmap}",
                    setting,
                    f"n{k['train_cap']}",
                    seed,
                    {
                        **m,
                        "params": 0,
                        "train_s": tt.elapsed,
                        **lat,
                        "test_full": full,
                        "n_train": len(idx),
                        "kernel_train_cap": k["train_cap"],
                        "n_eval": len(c.ev),
                        "n_support_vectors": svm.n_support,
                        "best_C": svm.best_c,
                        "hardware_circuits_per_inference": len(idx),
                        "feature_map": fmap,
                    },
                    {"fit_s": tt.elapsed},
                )


def stage_finetune(c: Ctx) -> None:
    """MiniLM fine-tuned on raw sentences: the upper bound (no 8-feature bottleneck)."""
    g, f, ft = c.g, c.f, c.g["finetune"]
    tr, te = f["frames"]["train"], f["frames"]["test"]
    ev_sets = {
        "test_eval": (te.text.iloc[c.ev].tolist(), c.y_ev),
        "test_full": (te.text.tolist(), f["y_test"]),
    }
    sizes = [(f"n{g['n_train']}", g["n_train"])] + ([("full", None)] if ft["full_size"] else [])
    for seed in g["seeds_finetune"]:
        for label, n in sizes:
            idx = c.train_idx(seed, n)
            r = finetune_and_eval(
                g["embed_model"],
                tr.text.iloc[idx].tolist(),
                f["y_train"][idx],
                ev_sets,
                ft["epochs"],
                ft["batch_size"],
                ft["lr"],
                seed,
            )
            c.save(
                "minilm_finetuned",
                "raw_text",
                label,
                seed,
                {
                    **r["eval"]["test_eval"],
                    "params": r["params"],
                    "train_s": r["train_s"],
                    "latency_ms_single": r["latency_ms_single"],
                    "latency_ms_batch_per_sentence": r["latency_ms_batch_per_sentence"],
                    "test_full": r["eval"]["test_full"],
                    "n_train": len(idx),
                    "n_eval": len(c.ev),
                    "device": r["device"],
                },
                {"train_s": r["train_s"]},
            )


def stage_barren(c: Ctx) -> None:
    """Gradient variance of <Z_0> vs qubit count (barren-plateau check)."""
    b = c.g["barren_plateau"]
    for q in b["qubits"]:
        with timed() as t:
            r = gradient_variance(q, b["n_layers"], b["n_samples"], c.f["X_train"], c.cfg["seed"])
        write_run_record(
            c.out,
            f"barren_q{q}",
            config=c.cfg,
            seed=c.cfg["seed"],
            metrics={"kind": "barren", **r},
            timings={"total_s": t.elapsed},
        )
        log.info(
            "barren q=%d var(first)=%.3e var(mean)=%.3e",
            q,
            r["var_first_param"],
            r["var_mean_all_params"],
        )


def stage_summary(c: Ctx) -> None:
    """Write ``summary.csv`` and ``summary_barren.csv`` from the records on disk."""
    perf, barren = summarize(c.out)
    perf.to_csv(c.out / "summary.csv", index=False)
    barren.to_csv(c.out / "summary_barren.csv", index=False)
    log.info("wrote %s (%d rows) and summary_barren.csv", c.out / "summary.csv", len(perf))


STAGES = {
    "features": stage_features,
    "classical": stage_classical,
    "vqc": stage_vqc,
    "kernel": stage_kernel,
    "finetune": stage_finetune,
    "barren": stage_barren,
    "summary": stage_summary,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all", choices=[*STAGES, "all"])
    args = ap.parse_args()
    c = Ctx(load_config(args.config))
    for name in STAGES if args.stage == "all" else [args.stage]:
        with timed() as t:
            STAGES[name](c)
        log.info("stage %s done in %.1fs", name, t.elapsed)


if __name__ == "__main__":
    main()
