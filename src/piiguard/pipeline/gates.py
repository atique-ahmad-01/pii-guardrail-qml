"""Fit the four gates on the same embeddings and score sentences with any of them."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from piiguard.gate import classical
from piiguard.gate.features import AngleScaler, balanced_subset, load_gate_frames
from piiguard.gate.finetune import MiniLMClassifier, train_minilm
from piiguard.gate.finetune import predict as minilm_predict
from piiguard.gate.gate_metrics import gate_metrics
from piiguard.gate.vqc import VQC
from piiguard.pipeline.assemble import pick_threshold
from piiguard.utils import ensure_dir, get_device, get_logger, set_seed

log = get_logger("pipeline.gates")

# Folder order in the split layout: classical, quantum, then the fine-tuned reference model.
GATE_ORDER = ("logreg", "mlp", "vqc", "minilm_ft")
SPLIT_LAYOUT = """
gate/
├── shared/            meta.json (thresholds), feature_scaler.joblib (PCA + scaling), encoder/
├── 001_logreg/        model.joblib
├── 002_mlp/           model.joblib
├── 003_vqc/           model.pt
└── 004_minilm_ft/     model.pt
"""


class Gates:
    """Logistic regression, MLP, VQC (exact simulation) and fine-tuned MiniLM as sentence gates."""

    def __init__(self, cfg: dict[str, Any], seed: int) -> None:
        self.cfg, self.seed = cfg, seed
        self.g, self.pc = cfg["gate"], cfg["pipeline"]
        self.device = get_device()
        self.models: dict[str, Any] = {}
        self.thresholds: dict[str, float] = {}
        self.info: dict[str, Any] = {}
        self._st: Any = None

    # ------------------------------------------------------------------ embeddings
    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalised MiniLM embeddings from a model that stays loaded between calls."""
        if self._st is None:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(self.g["embed_model"], device=self.device)
            self._st.max_seq_length = 128
        out = self._st.encode(
            texts,
            batch_size=self.g["embed_batch_size"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(out, dtype=np.float32)

    # ------------------------------------------------------------------ fitting
    def fit(self) -> Gates:
        """Train every gate and choose each gate's threshold on validation sentences."""
        frames = load_gate_frames(self.cfg["paths"]["data_dir"])
        tr, va = frames["train"], frames["val"]
        n_fit = self.pc["gate_train_n"]
        idx = balanced_subset(tr.label.to_numpy(), n_fit, self.seed)
        vidx = balanced_subset(va.label.to_numpy(), self.pc["val_n"], self.seed)
        tr_texts, y_tr = tr.text.to_numpy()[idx].tolist(), tr.label.to_numpy()[idx]
        va_texts, y_va = va.text.to_numpy()[vidx].tolist(), va.label.to_numpy()[vidx]
        log.info("fitting gates on %d train / %d val sentences", len(tr_texts), len(va_texts))

        emb_tr = self.embed(tr_texts)
        self.scaler = AngleScaler(self.g["n_features"], self.seed).fit(emb_tr)
        X_tr, X_va = self.scaler.transform(emb_tr), self.scaler.transform(self.embed(va_texts))

        for kind in ("logreg", "mlp"):
            t0 = time.perf_counter()
            kw = {"hidden": self.g["mlp"]["hidden"], "max_iter": self.g["mlp"]["max_iter"]}
            m = classical.make_model(kind, self.seed, **(kw if kind == "mlp" else {}))
            m.fit(X_tr, y_tr)
            self.models[kind] = m
            self.info[kind] = {"train_s": time.perf_counter() - t0}

        set_seed(self.seed)
        v = self.g["vqc"]
        t0 = time.perf_counter()
        vqc = VQC(v["n_qubits"], v["n_layers"], v["init_std"], self.seed)
        vqc.fit(X_tr, y_tr, v["epochs"], v["batch_size"], v["lr"])
        self.models["vqc"] = vqc
        self.info["vqc"] = {"train_s": time.perf_counter() - t0}

        ft = self.pc["minilm_ft"]
        fidx = balanced_subset(tr.label.to_numpy(), ft["train_n"], self.seed)
        model, tok, device, train_s = train_minilm(
            self.g["embed_model"],
            tr.text.to_numpy()[fidx].tolist(),
            tr.label.to_numpy()[fidx],
            ft["epochs"],
            ft["batch_size"],
            ft["lr"],
            self.seed,
        )
        self.models["minilm_ft"] = (model, tok, device)
        self.info["minilm_ft"] = {"train_s": train_s, "train_n": len(fidx)}

        p_va = self._score(va_texts, X_va)
        for name, p in p_va.items():
            self.thresholds[name] = pick_threshold(y_va, p, self.pc["target_recall"])
            self.info[name].update(
                {"threshold": self.thresholds[name], "val": gate_metrics(y_va, p)}
            )
            log.info(
                "gate %-10s thr=%.3f val AUC=%.3f",
                name,
                self.thresholds[name],
                self.info[name]["val"]["roc_auc"],
            )
        return self

    # ------------------------------------------------------------------ scoring
    def score(self, name: str, texts: list[str], X: np.ndarray | None = None) -> np.ndarray:
        """P(contains PII) from one gate; ``X`` lets callers reuse already-computed features."""
        if not texts:
            return np.zeros(0)
        if name == "minilm_ft":
            model, tok, device = self.models[name]
            return minilm_predict(model, tok, texts, device)
        if X is None:
            X = self.scaler.transform(self.embed(texts))
        if name == "vqc":
            return self.models[name].predict_proba(X)
        return self.models[name].predict_proba(X)[:, 1]

    def _score(self, texts: list[str], X: np.ndarray) -> dict[str, np.ndarray]:
        return {k: self.score(k, texts, X) for k in self.models}

    def score_all(self, texts: list[str]) -> dict[str, np.ndarray]:
        """P(contains PII) for every sentence from every gate (embeds once)."""
        if not texts:
            return {k: np.zeros(0) for k in self.models}
        return self._score(texts, self.scaler.transform(self.embed(texts)))

    def flags(self, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Boolean pass/skip decision per gate using the validation-chosen thresholds."""
        return {k: v >= self.thresholds[k] for k, v in scores.items()}

    # ------------------------------------------------------------------ persistence
    def save(self, out_dir: str | Path) -> Path:
        """Write every fitted gate and its threshold to ``out_dir`` (reload with ``Gates.load``)."""
        out = ensure_dir(out_dir)
        classical = {k: copy.copy(m) for k, m in self.models.items() if k in ("logreg", "mlp")}
        for m in classical.values():  # training-only RNG; its pickle is numpy-version specific
            if hasattr(m, "_random_state"):
                m._random_state = None
        joblib.dump({"scaler": self.scaler, **classical}, out / "classical.joblib")
        if "vqc" in self.models:
            v = self.models["vqc"]
            torch.save(
                {
                    "weights": v.weights.detach().cpu(),
                    "scale": v.scale.detach().cpu(),
                    "bias": v.bias.detach().cpu(),
                    "n_qubits": v.n,
                    "n_layers": v.L,
                    "seed": v.seed,
                },
                out / "vqc.pt",
            )
        if "minilm_ft" in self.models:
            torch.save(self.models["minilm_ft"][0].state_dict(), out / "minilm_gate.pt")
        meta = {
            "gates": sorted(self.models),
            "thresholds": self.thresholds,
            "embed_model": self.g["embed_model"],
            "embed_batch_size": self.g["embed_batch_size"],
            "n_features": self.g["n_features"],
            "target_recall": self.pc["target_recall"],
            "info": self.info,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        log.info("saved gates to %s", out)
        return out

    def save_split(self, out_dir: str | Path, encoder_from: str | None = None) -> Path:
        """Write one subfolder per gate plus ``shared/`` (see ``SPLIT_LAYOUT``).

        ``shared/encoder`` is a copy of the embedding model, so the folder works offline.
        """
        from sentence_transformers import SentenceTransformer

        out = ensure_dir(out_dir)
        shared = ensure_dir(out / "shared")
        flat = self.save(shared / "_flat")  # reuse the single-folder writer, then split it
        sk = joblib.load(flat / "classical.joblib")
        joblib.dump(sk["scaler"], shared / "feature_scaler.joblib")
        for i, name in enumerate(GATE_ORDER, start=1):
            if name not in self.models:
                continue
            sub = ensure_dir(out / f"{i:03d}_{name}")
            if name in ("logreg", "mlp"):
                joblib.dump(sk[name], sub / "model.joblib")
            else:
                src = flat / ("vqc.pt" if name == "vqc" else "minilm_gate.pt")
                src.replace(sub / "model.pt")
        (flat / "meta.json").replace(shared / "meta.json")
        for f in flat.iterdir():
            f.unlink()
        flat.rmdir()
        SentenceTransformer(encoder_from or self.g["embed_model"], device="cpu").save(
            str(shared / "encoder")
        )
        log.info("saved gates (split layout) to %s", out)
        return out

    @staticmethod
    def _files(d: Path) -> dict[str, Path]:
        """Where each part lives, for both the single-folder and the split layout."""
        if (d / "shared" / "meta.json").exists():
            files = {
                "meta": d / "shared" / "meta.json",
                "encoder": d / "shared" / "encoder",
                "scaler": d / "shared" / "feature_scaler.joblib",
            }
            for sub in d.iterdir():
                name = sub.name.split("_", 1)[-1]
                if sub.is_dir() and name in GATE_ORDER:
                    ext = "joblib" if name in ("logreg", "mlp") else "pt"
                    files[name] = sub / f"model.{ext}"
            return files
        return {
            "meta": d / "meta.json",
            "encoder": d / "encoder",
            "classical": d / "classical.joblib",
            "vqc": d / "vqc.pt",
            "minilm_ft": d / "minilm_gate.pt",
        }

    @classmethod
    def load(cls, in_dir: str | Path, only: list[str] | None = None) -> Gates:
        """Load gates written by ``save`` or ``save_split``; ``only`` limits which are loaded.

        The ``.joblib`` files are pickles: only load files you created yourself.
        """
        f = cls._files(Path(in_dir))
        meta = json.loads(f["meta"].read_text())
        if (f["encoder"] / "config.json").exists():  # bundled copy of the embedding model
            meta["embed_model"] = str(f["encoder"])
        cfg = {
            "gate": {
                "embed_model": meta["embed_model"],
                "embed_batch_size": meta["embed_batch_size"],
                "n_features": meta["n_features"],
            },
            "pipeline": {"seed": 0, "target_recall": meta["target_recall"]},
        }
        g = cls(cfg, 0)
        want = (set(only) if only else set(meta["gates"])) & set(meta["gates"])
        if "classical" in f:
            sk = joblib.load(f["classical"])
            g.scaler = sk["scaler"]
            for k in ("logreg", "mlp"):
                if k in want and k in sk:
                    g.models[k] = sk[k]
        else:
            g.scaler = joblib.load(f["scaler"])
            for k in ("logreg", "mlp"):
                if k in want and k in f:
                    g.models[k] = joblib.load(f[k])
        if "vqc" in want:
            state = torch.load(f["vqc"], map_location="cpu")
            v = VQC(state["n_qubits"], state["n_layers"], seed=state["seed"])
            with torch.no_grad():
                v.weights.copy_(state["weights"])
                v.scale.copy_(state["scale"])
                v.bias.copy_(state["bias"])
            g.models["vqc"] = v
        if "minilm_ft" in want:
            from transformers import AutoTokenizer

            model = MiniLMClassifier(meta["embed_model"]).to(g.device)
            model.load_state_dict(torch.load(f["minilm_ft"], map_location=g.device))
            model.eval()
            g.models["minilm_ft"] = (
                model,
                AutoTokenizer.from_pretrained(meta["embed_model"]),
                g.device,
            )
        g.thresholds = {k: v for k, v in meta["thresholds"].items() if k in g.models}
        g.info = meta["info"]
        return g
