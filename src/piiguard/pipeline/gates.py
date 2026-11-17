"""Fit the four gates on the same embeddings and score sentences with any of them."""

from __future__ import annotations

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
        classical = {k: m for k, m in self.models.items() if k in ("logreg", "mlp")}
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

    @classmethod
    def load(cls, in_dir: str | Path, only: list[str] | None = None) -> Gates:
        """Load gates written by ``save``; ``only`` restricts which gates are loaded.

        The ``.joblib`` file is a pickle: only load files you created yourself.
        """
        d = Path(in_dir)
        meta = json.loads((d / "meta.json").read_text())
        cfg = {
            "gate": {
                "embed_model": meta["embed_model"],
                "embed_batch_size": meta["embed_batch_size"],
                "n_features": meta["n_features"],
            },
            "pipeline": {"seed": 0, "target_recall": meta["target_recall"]},
        }
        g = cls(cfg, 0)
        want = set(only) if only else set(meta["gates"])
        sk = joblib.load(d / "classical.joblib")
        g.scaler = sk["scaler"]
        for k in ("logreg", "mlp"):
            if k in want and k in sk:
                g.models[k] = sk[k]
        if "vqc" in want and "vqc" in meta["gates"]:
            state = torch.load(d / "vqc.pt", map_location="cpu")
            v = VQC(state["n_qubits"], state["n_layers"], seed=state["seed"])
            with torch.no_grad():
                v.weights.copy_(state["weights"])
                v.scale.copy_(state["scale"])
                v.bias.copy_(state["bias"])
            g.models["vqc"] = v
        if "minilm_ft" in want and "minilm_ft" in meta["gates"]:
            from transformers import AutoTokenizer

            model = MiniLMClassifier(meta["embed_model"]).to(g.device)
            model.load_state_dict(torch.load(d / "minilm_gate.pt", map_location=g.device))
            model.eval()
            g.models["minilm_ft"] = (
                model,
                AutoTokenizer.from_pretrained(meta["embed_model"]),
                g.device,
            )
        g.thresholds = {k: v for k, v in meta["thresholds"].items() if k in g.models}
        g.info = meta["info"]
        return g
