"""Gate features: MiniLM embeddings -> PCA (8 dims, fit on train only) -> scaled to [0, pi]."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from piiguard.utils import ensure_dir, get_device, get_logger

log = get_logger("gate.features")


def load_gate_frames(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Balanced sentence tables written by ``prepare_data.py`` (train pool, val, test)."""
    d = Path(data_dir)
    return {s: pd.read_parquet(d / f"gate_{s}_balanced.parquet") for s in ("train", "val", "test")}


def balanced_subset(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Sorted indices of ``n`` rows with equal class counts (fewer if a class is too small)."""
    rng = np.random.default_rng(seed)
    per = min(n // 2, int((y == 1).sum()), int((y == 0).sum()))
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False) for c in (0, 1)])
    return np.sort(idx)


def embed(texts: list[str], model_name: str, device: str, batch_size: int = 256) -> np.ndarray:
    """L2-normalised sentence embeddings (float32).

    Sequences are capped at 128 tokens. If the accelerator runs out of memory the call is retried
    on CPU (a warning is logged), so a small GPU never aborts a run.
    """
    from sentence_transformers import SentenceTransformer

    def run(dev: str) -> np.ndarray:
        model = SentenceTransformer(model_name, device=dev)
        model.max_seq_length = 128
        out = model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(out, dtype=np.float32)

    try:
        return run(device)
    except RuntimeError as exc:
        if device == "cpu" or "out of memory" not in str(exc).lower():
            raise
        log.warning("%s out of memory while embedding; retrying on CPU", device)
        return run("cpu")


class AngleScaler:
    """PCA to ``n_components`` then min-max scaling to [0, pi]; fit on the train pool only."""

    def __init__(self, n_components: int = 8, seed: int = 0) -> None:
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.lo: np.ndarray | None = None
        self.hi: np.ndarray | None = None

    def fit(self, emb: np.ndarray) -> AngleScaler:
        z = self.pca.fit_transform(emb)
        self.lo, self.hi = z.min(axis=0), z.max(axis=0)
        return self

    def transform(self, emb: np.ndarray) -> np.ndarray:
        z = self.pca.transform(emb)
        out = (z - self.lo) / (self.hi - self.lo) * np.pi
        return np.clip(out, 0.0, np.pi)  # val/test may fall outside the train range


def build_features(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Embed, reduce and cache features for the train pool, val and test (balanced) sets."""
    g = cfg["gate"]
    cache = ensure_dir(cfg["paths"]["cache_dir"]) / f"gate_features_{cfg['profile']}.npz"
    frames = load_gate_frames(cfg["paths"]["data_dir"])
    sizes = {s: len(f) for s, f in frames.items()}
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=True)
        meta = z["meta"].item()
        if (
            meta["sizes"] == sizes
            and meta["model"] == g["embed_model"]
            and (meta["n_features"] == g["n_features"])
        ):
            log.info("loaded cached features %s", cache)
            return {**{k: z[k] for k in z.files if k != "meta"}, "frames": frames, **meta}
    device = get_device()
    log.info("embedding %s sentences on %s", sizes, device)
    t0 = time.perf_counter()
    emb = {
        s: embed(f.text.tolist(), g["embed_model"], device, g["embed_batch_size"])
        for s, f in frames.items()
    }
    embed_s = time.perf_counter() - t0
    scaler = AngleScaler(g["n_features"], cfg["seed"]).fit(emb["train"])
    arrays: dict[str, Any] = {}
    for s in frames:
        arrays[f"X_{s}"] = scaler.transform(emb[s])
        arrays[f"y_{s}"] = frames[s].label.to_numpy().astype(int)
    # latency of the shared front end (embed + PCA) for a few single sentences
    texts = frames["test"].text.tolist()[:20]
    t1 = time.perf_counter()
    for tx in texts:
        scaler.transform(embed([tx], g["embed_model"], device))
    front_ms = (time.perf_counter() - t1) / len(texts) * 1000
    meta = {
        "sizes": sizes,
        "model": g["embed_model"],
        "n_features": g["n_features"],
        "pca_explained_variance": [float(v) for v in scaler.pca.explained_variance_ratio_],
        "pca_explained_variance_total": float(scaler.pca.explained_variance_ratio_.sum()),
        "embed_total_s": embed_s,
        "front_end_ms_per_sentence_incl_model_load": front_ms,
        "device": device,
    }
    np.savez(cache, meta=np.array(meta, dtype=object), **arrays)
    log.info(
        "cached features -> %s (PCA keeps %.1f%% variance)",
        cache,
        100 * meta["pca_explained_variance_total"],
    )
    return {**arrays, "frames": frames, **meta}
