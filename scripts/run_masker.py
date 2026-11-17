"""Phase 4: train and evaluate the PII maskers.

Usage: python scripts/run_masker.py --config configs/smoke.yaml --model all
Models: deberta, qwen_lora, smollm_lora, qwen_zeroshot (or ``all``).
``--summary-only`` re-aggregates existing records.
Writes per-(model, set) run records, per-document counts and predictions under
``<results_dir>/masker/``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from piiguard.data.eval_sets import load_sets
from piiguard.data.load import frame_to_docs
from piiguard.eval.metrics import counts_frame, summarize_frame
from piiguard.masker.summary import summarize
from piiguard.utils import (
    ensure_dir,
    get_device,
    get_logger,
    load_config,
    set_seed,
    write_run_record,
)

log = get_logger("run_masker")
MODELS = ["deberta", "qwen_lora", "smollm_lora", "qwen_zeroshot"]


def load_cfg_sets(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Evaluation sets named in ``cfg['masker']['eval_sets']`` (optionally truncated)."""
    m = cfg["masker"]
    return load_sets(cfg["paths"]["data_dir"], m["eval_sets"], m.get("eval_max_docs"))


def evaluate(
    cfg: dict[str, Any],
    model: str,
    seed: int,
    sets: dict[str, dict[str, Any]],
    predict: Any,
    is_llm: bool,
    extra_cost: dict[str, Any],
) -> None:
    """Predict on every set, write counts/predictions parquet and run records."""
    out = ensure_dir(Path(cfg["paths"]["results_dir"]) / "masker")
    ensure_dir(out / "counts")
    ensure_dir(out / "preds")
    n_docs, t_total = 0, 0.0
    for name, s in sets.items():
        texts = [d["text"] for d in s["docs"]]
        t0 = time.perf_counter()
        spans, valid, raw = predict(s["prompts"], texts)
        dt = time.perf_counter() - t0
        n_docs, t_total = n_docs + len(texts), t_total + dt
        counts = counts_frame(texts, [d["spans"] for d in s["docs"]], spans, s["eval_labels"])
        counts.insert(0, "id", [d["id"] for d in s["docs"]])
        counts.to_parquet(out / "counts" / f"{model}__{name}__seed{seed}.parquet", index=False)
        pd.DataFrame(
            {
                "id": counts["id"],
                "raw_output": raw,
                "valid_json": valid,
                "spans_json": [json.dumps(x) for x in spans],
            }
        ).to_parquet(out / "preds" / f"{model}__{name}__seed{seed}.parquet", index=False)
        m = summarize_frame(counts.drop(columns="id"))
        m.update(
            {
                "kind": "masker_eval",
                "model": model,
                "set": name,
                "valid_json_rate": float(sum(valid) / len(valid)) if is_llm else None,
                "predict_s": dt,
            }
        )
        write_run_record(
            out, f"{model}__{name}", config=cfg, seed=seed, metrics=m, timings={"predict_s": dt}
        )
        log.info(
            "%s seed%d %-12s F1=%.3f char_leak=%.3f entity_leak=%.3f%s",
            model,
            seed,
            name,
            m["entity_f1"],
            m["char_leak_rate"],
            m["entity_leak_rate"],
            f" valid_json={m['valid_json_rate']:.2f}" if is_llm else "",
        )
    cost = {
        "kind": "masker_cost",
        "model": model,
        "throughput_docs_per_s": n_docs / t_total,
        "device": get_device(),
        **extra_cost,
    }
    write_run_record(
        out,
        f"{model}__cost",
        config=cfg,
        seed=seed,
        metrics=cost,
        timings={"train_s": extra_cost.get("train_s") or 0.0},
    )


def run_deberta(cfg: dict[str, Any], seed: int, sets: dict[str, Any]) -> None:
    """Train DeBERTa for one seed and evaluate it."""
    from piiguard.masker.token_clf import DebertaMasker, train_deberta

    d = Path(cfg["paths"]["data_dir"])
    label_list = json.loads((d / "labels.json").read_text())
    train = frame_to_docs(pd.read_parquet(d / "train.parquet"))
    val = frame_to_docs(pd.read_parquet(d / "val.parquet"))
    ck = ensure_dir(Path(cfg["paths"]["ckpt_dir"]) / f"deberta_seed{seed}")
    info = train_deberta(cfg, train, val, label_list, seed, ck)
    m = DebertaMasker(ck / "best", cfg["masker"]["deberta"]["max_length"])
    texts = [x for s in sets.values() for x in [d_["text"] for d_ in s["docs"]]][:8]
    if m.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t = []
    for tx in texts[: cfg["masker"]["n_latency_docs"]]:
        t0 = time.perf_counter()
        m.predict([tx])
        t.append((time.perf_counter() - t0) * 1000)
    import numpy as np

    cost = {
        "train_s": info["train_s"],
        "trainable_params": info["trainable_params"],
        "best_val_f1": info["best_val_f1"],
        "epochs_run": info["epochs_run"],
        "latency_ms_p50": float(np.percentile(t, 50)),
        "latency_ms_p95": float(np.percentile(t, 95)),
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if m.device == "cuda" else None,
        "train_truncated_docs": info["train_truncated_docs"],
    }

    def predict(prompts: list[str], texts: list[str]) -> Any:
        return m.predict(texts), [True] * len(texts), [""] * len(texts)

    evaluate(cfg, "deberta", seed, sets, predict, False, cost)
    log.info("deberta seed%d: %d eval docs truncated at max_length", seed, m.truncated)


def run_llm(cfg: dict[str, Any], model: str, seed: int, sets: dict[str, Any]) -> None:
    """Train (unless zero-shot) and evaluate one LLM masker for one seed."""
    from piiguard.masker.llm_infer import LLMMasker
    from piiguard.masker.llm_sft import train_lora

    mc = cfg["masker"]
    d = Path(cfg["paths"]["data_dir"])
    if model == "qwen_zeroshot":
        name, adapter, info = (
            mc["zero_shot"]["model"],
            None,
            {"train_s": 0.0, "trainable_params": 0},
        )
    else:
        name = mc["llms"][model.removesuffix("_lora")]["model"]
        train = pd.read_parquet(d / "train.parquet")
        ck = ensure_dir(Path(cfg["paths"]["ckpt_dir"]) / f"{model}_seed{seed}")
        info = train_lora(cfg, name, train["prompt"].tolist(), train["target"].tolist(), seed, ck)
        adapter = ck / "adapter"
        log.info(
            "%s seed%d trained: %s",
            model,
            seed,
            {k: v for k, v in info.items() if k != "log_history"},
        )
    set_seed(seed)
    m = LLMMasker(name, adapter, mc["inference"]["max_new_tokens"], mc["inference"]["batch_size"])
    first = next(iter(sets.values()))["prompts"]
    lat = m.latency(first, mc["n_latency_docs"])
    cost = {
        **{k: v for k, v in info.items() if k != "log_history"},
        **lat,
        "total_params": m.n_params,
    }
    evaluate(cfg, model, seed, sets, m.predict, True, cost)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="all", choices=[*MODELS, "all"])
    ap.add_argument("--seeds", type=int, nargs="*", help="override the config seed list")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(Path(cfg["paths"]["results_dir"]) / "masker")
    if not args.summary_only:
        sets = load_cfg_sets(cfg)
        log.info("eval sets: %s", {k: len(v["docs"]) for k, v in sets.items()})
        for model in MODELS if args.model == "all" else [args.model]:
            seeds = [0] if model == "qwen_zeroshot" else (args.seeds or cfg["masker"]["seeds"])
            for seed in seeds:
                t0 = time.perf_counter()
                (
                    run_deberta
                    if model == "deberta"
                    else lambda c, s, st, m=model: run_llm(c, m, s, st)
                )(cfg, seed, sets)
                log.info("%s seed%d finished in %.0fs", model, seed, time.perf_counter() - t0)
    ev, cost = summarize(out)
    ev.to_csv(out / "summary.csv", index=False)
    cost.to_csv(out / "summary_cost.csv", index=False)
    log.info("wrote summary.csv (%d rows) and summary_cost.csv", len(ev))


if __name__ == "__main__":
    main()
