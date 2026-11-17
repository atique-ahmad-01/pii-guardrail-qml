"""Phase 5: two-stage guardrail (gate -> masker) versus masker-only baselines.

Arms (all scored on the same documents with the same metrics):
  masker_doc    the masker on the whole document (as in Phase 4)
  masker_sent   the masker on every sentence (isolates the effect of sentence splitting)
  gate_<name>   only sentences the gate flags are shown to the masker
  gate_oracle   only sentences that truly contain PII (upper bound for any gate)

Writes run records, per-document counts and ``summary.csv`` under ``<results_dir>/pipeline/``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from piiguard.data.eval_sets import load_sets
from piiguard.data.sentences import label_sentences, split_sentences
from piiguard.eval.metrics import counts_frame, summarize_frame
from piiguard.masker.token_clf import DebertaMasker
from piiguard.pipeline.assemble import gate_sentence_stats, gated_spans
from piiguard.pipeline.gates import Gates
from piiguard.pipeline.summary import summarize
from piiguard.utils import ensure_dir, get_logger, load_config, set_seed, write_run_record

log = get_logger("run_pipeline")


def run_set(
    cfg: dict[str, Any],
    name: str,
    s: dict[str, Any],
    gates: Gates,
    masker: DebertaMasker,
    out: Path,
    seed: int,
) -> None:
    """Score every arm on one evaluation set."""
    docs = s["docs"]
    texts = [d["text"] for d in docs]
    golds = [d["spans"] for d in docs]
    offs = [split_sentences(t) for t in texts]
    sent_texts = [t[a:b] for t, o in zip(texts, offs, strict=True) for a, b in o]
    sent_y = np.array([r["label"] for d in docs for r in label_sentences(d)])
    bounds = np.cumsum([0] + [len(o) for o in offs])

    scores = gates.score_all(sent_texts)
    flags = gates.flags(scores)
    flags["oracle"] = sent_y.astype(bool)
    all_sent_spans = masker.predict(sent_texts)
    arms: dict[str, list[list[dict[str, Any]]]] = {"masker_doc": masker.predict(texts)}
    arms["masker_sent"] = _assemble(offs, all_sent_spans, np.ones(len(sent_texts), bool), bounds)
    for g, f in flags.items():
        arms[f"gate_{g}"] = _assemble(offs, all_sent_spans, f, bounds)

    for arm, preds in arms.items():
        counts = counts_frame(texts, golds, preds, s["eval_labels"])
        counts.insert(0, "id", [d["id"] for d in docs])
        counts.to_parquet(out / "counts" / f"{arm}__{name}.parquet", index=False)
        m = summarize_frame(counts.drop(columns="id"))
        m.update({"kind": "pipeline_eval", "arm": arm, "set": name})
        if arm.startswith("gate_"):
            m.update(gate_sentence_stats(sent_y, flags[arm.removeprefix("gate_")]))
        elif arm == "masker_sent":
            m.update(gate_sentence_stats(sent_y, np.ones(len(sent_y), bool)))
        write_run_record(out, f"pipeline__{arm}__{name}", config=cfg, seed=seed, metrics=m)
        log.info(
            "%-12s %-14s F1=%.3f char_leak=%.4f skipped=%s",
            name,
            arm,
            m["entity_f1"],
            m["char_leak_rate"],
            f"{m['skipped_fraction']:.2f}" if "skipped_fraction" in m else "-",
        )


def _assemble(
    offs: list[list[tuple[int, int]]],
    sent_spans: list[list[dict[str, Any]]],
    flags: np.ndarray,
    bounds: np.ndarray,
) -> list[list[dict[str, Any]]]:
    """Per-document spans built from the flagged sentences' masker output."""
    return [
        gated_spans(o, sent_spans[bounds[i] : bounds[i + 1]], flags[bounds[i] : bounds[i + 1]])
        for i, o in enumerate(offs)
    ]


def measure_latency(
    cfg: dict[str, Any],
    docs: list[dict[str, Any]],
    gates: Gates,
    masker: DebertaMasker,
    out: Path,
    seed: int,
) -> None:
    """End-to-end latency per document (one document at a time) for every arm."""
    n = cfg["pipeline"]["latency_docs"]
    docs = docs[:n]
    for d in docs[:2]:  # warm-up
        masker.predict([d["text"]])
        gates.score_all([d["text"]])

    def timed(fn: Any) -> dict[str, float]:
        ts = []
        for d in docs:
            t0 = time.perf_counter()
            fn(d["text"])
            ts.append((time.perf_counter() - t0) * 1000)
        return {
            "latency_ms_p50": float(np.percentile(ts, 50)),
            "latency_ms_p95": float(np.percentile(ts, 95)),
        }

    def via_gate(g: str) -> Any:
        def fn(text: str) -> None:
            offs = split_sentences(text)
            sents = [text[a:b] for a, b in offs]
            keep = gates.score(g, sents) >= gates.thresholds[g]
            masker.predict([x for x, k in zip(sents, keep, strict=True) if k])

        return fn

    def sentences_only(text: str) -> None:
        masker.predict([text[a:b] for a, b in split_sentences(text)])

    rows = {
        "masker_doc": timed(lambda t: masker.predict([t])),
        "masker_sent": timed(sentences_only),
    }
    for g in gates.models:
        rows[f"gate_{g}"] = timed(via_gate(g))
    for arm, r in rows.items():
        write_run_record(
            out,
            f"pipeline__{arm}__latency",
            config=cfg,
            seed=seed,
            metrics={"kind": "pipeline_latency", "arm": arm, "n_docs": len(docs), **r},
        )
        log.info(
            "latency %-14s p50=%.1f ms p95=%.1f ms", arm, r["latency_ms_p50"], r["latency_ms_p95"]
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pc = cfg["pipeline"]
    out = ensure_dir(Path(cfg["paths"]["results_dir"]) / "pipeline")
    if not args.summary_only:
        ensure_dir(out / "counts")
        seed = pc["seed"]
        set_seed(seed)
        t0 = time.perf_counter()
        gates = Gates(cfg, seed).fit()
        write_run_record(
            out,
            "pipeline__gates",
            config=cfg,
            seed=seed,
            metrics={"kind": "pipeline_gates", "info": gates.info},
            timings={"fit_s": time.perf_counter() - t0},
        )
        masker = DebertaMasker(Path(pc["ckpt"]), cfg["masker"]["deberta"]["max_length"])
        sets = load_sets(cfg["paths"]["data_dir"], pc["eval_sets"], pc["eval_n"])
        for name, s in sets.items():
            run_set(cfg, name, s, gates, masker, out, seed)
        measure_latency(
            cfg,
            sets["clean"]["docs"] if "clean" in sets else next(iter(sets.values()))["docs"],
            gates,
            masker,
            out,
            seed,
        )
    df = summarize(out)
    df.to_csv(out / "summary.csv", index=False)
    log.info("wrote %s (%d rows)", out / "summary.csv", len(df))


if __name__ == "__main__":
    main()
