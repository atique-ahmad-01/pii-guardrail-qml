"""Phase 1: build processed splits, gate sentence sets, SFT pairs and the data card.

Usage: python scripts/prepare_data.py --config configs/smoke.yaml [--skip-tokenizer]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from piiguard.data.bio import build_label_list, roundtrip_ok
from piiguard.data.load import (
    docs_to_frame,
    download_file,
    label_counts,
    read_jsonl,
    sample_docs,
    split_docs,
    to_docs,
)
from piiguard.data.sentences import balance, sentence_table
from piiguard.utils import ensure_dir, get_logger, load_config, set_seed, timed, write_run_record

log = get_logger("prepare_data")
FIX_HEADER = [
    "file",
    "spans ok",
    "relocated",
    "corrupted (value not found)",
    "overlap",
    "docs excluded",
    "docs kept",
]
LICENSE_URL = "https://huggingface.co/datasets/ai4privacy/pii-masking-300k"


def md_table(header: list[str], rows: list[list]) -> str:
    """Render a Markdown table."""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def alignment_stats(docs: list[dict], model_name: str, label_list: list[str], max_len: int) -> dict:
    """Round-trip every gold span through BIO encode/decode with the real tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    label2id = {lab: i for i, lab in enumerate(label_list)}
    total = exact = truncated = 0
    for d in docs:
        enc = tok(d["text"], return_offsets_mapping=True, truncation=True, max_length=max_len)
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        last = max((e for s, e in offsets), default=0)
        inside = [sp for sp in d["spans"] if sp["end"] <= last]
        truncated += len(d["spans"]) - len(inside)
        total += len(inside)
        exact += sum(roundtrip_ok(d["text"], inside, offsets, label2id))
    return {"tokenizer": model_name, "spans": total, "exact": exact, "truncated_spans": truncated}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-tokenizer", action="store_true", help="skip the BIO round-trip check")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg, paths, seed = cfg["data"], cfg["paths"], cfg["seed"]
    set_seed(seed)
    out_dir = ensure_dir(paths["data_dir"])
    timings: dict[str, float] = {}

    # ---- load raw files
    fix_stats: dict[str, Counter] = {}
    pools: dict[str, list[dict]] = {}
    with timed() as t:
        for key in ["train_en", "test_en", *dcfg["stress_langs"]]:
            path = download_file(dcfg["hf_dataset"], dcfg["files"][key], paths["raw_dir"])
            pools[key], fix_stats[key] = to_docs(read_jsonl(path))
            log.info("%s: %d docs, span fixes %s", key, len(pools[key]), dict(fix_stats[key]))
    timings["load_s"] = t.elapsed

    # ---- splits
    splits = split_docs(
        pools["train_en"], pools["test_en"], dcfg["n_train"], dcfg["n_val"], dcfg["n_test"], seed
    )
    for lang in dcfg["stress_langs"]:
        splits[f"cross_{lang}"] = sample_docs(pools[lang], dcfg["n_cross_lingual"], seed + 3)
    ids = {k: {d["id"] for d in v} for k, v in splits.items()}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])

    label_list = build_label_list(label_counts(splits["train"]).keys())
    (out_dir / "labels.json").write_text(json.dumps(label_list, indent=2))
    for name, docs in splits.items():
        docs_to_frame(docs).to_parquet(out_dir / f"{name}.parquet", index=False)

    # ---- gate sentence sets
    gate_stats: dict[str, dict] = {}
    for name in ["train", "val", "test"]:
        df = sentence_table(splits[name])
        bal = balance(df, seed)
        df.to_parquet(out_dir / f"gate_{name}_all.parquet", index=False)
        bal.to_parquet(out_dir / f"gate_{name}_balanced.parquet", index=False)
        gate_stats[name] = {
            "sentences": len(df),
            "positive": int((df.label == 1).sum()),
            "negative": int((df.label == 0).sum()),
            "balanced_per_class": int((bal.label == 1).sum()),
        }
    neg_train = gate_stats["train"]["negative"]
    gate_ok = neg_train >= dcfg["gate_balanced_min_negatives"]

    # ---- BIO alignment check (real tokenizer)
    align: dict | str = "skipped"
    if not args.skip_tokenizer:
        try:
            with timed() as t:
                mcfg = cfg["masker"]["deberta"]
                align = alignment_stats(
                    splits["test"], mcfg["model"], label_list, mcfg["max_length"]
                )
            timings["alignment_s"] = t.elapsed
        except Exception as exc:  # report, never hide
            align = f"FAILED: {type(exc).__name__}: {exc}"
            log.error("tokenizer alignment check failed: %s", align)

    # ---- data card
    train_labels = set(label_counts(splits["train"]))
    counts = {n: label_counts(d) for n, d in splits.items()}
    names = list(splits)
    label_rows = [
        [lab, *[counts[n].get(lab, 0) for n in names]]
        for lab in sorted(
            set().union(*counts.values()), key=lambda lab: -counts["train"].get(lab, 0)
        )
    ]
    split_rows = [
        [
            n,
            len(d),
            sum(len(x["spans"]) for x in d),
            sum(1 for x in d if not x["spans"]),
            sum(len(x["text"]) for x in d) // max(len(d), 1),
        ]
        for n, d in splits.items()
    ]
    fix_rows = [
        [
            k,
            *[
                v.get(f, 0)
                for f in (
                    "ok",
                    "relocated",
                    "dropped_value_not_found",
                    "dropped_overlap",
                    "docs_excluded",
                    "docs_kept",
                )
            ],
        ]
        for k, v in fix_stats.items()
    ]
    gate_rows = [
        [n, g["sentences"], g["positive"], g["negative"], g["balanced_per_class"]]
        for n, g in gate_stats.items()
    ]
    unseen = {
        n: sorted(set(counts[n]) - train_labels) for n in names if set(counts[n]) - train_labels
    }
    card = f"""# Data card ({cfg["profile"]} profile)

Generated by `scripts/prepare_data.py`. Source: [{dcfg["hf_dataset"]}]({LICENSE_URL}).
The dataset has its own license: check the dataset page before redistributing derived data.
Seed: {seed}.

## Splits
Train/val are drawn from the English *train* file, disjoint. Test is drawn from the English
*validation* file. Cross-lingual sets come from the German/French/Italian *validation* files and are
never used for training. Sampling is stratified by each document's rarest label.

{md_table(["split", "docs", "spans", "docs without PII", "mean chars/doc"], split_rows)}

## Span offset repair (per source file, all rows before sampling)
`text[start:end]` did not always equal the stated `value`. Mismatches were relocated to the nearest
occurrence of `value`. Spans whose value is not in the text are corrupted annotations in the source
(the value holds template leftovers such as `[GIVENNAME_B(...`). A document containing such a span,
or an overlapping span, is **excluded** so unlabeled PII never counts as clean text.

{md_table(FIX_HEADER, fix_rows)}

## Spans per label
{md_table(["label", *names], label_rows)}

Labels present in a split but absent from train: {unseen if unseen else "none"}.

## Gate (sentence-level) class balance
A sentence is positive if any PII span overlaps it. The balanced set subsamples the larger class.
Negative sentences in train: {neg_train}; required minimum: {dcfg["gate_balanced_min_negatives"]}
-> {"OK" if gate_ok else "TOO FEW NEGATIVES: a fix is needed (see notes)"}.

{md_table(["split", "sentences", "positive", "negative", "balanced per class"], gate_rows)}

## BIO alignment round-trip (test split)
{json.dumps(align, indent=2) if isinstance(align, dict) else align}

## Files
`<split>.parquet` columns: `id, language, text, spans_json, prompt, target`.
`gate_<split>_all.parquet` and `gate_<split>_balanced.parquet`: sentence tables.
`labels.json`: BIO label list (index = class id).
"""
    (out_dir / "data_card.md").write_text(card)

    metrics = {
        "splits": {n: len(d) for n, d in splits.items()},
        "span_fix": {k: dict(v) for k, v in fix_stats.items()},
        "gate": gate_stats,
        "gate_negatives_ok": gate_ok,
        "bio_alignment": align,
    }
    rec = write_run_record(
        Path(paths["results_dir"]) / "data",
        "prepare_data",
        config=cfg,
        seed=seed,
        metrics=metrics,
        timings=timings,
    )
    log.info("wrote %s and %s", out_dir / "data_card.md", rec)


if __name__ == "__main__":
    main()
