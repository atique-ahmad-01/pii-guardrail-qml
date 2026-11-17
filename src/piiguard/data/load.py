"""Load ai4privacy/pii-masking-300k, repair span offsets, and draw stratified splits."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

Span = dict[str, Any]  # {"start": int, "end": int, "label": str, "value": str}
Doc = dict[str, Any]  # {"id": str, "language": str, "text": str, "spans": list[Span]}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def download_file(repo: str, filename: str, raw_dir: str | Path) -> Path:
    """Download one dataset file from the Hugging Face Hub into ``raw_dir`` (cached)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset", local_dir=str(raw_dir)))


def fix_spans(text: str, spans: Sequence[dict[str, Any]]) -> tuple[list[Span], Counter]:
    """Return spans whose offsets reproduce their ``value`` in ``text``.

    The raw data has a small share of spans where ``text[start:end] != value``. Such a span is
    moved to the occurrence of ``value`` nearest to its stated start, or dropped if ``value`` does
    not occur. Overlapping spans are also dropped (BIO tags cannot represent them); the earlier
    span wins. Counts of every outcome are returned so the data card can report them.
    """
    stats: Counter = Counter()
    fixed: list[Span] = []
    for m in sorted(spans, key=lambda m: (m["start"], m["end"])):
        s, e, v = int(m["start"]), int(m["end"]), m["value"]
        if text[s:e] == v:
            stats["ok"] += 1
        else:
            occ, i = [], text.find(v)
            while v and i != -1:
                occ.append(i)
                i = text.find(v, i + 1)
            if not occ:
                stats["dropped_value_not_found"] += 1
                continue
            s = min(occ, key=lambda o: abs(o - s))
            e = s + len(v)
            stats["relocated"] += 1
        fixed.append({"start": s, "end": e, "label": m["label"], "value": v})
    fixed.sort(key=lambda sp: (sp["start"], sp["end"]))
    result: list[Span] = []
    for sp in fixed:
        if result and sp["start"] < result[-1]["end"]:
            stats["dropped_overlap"] += 1
            continue
        result.append(sp)
    return result, stats


def to_docs(rows: Sequence[dict[str, Any]]) -> tuple[list[Doc], Counter]:
    """Convert raw rows into ``Doc`` dicts; also return repair stats.

    A document is excluded entirely if any of its spans had to be dropped (value not in the text,
    or overlapping). In the source data such spans are corrupted annotations whose real PII is then
    left unlabeled; keeping the document would count that PII as clean text and understate leaks.
    Excluded documents are counted under ``docs_excluded``.
    """
    docs: list[Doc] = []
    total: Counter = Counter()
    for r in rows:
        spans, stats = fix_spans(r["source_text"], r["privacy_mask"])
        total.update(stats)
        if stats["dropped_value_not_found"] or stats["dropped_overlap"]:
            total["docs_excluded"] += 1
            continue
        docs.append(
            {"id": r["id"], "language": r["language"], "text": r["source_text"], "spans": spans}
        )
    total["docs_kept"] = len(docs)
    return docs, total


def stratified_sample(keys: Sequence[str], n: int, seed: int) -> list[int]:
    """Pick ``n`` indices, proportional to each stratum's share (largest remainder), seeded."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    total = len(keys)
    n = min(n, total)
    quota = {k: len(v) * n / total for k, v in groups.items()}
    take = {k: int(q) for k, q in quota.items()}
    leftover = n - sum(take.values())
    for k in sorted(quota, key=lambda k: (quota[k] - take[k], k), reverse=True)[:leftover]:
        take[k] += 1
    chosen: list[int] = []
    for k in sorted(groups):
        idx = np.array(groups[k])
        rng.shuffle(idx)
        chosen.extend(idx[: take[k]].tolist())
    return sorted(chosen)


def rarest_label_key(docs: Sequence[Doc]) -> list[str]:
    """Stratum key per document: its globally rarest label, or ``NONE`` if it has no PII."""
    freq = Counter(sp["label"] for d in docs for sp in d["spans"])
    keys = []
    for d in docs:
        labels = {sp["label"] for sp in d["spans"]}
        keys.append(min(labels, key=lambda lab: (freq[lab], lab)) if labels else "NONE")
    return keys


def split_docs(
    train_pool: Sequence[Doc],
    test_pool: Sequence[Doc],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    key_fn: Callable[[Sequence[Doc]], list[str]] = rarest_label_key,
) -> dict[str, list[Doc]]:
    """Draw disjoint train/val from ``train_pool`` and test from ``test_pool`` (all stratified)."""
    keys = key_fn(train_pool)
    train_idx = stratified_sample(keys, n_train, seed)
    taken = set(train_idx)
    rest = [i for i in range(len(train_pool)) if i not in taken]
    val_rel = stratified_sample([keys[i] for i in rest], n_val, seed + 1)
    val_idx = [rest[j] for j in val_rel]
    test_idx = stratified_sample(key_fn(test_pool), n_test, seed + 2)
    return {
        "train": [train_pool[i] for i in train_idx],
        "val": [train_pool[i] for i in val_idx],
        "test": [test_pool[i] for i in test_idx],
    }


def sample_docs(pool: Sequence[Doc], n: int, seed: int) -> list[Doc]:
    """Stratified sample of ``n`` documents from one pool."""
    return [pool[i] for i in stratified_sample(rarest_label_key(pool), n, seed)]


def label_counts(docs: Sequence[Doc]) -> Counter:
    """Number of gold spans per label."""
    return Counter(sp["label"] for d in docs for sp in d["spans"])
