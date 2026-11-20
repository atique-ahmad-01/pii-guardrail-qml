"""DeBERTa-v3-small token classifier (BIO tags) trained with the Hugging Face Trainer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
from torch.nn.functional import cross_entropy
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from piiguard.data.bio import IGNORE, bio_to_spans, spans_to_bio
from piiguard.utils import get_device, set_seed

Doc = dict[str, Any]


class BioDataset(torch.utils.data.Dataset):
    """Tokenised documents with BIO labels aligned through the tokenizer's offset mapping."""

    def __init__(self, docs: list[Doc], tok: Any, label2id: dict[str, int], max_length: int):
        self.items: list[dict[str, Any]] = []
        self.truncated = 0
        for d in docs:
            enc = tok(
                d["text"], truncation=True, max_length=max_length, return_offsets_mapping=True
            )
            offsets = [tuple(o) for o in enc["offset_mapping"]]
            last = max((e for s, e in offsets), default=0)
            self.truncated += int(last < len(d["text"].rstrip()))
            self.items.append(
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": spans_to_bio(offsets, d["spans"], label2id),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.items[i]


class MeanCETrainer(Trainer):
    """Plain mean cross-entropy over labelled tokens.

    Independent of the Trainer's ``num_items_in_batch`` normalisation, which changed between
    transformers versions and made the loss explode on transformers 5.x.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        out = model(**inputs)
        n = out.logits.shape[-1]
        loss = cross_entropy(
            out.logits.reshape(-1, n).float(), labels.reshape(-1), ignore_index=IGNORE
        )
        return (loss, out) if return_outputs else loss


def _compute_metrics(id2label: dict[int, str]) -> Any:
    def fn(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=-1)
        y_true, y_pred = [], []
        for p, y in zip(pred, labels, strict=True):
            keep = y != IGNORE
            y_true.append([id2label[int(i)] for i in y[keep]])
            y_pred.append([id2label[int(i)] for i in p[keep]])
        return {"f1": f1_score(y_true, y_pred, mode="strict", scheme=IOB2)}

    return fn


def train_deberta(
    cfg: dict[str, Any],
    train_docs: list[Doc],
    val_docs: list[Doc],
    label_list: list[str],
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Fine-tune with early stopping on validation entity F1; saves the best model."""
    mc = cfg["masker"]["deberta"]
    set_seed(seed)
    device = get_device()
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = dict(enumerate(label_list))
    tok = AutoTokenizer.from_pretrained(mc["model"])
    tr = BioDataset(train_docs, tok, label2id, mc["max_length"])
    va = BioDataset(val_docs, tok, label2id, mc["max_length"])
    model = AutoModelForTokenClassification.from_pretrained(
        mc["model"], num_labels=len(label_list), id2label=id2label, label2id=label2id
    ).float()  # transformers 5.x loads the stored low-precision dtype; AdamW then yields NaNs
    steps = math.ceil(len(tr) / mc["batch_size"]) * mc["epochs"]
    args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=mc["epochs"],
        per_device_train_batch_size=mc["batch_size"],
        per_device_eval_batch_size=32,
        learning_rate=mc["lr"],
        weight_decay=0.01,
        warmup_steps=int(0.1 * steps),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=device == "cuda",
        seed=seed,
        report_to="none",
        logging_steps=50,
    )
    trainer = MeanCETrainer(
        model=model,
        args=args,
        train_dataset=tr,
        eval_dataset=va,
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=_compute_metrics(id2label),
        callbacks=[EarlyStoppingCallback(mc["early_stopping_patience"])],
    )
    import time

    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0
    best = trainer.state.best_metric
    trainer.save_model(str(out_dir / "best"))
    tok.save_pretrained(str(out_dir / "best"))
    return {
        "train_s": train_s,
        "best_val_f1": float(best) if best is not None else None,
        "epochs_run": float(trainer.state.epoch),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_truncated_docs": tr.truncated,
        "log_history": trainer.state.log_history,
    }


def _tokenizer_compat(path: Path) -> dict[str, Any]:
    """Let transformers 4.x load a tokenizer saved by 5.x.

    5.x writes ``extra_special_tokens`` as a list; 4.x expects a dict and crashes. For DeBERTa
    the list only repeats [PAD]/[CLS]/[SEP], which are already special tokens, so it is dropped.
    """
    import transformers

    cfg = Path(path) / "tokenizer_config.json"
    if int(transformers.__version__.split(".")[0]) >= 5 or not cfg.exists():
        return {}
    extra = json.loads(cfg.read_text()).get("extra_special_tokens")
    return {"extra_special_tokens": {}} if isinstance(extra, list) else {}


class DebertaMasker:
    """Loads a saved token classifier and predicts character spans."""

    def __init__(self, path: Path, max_length: int) -> None:
        self.device = get_device()
        self.tok = AutoTokenizer.from_pretrained(str(path), **_tokenizer_compat(path))
        self.model = (
            AutoModelForTokenClassification.from_pretrained(str(path)).float().to(self.device)
        )
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.max_length = max_length
        self.truncated = 0

    @torch.no_grad()
    def predict(self, texts: list[str], batch_size: int = 32) -> list[list[dict[str, Any]]]:
        """Predicted spans for each text (decoded from the argmax BIO tags)."""
        out: list[list[dict[str, Any]]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = self.tok(
                chunk,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
            tags = logits.argmax(-1).cpu().numpy()
            for text, off, tg in zip(chunk, offsets.tolist(), tags, strict=True):
                off = [tuple(o) for o in off]
                self.truncated += int(max((e for s, e in off), default=0) < len(text.rstrip()))
                out.append(bio_to_spans(text, off, tg.tolist(), self.id2label))
        return out


def save_label_list(path: Path, labels: list[str]) -> None:
    """Persist the BIO label list next to a checkpoint."""
    path.write_text(json.dumps(labels))
