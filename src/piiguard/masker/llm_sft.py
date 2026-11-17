"""LoRA supervised fine-tuning of small instruction LLMs to emit PII as a JSON list."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from piiguard.utils import get_device, set_seed

IGNORE = -100


def chat_text(tok: Any, prompt: str) -> str:
    """The user turn rendered with the model's chat template, ready for the assistant reply."""
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def build_sft_example(tok: Any, prompt: str, target: str, max_length: int) -> dict[str, list[int]]:
    """Token ids and labels; loss is computed on the target (plus end token) only.

    If prompt + target is too long, the target is cut (the prompt is never cut).
    """
    p = tok(chat_text(tok, prompt), add_special_tokens=False)["input_ids"]
    t = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
    t = t[: max(0, max_length - len(p))]
    ids = p + t
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [IGNORE] * len(p) + t}


class SftDataset(torch.utils.data.Dataset):
    """Tokenised prompt/target pairs."""

    def __init__(self, tok: Any, prompts: list[str], targets: list[str], max_length: int) -> None:
        self.items = [
            build_sft_example(tok, p, t, max_length) for p, t in zip(prompts, targets, strict=True)
        ]
        self.target_lengths = [sum(1 for x in it["labels"] if x != IGNORE) for it in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        return self.items[i]


class PadCollator:
    """Right-pad ids, mask and labels to the longest sequence in the batch."""

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (n - len(b["input_ids"])) for b in batch]
        att = [b["attention_mask"] + [0] * (n - len(b["attention_mask"])) for b in batch]
        lab = [b["labels"] + [IGNORE] * (n - len(b["labels"])) for b in batch]
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(att),
            "labels": torch.tensor(lab),
        }


def load_tokenizer(name: str) -> Any:
    """Tokenizer with a pad token (falls back to EOS)."""
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def model_dtype(device: str) -> torch.dtype:
    """bf16 on CUDA, float32 elsewhere (safest for LoRA training on MPS/CPU)."""
    return torch.bfloat16 if device == "cuda" else torch.float32


def train_lora(
    cfg: dict[str, Any],
    model_name: str,
    train_prompts: list[str],
    train_targets: list[str],
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Fine-tune a LoRA adapter (attention + MLP projections); saves it to ``out_dir``."""
    lc, sc = cfg["masker"]["lora"], cfg["masker"]["sft"]
    set_seed(seed)
    device = get_device()
    tok = load_tokenizer(model_name)
    ds = SftDataset(tok, train_prompts, train_targets, sc["max_length"])
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype(device))
    model = get_peft_model(
        model,
        LoraConfig(
            r=lc["r"],
            lora_alpha=lc["alpha"],
            lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    per_epoch = math.ceil(len(ds) / (sc["batch_size"] * sc["grad_accum"]))
    steps = sc.get("max_steps", -1)
    total_steps = steps if steps and steps > 0 else per_epoch * sc["epochs"]
    args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=sc["epochs"],
        max_steps=steps if steps and steps > 0 else -1,
        per_device_train_batch_size=sc["batch_size"],
        gradient_accumulation_steps=sc["grad_accum"],
        learning_rate=sc["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=int(0.03 * total_steps),
        bf16=device == "cuda",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds, data_collator=PadCollator(tok.pad_token_id)
    )
    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0
    model.save_pretrained(str(out_dir / "adapter"))
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    return {
        "train_s": train_s,
        "trainable_params": trainable,
        "total_params": total,
        "n_train": len(ds),
        "target_tokens_p50": float(np.percentile(ds.target_lengths, 50)),
        "target_tokens_p99": float(np.percentile(ds.target_lengths, 99)),
        "target_tokens_max": int(max(ds.target_lengths)),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "steps": int(trainer.state.global_step),
    }
