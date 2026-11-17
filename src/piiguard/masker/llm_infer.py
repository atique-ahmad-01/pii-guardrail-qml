"""Greedy-decoding inference for the LLM maskers (fine-tuned adapter or zero-shot base)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from piiguard.masker.llm_sft import chat_text, load_tokenizer, model_dtype
from piiguard.masker.spans import map_items_to_spans, parse_llm_output
from piiguard.utils import get_device


class LLMMasker:
    """A base model, optionally with a LoRA adapter merged in, that returns PII spans."""

    def __init__(self, model_name: str, adapter: Path | None, max_new_tokens: int, batch_size: int):
        self.device = get_device()
        self.tok = load_tokenizer(model_name)
        self.tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=model_dtype(self.device)
        )
        if adapter is not None:
            model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def generate(self, prompts: list[str], batch_size: int | None = None) -> list[str]:
        """Greedy continuation of each prompt (decoded new tokens only, input order kept)."""
        bs = batch_size or self.batch_size
        texts = [chat_text(self.tok, p) for p in prompts]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths per batch
        out = [""] * len(texts)
        for i in range(0, len(order), bs):
            idx = order[i : i + bs]
            enc = self.tok(
                [texts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            gen = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1] :]
            for j, row in zip(idx, new, strict=True):
                out[j] = self.tok.decode(row, skip_special_tokens=True)
        return out

    def predict(
        self, prompts: list[str], texts: list[str]
    ) -> tuple[list[list[dict[str, Any]]], list[bool], list[str]]:
        """Spans, valid-JSON flags and raw outputs for each document."""
        raw = self.generate(prompts)
        spans, valid = [], []
        for text, r in zip(texts, raw, strict=True):
            items, ok = parse_llm_output(r)
            valid.append(ok)
            spans.append(map_items_to_spans(text, items) if ok else [])
        return spans, valid, raw

    def latency(self, prompts: list[str], n: int) -> dict[str, float | None]:
        """Per-document latency with batch size 1 (p50/p95 ms) and peak CUDA memory."""
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.generate(prompts[:1], batch_size=1)  # warm-up
        times = []
        for p in prompts[:n]:
            t = time.perf_counter()
            self.generate([p], batch_size=1)
            times.append((time.perf_counter() - t) * 1000)
        peak = torch.cuda.max_memory_allocated() / 2**20 if self.device == "cuda" else None
        return {
            "latency_ms_p50": float(np.percentile(times, 50)),
            "latency_ms_p95": float(np.percentile(times, 95)),
            "peak_memory_mb": peak,
        }
