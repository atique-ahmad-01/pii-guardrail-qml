"""Upper bound: MiniLM fine-tuned end to end on raw sentences (no 8-feature bottleneck)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from piiguard.gate.gate_metrics import gate_metrics
from piiguard.utils import get_device, set_seed


class MiniLMClassifier(torch.nn.Module):
    """Mean-pooled MiniLM encoder with a linear head."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logit per sentence."""
        h = self.enc(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        ).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


def _batches(texts: list[str], tok: Any, bs: int, device: str) -> Any:
    for i in range(0, len(texts), bs):
        enc = tok(
            texts[i : i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        yield {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def predict(
    model: MiniLMClassifier, tok: Any, texts: list[str], device: str, bs: int = 128
) -> np.ndarray:
    """P(contains PII) for each text."""
    model.eval()
    out = []
    for enc in _batches(texts, tok, bs, device):
        out.append(torch.sigmoid(model(**enc).float()).cpu().numpy())
    return np.concatenate(out)


def train_minilm(
    model_name: str,
    train_texts: list[str],
    train_y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[MiniLMClassifier, Any, str, float]:
    """Fine-tune MiniLM with a linear head: ``(model, tokenizer, device, train_seconds)``."""
    set_seed(seed)
    device = get_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = MiniLMClassifier(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    amp = device == "cuda"
    g = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        order = g.permutation(len(train_texts))
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            enc = next(_batches([train_texts[j] for j in idx], tok, len(idx), device))
            y = torch.as_tensor(train_y[idx], dtype=torch.float32, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logit = model(**enc)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    return model, tok, device, time.perf_counter() - t0


def finetune_and_eval(
    model_name: str,
    train_texts: list[str],
    train_y: np.ndarray,
    eval_sets: dict[str, tuple[list[str], np.ndarray]],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    """Fine-tune on ``train_texts`` and score every set in ``eval_sets``; also time inference."""
    model, tok, device, train_s = train_minilm(
        model_name, train_texts, train_y, epochs, batch_size, lr, seed
    )
    results: dict[str, Any] = {}
    for name, (texts, y) in eval_sets.items():
        results[name] = gate_metrics(y, predict(model, tok, texts, device))
    first = next(iter(eval_sets.values()))[0]
    for tx in first[:2]:
        predict(model, tok, [tx], device)
    n_single = min(50, len(first))
    t1 = time.perf_counter()
    for tx in first[:n_single]:
        predict(model, tok, [tx], device)
    single = (time.perf_counter() - t1) / n_single * 1000
    t2 = time.perf_counter()
    predict(model, tok, first, device)
    batch = (time.perf_counter() - t2) / len(first) * 1000
    return {
        "eval": results,
        "params": sum(p.numel() for p in model.parameters()),
        "train_s": train_s,
        "latency_ms_single": single,
        "latency_ms_batch_per_sentence": batch,
        "device": device,
    }
