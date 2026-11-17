"""Score a saved masker on the test sets without retraining.

Examples:
  python scripts/eval_checkpoint.py --masker checkpoints/deberta_seed0/best \
      --sets clean obf_all --n 200
  python scripts/eval_checkpoint.py --masker checkpoints/qwen_lora_seed0/adapter --sets rare --n 100

Needs the processed data (``make data stress``). Prints entity F1 and character leak rate per set.
"""

from __future__ import annotations

import argparse
import time

from piiguard.data.eval_sets import load_sets
from piiguard.eval.metrics import counts_frame, summarize_frame
from piiguard.inference import PiiGuardrail
from piiguard.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--masker", required=True, help="DeBERTa folder (.../best) or LoRA adapter folder"
    )
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--sets", nargs="+", default=["clean"])
    ap.add_argument("--n", type=int, default=200, help="documents per set (first N)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    guard = PiiGuardrail.from_dir(args.masker)
    sets = load_sets(cfg["paths"]["data_dir"], args.sets, args.n)
    print(f"{'set':<14}{'docs':>6}{'entity F1':>11}{'char leak %':>13}{'ms/doc':>9}")
    for name, s in sets.items():
        texts = [d["text"] for d in s["docs"]]
        t0 = time.perf_counter()
        preds = [guard.mask(t).spans for t in texts]
        ms = (time.perf_counter() - t0) / len(texts) * 1000
        m = summarize_frame(
            counts_frame(texts, [d["spans"] for d in s["docs"]], preds, s["eval_labels"])
        )
        f1, leak = m["entity_f1"], 100 * m["char_leak_rate"]
        print(f"{name:<14}{len(texts):>6}{f1:>11.3f}{leak:>13.2f}{ms:>9.1f}")


if __name__ == "__main__":
    main()
