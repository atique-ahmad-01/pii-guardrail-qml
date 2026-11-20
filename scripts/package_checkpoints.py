"""Package the best trained models into self-contained folders for apps.

    python scripts/package_checkpoints.py --src <training checkpoints> --out checkpoints

Output (every folder runs on its own, no downloads):

    checkpoints/
    ├── manifest.json        which run each folder came from, and why it was picked
    ├── masker/
    │   ├── 001_deberta/     DeBERTa-v3-small token classifier (weights + tokenizer)
    │   ├── 002_qwen_lora/   Qwen2.5-0.5B with its LoRA adapter merged in
    │   └── 003_smollm_lora/ SmolLM2-360M with its LoRA adapter merged in
    └── gate/                see piiguard.pipeline.gates.SPLIT_LAYOUT
        ├── shared/          thresholds, PCA scaler, MiniLM encoder
        ├── 001_logreg/  002_mlp/  003_vqc/  004_minilm_ft/

Seeds: one per masker, chosen by the user from the clean test results
(``results/masker/<model>__clean_seed<n>.json``: highest entity F1 and lowest character leak).
Choosing on test data flatters the pick slightly; these folders are for apps, not new results.
The gates come from the single pipeline run (seed 0), so there is nothing to choose.

LoRA adapters are merged in float32 and saved in bfloat16 (the dtype used for GPU inference).
Base models download from Hugging Face if they are not cached.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# (output folder, training run, sub-folder, kind)
MASKERS = [
    ("001_deberta", "deberta_seed2", "best", "deberta"),
    ("002_qwen_lora", "qwen_lora_seed2", "adapter", "lora"),
    ("003_smollm_lora", "smollm_lora_seed2", "adapter", "lora"),
]
WHY = "seed 2: best clean-test entity F1 and char leak of 3 seeds (user's choice)"


def merge_lora(adapter: Path, out: Path) -> str:
    """Merge a LoRA adapter into its base model and save model + tokenizer to ``out``."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from piiguard.masker.llm_sft import load_tokenizer

    base = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    load_tokenizer(base).save_pretrained(out)
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path, help="training-layout checkpoints")
    ap.add_argument("--out", default=ROOT / "checkpoints", type=Path)
    args = ap.parse_args()
    from piiguard.pipeline.gates import Gates

    manifest: dict[str, list[dict[str, str]]] = {"masker": [], "gate": []}
    mdir = args.out / "masker"
    mdir.mkdir(parents=True, exist_ok=True)
    for folder, run, sub, kind in MASKERS:
        src, dst = args.src / run / sub, mdir / folder
        entry = {"folder": f"masker/{folder}", "source": f"{run}/{sub}", "why": WHY}
        if dst.exists():
            print(f"skip   {entry['folder']} (exists)")
        elif not src.exists():
            print(f"miss   {src} (not found)")
            continue
        elif kind == "deberta":
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("training_args.bin"))
        else:
            entry["base_model"] = merge_lora(src, dst)
            entry["merge"] = "float32 merge, saved as bfloat16"
        print(f"done   {entry['folder']}")
        manifest["masker"].append(entry)

    gdir = args.out / "gate"
    if gdir.exists():
        print("skip   gate (exists)")
    else:
        Gates.load(args.src / "gates").save_split(gdir)
        print("done   gate")
    for sub in sorted(p.name for p in gdir.iterdir() if p.is_dir()):
        manifest["gate"].append({"folder": f"gate/{sub}", "source": "gates (pipeline seed 0)"})

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
