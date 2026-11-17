"""Fit the four gates and save them for inference.

Usage: python scripts/export_gates.py --config configs/full.yaml --out checkpoints/gates
Takes about two minutes on a GPU (the VQC trains on a CPU simulator).
"""

from __future__ import annotations

import argparse

from piiguard.pipeline.gates import Gates
from piiguard.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="checkpoints/gates")
    args = ap.parse_args()
    cfg = load_config(args.config)
    Gates(cfg, cfg["pipeline"]["seed"]).fit().save(args.out)


if __name__ == "__main__":
    main()
