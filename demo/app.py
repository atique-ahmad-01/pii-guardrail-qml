"""Optional Gradio demo of the two-stage guardrail.

Run: ``python demo/app.py --config configs/full.yaml``.

Needs ``pip install gradio`` (extra ``demo``), a trained DeBERTa checkpoint from Phase 4, and the
processed data from Phase 1 (the gate is refit at start-up; this takes about a minute).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from piiguard.masker.token_clf import DebertaMasker
from piiguard.pipeline.demo import run_demo
from piiguard.pipeline.gates import Gates
from piiguard.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--gate", default="logreg", choices=["logreg", "mlp", "vqc", "minilm_ft"])
    args = ap.parse_args()
    import gradio as gr

    cfg = load_config(args.config)
    gates = Gates(cfg, cfg["pipeline"]["seed"]).fit()
    gates.thresholds = {args.gate: gates.thresholds[args.gate]}
    masker = DebertaMasker(Path(cfg["pipeline"]["ckpt"]), cfg["masker"]["deberta"]["max_length"])

    def fn(text: str, use_gate: bool) -> tuple[str, list[list[str]]]:
        r = run_demo(text, gates_view, masker, use_gate)
        rows = [[s, f"{p:.3f}", "masker" if f else "skipped"] for s, p, f in r.sentences]
        return r.masked, rows

    class _View:
        """Gate restricted to the chosen model."""

        thresholds = gates.thresholds

        @staticmethod
        def score(name: str, texts: list[str]):
            return gates.score(name, texts)

    gates_view = _View()
    gr.Interface(
        fn,
        [
            gr.Textbox(lines=8, label="Text"),
            gr.Checkbox(True, label="Use the gate (skip sentences without PII)"),
        ],
        [
            gr.Textbox(label="Masked text"),
            gr.Dataframe(headers=["sentence", "gate score", "decision"]),
        ],
        title="Two-stage PII guardrail (research demo)",
    ).launch()


if __name__ == "__main__":
    main()
