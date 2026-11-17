# pii-guardrail-qml

**Clean F1 hides leaks: stress-testing small-language-model PII maskers, with a quantum-vs-classical benchmark for the PII gate.**

A two-stage PII guardrail for LLM pipelines:

```
text ─► Stage 1 GATE (contains PII?) ─► Stage 2 MASKER (find + redact spans) ─► safe text
         classical vs quantum              encoder vs small instruction LLMs
```

The full task description is in [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md). The results are in
[`reports/report.md`](reports/report.md), generated from files in `results/`.

**Honesty rule:** every number in `reports/` comes from files in `results/`. Nothing is hand-written,
and this README quotes no results; read the report.

## What is measured

| Part | Question | Models |
|---|---|---|
| Stage 1 gate | Does a quantum classifier beat classical ones on identical 8-feature inputs? | logistic regression, MLP, VQC (exact / shots / depolarizing noise), quantum-kernel SVM (angle, IQP), fine-tuned MiniLM as an upper bound |
| Extra quantum scenarios | Low-data learning curves, VQC depth and data re-uploading, number of features / qubits | same features, exact simulation |
| Stage 2 masker | How much PII leaks under stress, and does clean F1 predict it? | DeBERTa-v3-small token classifier; Qwen2.5-0.5B and SmolLM2-360M with LoRA (JSON output); Qwen zero-shot |
| Stress sets | Obfuscated emails / phones / names / IDs, rare PII types, German / French / Italian, context-only identifiers | built from the held-out test split |
| Pipeline | Does the gate save masker work, and what does it cost in leaks and latency? | each gate in front of DeBERTa, plus masker-only baselines and an oracle gate |
| Statistics | Are differences real? | 95% bootstrap CIs over documents, paired bootstrap tests, Holm correction |

Headline privacy metric: **character leak rate**, the share of gold PII characters not covered by any
predicted mask (type ignored). Strict entity F1 is reported next to it.

## Quick start

```bash
make setup      # uv sync --all-extras
make test       # unit tests (offline)
make lint
make smoke      # tiny profile of every stage, a few minutes
make reproduce  # full profile: data -> stress -> gate -> masker -> pipeline -> eval -> report (CUDA GPU)
```

Individual stages: `make data stress gate masker pipeline eval report`. The extra quantum scenarios run
with `python scripts/run_quantum_extras.py --config configs/full.yaml`. Device is auto-selected
`cuda → mps → cpu`. Configs: `configs/smoke.yaml`, `configs/full.yaml`.

Optional demo (`pip install gradio`): `python demo/app.py --config configs/full.yaml`.

## Layout

```
configs/    smoke.yaml, full.yaml        scripts/   one entry point per phase
src/piiguard/data/      loading, BIO, sentences, stress sets, SFT pairs
src/piiguard/gate/      classical, VQC, quantum kernel, MiniLM, metrics
src/piiguard/masker/    DeBERTa, LoRA SFT, LLM inference, span parsing
src/piiguard/pipeline/  gates, assembly, demo core
src/piiguard/eval/      metrics, bootstrap
results/    run records (config, seed, git commit, timings, metrics) and summary CSVs
reports/    report.md and figures
```

## Status

| Phase | Description | Status |
|---|---|---|
| 0 | Scaffold | done |
| 1 | Data | done (see `data/processed*/data_card.md`) |
| 2 | Stress suite | done (context-only examples await the owner's review) |
| 3 | Gate (quantum vs classical) | done |
| 4 | Maskers | done |
| 5 | Pipeline | code done; see report for the run |
| 6 | Evaluation (bootstrap, paired tests) | done |
| 7 | Report | done (`reports/report.md`) |
| 8 | Demo | done (optional, Gradio) |
| extras | Quantum scenarios | done |

## Limitations

See the limitations section of the report. In short: quantum results are classical simulations (no
hardware), the data is synthetic, the 8-feature bottleneck bounds every gate model, and the
LLM span recovery is exact-string matching.

## Dataset license

Data comes from [`ai4privacy/pii-masking-300k`](https://huggingface.co/datasets/ai4privacy/pii-masking-300k), which carries its own license and terms. Check the dataset card before redistributing any derived data. Processed data is git-ignored here.
