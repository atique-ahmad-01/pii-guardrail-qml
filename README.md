# pii-guardrail-qml

**Clean F1 hides leaks: stress-testing small-language-model PII maskers, with a quantum-vs-classical benchmark for the PII gate.**

A two-stage PII guardrail for LLM pipelines:

```
text ─► Stage 1 GATE (contains PII?) ─► Stage 2 MASKER (find + redact spans) ─► safe text
         classical vs quantum              encoder vs small instruction LLMs
```

The full task description is in [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md).

**Honesty rule:** every number in `reports/` comes from files in `results/`. Nothing is hand-written.

## Quick start

```bash
make setup      # uv sync --all-extras
make test       # unit tests (offline)
make lint
make smoke      # tiny profile, intended <15 min on a laptop
make reproduce  # full profile: run on a CUDA GPU
```

Device is auto-selected `cuda → mps → cpu`. Configs: `configs/smoke.yaml`, `configs/full.yaml`.

## Status

| Phase | Description | Status |
|---|---|---|
| 0 | Scaffold | done |
| 1 | Data | done (full + smoke built, see `data/processed*/data_card.md`) |
| 2 | Stress suite | done (full + smoke built; context-only examples await owner review) |
| 3 | Gate (quantum vs classical) | done (full run on GPU host: `results/gate/summary*.csv`) |
| 4 | Maskers | pending |
| 5 | Pipeline | pending |
| 6 | Evaluation | pending |
| 7 | Report | pending |
| 8 | Demo | optional |

## Dataset license

Data comes from [`ai4privacy/pii-masking-300k`](https://huggingface.co/datasets/ai4privacy/pii-masking-300k), which carries its own license and terms. Check the dataset card before redistributing any derived data. Processed data is git-ignored here.
