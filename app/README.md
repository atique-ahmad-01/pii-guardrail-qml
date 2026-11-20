# PII Guardrail app (Streamlit)

A small, self-contained UI for trying the trained models on your own text. It is separate from the
research code: it does no training or evaluation and only calls `piiguard.inference.PiiGuardrail`.

```mermaid
flowchart LR
    UI[Streamlit app] --> API[PiiGuardrail]
    API --> G[checkpoints/gate/*]
    API --> M[checkpoints/masker/*]
    UI -. scores .-> R[(results/*.csv)]
```

## Run

```bash
# from the repository root, inside the project environment
uv pip install -r app/requirements.txt
streamlit run app/streamlit_app.py           # http://localhost:8501
```

Options: `streamlit run app/streamlit_app.py -- --checkpoints <dir>` or set
`PIIGUARD_CHECKPOINTS=<dir>` to load models from another folder. Add
`--server.address localhost` to keep the app private to this machine.

## What you can do

![demo](../docs/demo.gif)

| Tab | Purpose |
|---|---|
| 🛡️ **Try it** | Pick an example or paste text, press **Hide personal data**. See what was found (coloured by type), the safe text, and download or copy it. |
| 📊 **Results** | Test-set F1 and leak rate for each masker, and ROC-AUC and speed for each gate, read from `results/`. |
| ⚙️ **How it works** | The steps, what the colours mean, and the limits. |

**⚙️ Settings** (next to the button) lets you pick the model (DeBERTa, Qwen + LoRA, SmolLM2 + LoRA), an optional gate (classical, quantum circuit or MiniLM) and the mode. The defaults (DeBERTa, no gate, whole text) are the safest and fastest.

The theme lives in `.streamlit/config.toml` at the repository root.

## Models

The app reads the packaged folders (built with `scripts/package_checkpoints.py`, see the main README):

```
checkpoints/
├── masker/
│   ├── 001_deberta/         DeBERTa-v3-small token classifier
│   ├── 002_qwen_lora/       Qwen2.5-0.5B with its LoRA adapter merged in
│   └── 003_smollm_lora/     SmolLM2-360M with its LoRA adapter merged in
└── gate/
    ├── shared/              thresholds, PCA feature scaler, MiniLM encoder
    ├── 001_logreg/  002_mlp/  003_vqc/  004_minilm_ft/
```

Any folder added under `masker/` (a DeBERTa model, a merged language model or a LoRA adapter) appears
in the list automatically. Each model loads once and stays in memory, so only the first request is slow.

## Notes

- **Privacy:** text is processed in memory on the machine running the app and is not saved. The app has
  no login; do not expose it on a network without authentication and HTTPS.
- **Speed:** DeBERTa needs well under a second per text on a CPU. The language-model maskers take
  several seconds per text on a CPU; a GPU is recommended for them.
- **Limits:** the models were trained on a synthetic dataset and can miss PII, especially when it is
  disguised. Treat the output as a safety net, not a guarantee.
