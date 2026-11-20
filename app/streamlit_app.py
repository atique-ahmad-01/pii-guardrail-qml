"""Streamlit app: try the trained PII guardrail on your own text.

Pick a gate (stage 1) and a masker (stage 2) independently, paste text, and see the masked output,
what was hidden, the gate's decisions and the latency. A second tab compares maskers side by side.

Run from the repository root:
    streamlit run app/streamlit_app.py
    streamlit run app/streamlit_app.py -- --checkpoints /path/to/checkpoints

The app only uses the public ``piiguard.inference`` API; it does not train or evaluate anything.
Measured scores shown in the app are read from ``results/`` (never typed by hand).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))  # works without `pip install -e .`

from piiguard.inference import PiiGuardrail, is_causal_lm  # noqa: E402

# ------------------------------------------------------------------ static descriptions
MASKER_INFO = {
    "deberta": (
        "DeBERTa-v3-small",
        "Token classifier. Fastest, and the lowest leak on clean text. Recommended.",
    ),
    "qwen_lora": (
        "Qwen2.5-0.5B + LoRA",
        "Small chat model fine-tuned to list PII as JSON. Slower; robust on some disguised text.",
    ),
    "smollm_lora": (
        "SmolLM2-360M + LoRA",
        "Smallest chat model, same recipe as Qwen. Shows how model size affects quality.",
    ),
}
GATE_INFO = {
    "none": ("No gate", "Every text goes straight to the masker."),
    "logreg": ("Logistic regression", "Classical model on 8 sentence features. Microseconds."),
    "mlp": ("Small neural net (MLP)", "Classical model on the same 8 features."),
    "vqc": ("Quantum circuit (VQC)", "8-qubit variational circuit, run on a simulator."),
    "minilm_ft": ("Fine-tuned MiniLM", "Reads the whole sentence. The most accurate gate."),
}
# results/ summary rows that describe each packaged model
GATE_SUMMARY_ROW = {
    "logreg": ("logreg", "classical", "n2000"),
    "mlp": ("mlp", "classical", "n2000"),
    "vqc": ("vqc", "exact", "n2000"),
    "minilm_ft": ("minilm_finetuned", "raw_text", "full"),
}
MODES = {
    "document": ("Whole text", "Masker always reads the whole text. Safest."),
    "doc_gate": ("Gate, then whole text", "Skip the masker only if no sentence is flagged."),
    "sentence": ("Flagged sentences only", "Fastest, but measured to leak much more."),
}
EXAMPLES = {
    "Contact details": (
        "Hi, I'm Jane Miller. Mail me at jane.miller@example.com or call 555-014-2231. "
        "My flat is at 42 Baker Street, London."
    ),
    "Disguised PII": "Reach john at gmail dot com, phone five five five 0142, id 12-34-56.",
    "Account data": (
        "Customer ID 88-213-907 asked to update the IBAN DE89 3704 0044 0532 0130 00. "
        "Date of birth: 14/03/1987."
    ),
    "No PII": "The meeting moved to the second floor. Please bring the quarterly slides.",
}


# ------------------------------------------------------------------ discovery
def checkpoints_dir() -> Path:
    """From ``-- --checkpoints``, then $PIIGUARD_CHECKPOINTS, then ``<repo>/checkpoints``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default=None)
    args, _ = ap.parse_known_args()
    return Path(args.checkpoints or os.environ.get("PIIGUARD_CHECKPOINTS") or REPO / "checkpoints")


def model_key(folder: str) -> str:
    """``002_qwen_lora`` / ``qwen_lora_seed2`` -> ``qwen_lora``."""
    return re.sub(r"_seed\d+$", "", re.sub(r"^\d+_", "", folder))


def find_maskers(root: Path) -> dict[str, Path]:
    """Masker folders in ``checkpoints/masker/`` (DeBERTa, merged LLM or LoRA adapter)."""
    mdir = root / "masker"
    if not mdir.is_dir():
        return {}
    return {
        d.name: d
        for d in sorted(p for p in mdir.iterdir() if p.is_dir())
        if (d / "config.json").exists() or (d / "adapter_config.json").exists()
    }


def find_gates(root: Path) -> tuple[Path | None, list[str]]:
    """``checkpoints/gate/`` and the gate names it holds."""
    d = root / "gate"
    for meta in (d / "shared" / "meta.json", d / "meta.json"):
        if meta.exists():
            names = json.loads(meta.read_text())["gates"]
            order = list(GATE_INFO)
            return d, sorted(names, key=lambda g: order.index(g) if g in order else len(order))
    return None, []


def masker_label(folder: str) -> str:
    return MASKER_INFO.get(model_key(folder), (folder, ""))[0]


def is_llm(path: Path) -> bool:
    return (path / "adapter_config.json").exists() or is_causal_lm(path)


# ------------------------------------------------------------------ cached loading
@st.cache_resource(show_spinner="Loading masker (first time only)...")
def load_masker(path: str) -> Any:
    return PiiGuardrail.from_dir(path).masker


@st.cache_resource(show_spinner="Loading gate (first time only)...")
def load_gates(gates_dir: str, name: str) -> Any:
    from piiguard.pipeline.gates import Gates

    return Gates.load(gates_dir, only=[name])


@st.cache_resource
def device_name() -> str:
    from piiguard.utils import get_device

    return get_device()


@st.cache_data
def read_summary(name: str) -> pd.DataFrame | None:
    p = REPO / "results" / name
    return pd.read_csv(p) if p.exists() else None


def masker_scores(key: str) -> dict[str, float] | None:
    df = read_summary("masker/summary.csv")
    if df is None:
        return None
    rows = df[df["model"] == key].set_index("set")
    if "clean" not in rows.index:
        return None
    out = {
        "f1": rows.loc["clean", "entity_f1_mean"],
        "leak": rows.loc["clean", "char_leak_rate_mean"],
        "seeds": int(rows.loc["clean", "n_seeds"]),
    }
    if "obf_all" in rows.index:
        out["f1_obf"] = rows.loc["obf_all", "entity_f1_mean"]
        out["leak_obf"] = rows.loc["obf_all", "char_leak_rate_mean"]
    return out


def gate_scores(key: str) -> dict[str, float] | None:
    df = read_summary("gate/summary.csv")
    if df is None or key not in GATE_SUMMARY_ROW:
        return None
    m, s, n = GATE_SUMMARY_ROW[key]
    r = df[(df["model"] == m) & (df["setting"] == s) & (df["size"] == n)]
    if r.empty:
        return None
    r = r.iloc[0]
    return {"auc": r["roc_auc_mean"], "ms": r["latency_ms_single_mean"], "seeds": int(r["n_seeds"])}


# ------------------------------------------------------------------ rendering helpers
CSS = """
<style>
.pii-text {line-height: 2.1; white-space: pre-wrap; font-size: 1rem;}
.pii-span {border-radius: 6px; padding: 1px 4px; border: 1px solid; }
.pii-tag {font-size: 0.68em; font-weight: 600; margin-left: 4px; opacity: 0.85;
          letter-spacing: 0.02em;}
.chip {display: inline-block; border-radius: 999px; padding: 2px 10px; margin: 0 6px 6px 0;
       font-size: 0.85em; border: 1px solid;}
.flow {display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 4px 0 8px;}
.flow .step {border: 1px solid rgba(128,128,128,0.35); border-radius: 10px; padding: 6px 12px;
             font-size: 0.9em;}
.flow .step b {display: block; font-size: 0.72em; opacity: 0.65; font-weight: 600;
               text-transform: uppercase; letter-spacing: 0.05em;}
.flow .arrow {opacity: 0.5;}
</style>
"""


# Distinct, theme-neutral hues (translucent fill + solid border work in light and dark mode).
PALETTE = [
    (59, 130, 246),
    (234, 88, 12),
    (22, 163, 74),
    (219, 39, 119),
    (147, 51, 234),
    (202, 138, 4),
    (8, 145, 178),
    (220, 38, 38),
    (101, 163, 13),
    (79, 70, 229),
]


def label_color(label: str) -> tuple[str, str]:
    """A stable colour per PII label: (fill, border/text)."""
    r, g, b = PALETTE[int(hashlib.md5(label.encode()).hexdigest(), 16) % len(PALETTE)]
    return f"rgba({r},{g},{b},0.20)", f"rgba({r},{g},{b},0.95)"


def highlight(text: str, spans: list[dict[str, Any]]) -> str:
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: s["start"]):
        if s["start"] < pos:
            continue
        label = str(s.get("label", "PII"))
        bg, fg = label_color(label)
        out.append(html.escape(text[pos : s["start"]]))
        out.append(
            f"<span class='pii-span' style='background:{bg};border-color:{fg}'>"
            f"{html.escape(text[s['start'] : s['end']])}"
            f"<span class='pii-tag' style='color:{fg}'>{html.escape(label)}</span></span>"
        )
        pos = s["end"]
    out.append(html.escape(text[pos:]))
    return f"<div class='pii-text'>{''.join(out)}</div>"


def label_chips(spans: list[dict[str, Any]]) -> str:
    counts = Counter(str(s.get("label", "PII")) for s in spans)
    chips = []
    for label, n in counts.most_common():
        bg, fg = label_color(label)
        chips.append(
            f"<span class='chip' style='background:{bg};border-color:{fg}'>"
            f"{html.escape(label)} &times; {n}</span>"
        )
    return "".join(chips)


def flow_strip(gate: str, masker: str, mode: str) -> str:
    def step(title: str, body: str) -> str:
        return f"<div class='step'><b>{title}</b>{html.escape(body)}</div>"

    parts = [step("Input", "Your text")]
    if gate != "none":
        parts.append(step("1 · Gate", GATE_INFO[gate][0]))
    parts.append(step("2 · Masker" if gate != "none" else "Masker", masker_label(masker)))
    parts.append(step("Output", "Safe text"))
    arrow = "<span class='arrow'>→</span>"
    mode_txt = f"<span style='opacity:.6;font-size:.85em'>mode: {MODES[mode][0]}</span>"
    return f"<div class='flow'>{arrow.join(parts)}&nbsp;{mode_txt}</div>"


def set_example() -> None:
    choice = st.session_state.get("example")
    if choice:
        st.session_state["text"] = EXAMPLES[choice]


def clear_text() -> None:
    st.session_state["text"] = ""
    st.session_state.pop("example", None)
    st.session_state.pop("last", None)


# ------------------------------------------------------------------ app
def sidebar(maskers: dict[str, Path], gates: list[str], root: Path) -> tuple[str, str, str]:
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        st.markdown("**Masker** · finds and hides PII")
        masker = st.selectbox(
            "Masker", list(maskers), format_func=masker_label, label_visibility="collapsed"
        )
        key = model_key(masker)
        st.caption(MASKER_INFO.get(key, ("", ""))[1])
        sc = masker_scores(key)
        if sc:
            c1, c2 = st.columns(2)
            c1.metric("F1 (clean)", f"{sc['f1']:.3f}")
            c2.metric("Leak (clean)", f"{sc['leak']:.1%}")
        if is_llm(maskers[masker]) and device_name() == "cpu":
            st.warning("Language models are slow on a CPU (several seconds per text).", icon="🐢")

        st.markdown("**Gate** · quick *is there PII?* check")
        gate = st.selectbox(
            "Gate",
            ["none", *gates],
            format_func=lambda g: GATE_INFO.get(g, (g, ""))[0],
            label_visibility="collapsed",
        )
        st.caption(GATE_INFO.get(gate, ("", ""))[1])
        gs = gate_scores(gate)
        if gs:
            c1, c2 = st.columns(2)
            c1.metric("ROC-AUC", f"{gs['auc']:.3f}")
            c2.metric("ms / sentence", f"{gs['ms']:.2f}")

        st.markdown("**Mode** · how the two stages combine")
        options = ["document"] if gate == "none" else list(MODES)
        mode = st.radio(
            "Mode",
            options,
            index=1 if len(options) > 1 else 0,
            format_func=lambda m: MODES[m][0],
            captions=[MODES[m][1] for m in options],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption(
            f"🖥️ Running on **{device_name().upper()}** · models from `{root.name}/`  \n"
            "🔒 Text is processed locally and never stored or sent anywhere."
        )
        if sc or gs:
            st.caption("Scores: test-set means over seeds, from `results/`.")
    return masker, gate, mode


def tab_mask(maskers: dict[str, Path], gates_dir: Path | None, masker: str, gate: str, mode: str):
    st.pills(
        "Try an example",
        list(EXAMPLES),
        selection_mode="single",
        key="example",
        on_change=set_example,
    )
    text = st.text_area(
        "Text to protect",
        key="text",
        height=170,
        placeholder="Paste an email, a chat message or a document...",
    )
    n_chars = len(text)
    c1, c2, c3 = st.columns([1, 1, 4])
    run = c1.button("🛡️ Mask", type="primary", width="stretch", disabled=not text.strip())
    c2.button("Clear", on_click=clear_text, width="stretch")
    c3.caption(f"{n_chars:,} characters")

    if run:
        try:
            m = load_masker(str(maskers[masker]))
            g = load_gates(str(gates_dir), gate) if gate != "none" else None
            guard = PiiGuardrail(m, g, gate if g is not None else None)
            with st.spinner("Masking..."):
                t0 = time.perf_counter()
                r = guard.mask(text, mode=mode)
                ms = (time.perf_counter() - t0) * 1000
        except Exception as e:  # show the problem instead of a stack trace
            st.error(f"Could not run the guardrail: {e}")
            return
        thr = g.thresholds[gate] if g is not None else None
        st.session_state["last"] = {
            "text": text,
            "r": r,
            "ms": ms,
            "masker": masker,
            "gate": gate,
            "thr": thr,
        }

    last = st.session_state.get("last")
    if not last:
        st.info("Choose models on the left, enter text (or pick an example) and press **Mask**.")
        return
    r, src = last["r"], last["text"]
    if src != text:
        st.caption("Showing the result for the previous text. Press **Mask** to update.")

    st.markdown("---")
    k = st.columns(4)
    k[0].metric("Latency", f"{last['ms']:,.0f} ms")
    k[1].metric("PII found", len(r.spans))
    k[2].metric("Masker ran", "Yes" if r.masker_ran else "Skipped")
    if r.sentences:
        n_flag = sum(f for _, _, f in r.sentences)
        k[3].metric("Sentences flagged", f"{n_flag} / {len(r.sentences)}")
    else:
        k[3].metric("Gate", "Off")

    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.markdown("##### 🔍 Detected PII")
        if r.spans:
            st.markdown(highlight(src, r.spans), unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='pii-text'>{html.escape(src)}</div>", unsafe_allow_html=True)
            st.caption("No PII detected.")
    with right, st.container(border=True):
        st.markdown("##### ✅ Masked output")
        st.code(r.masked, language=None, wrap_lines=True)
        st.download_button(
            "Download masked text", r.masked, file_name="masked.txt", mime="text/plain"
        )

    if r.spans:
        st.markdown(label_chips(r.spans), unsafe_allow_html=True)
        with st.expander(f"All {len(r.spans)} spans"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "type": s.get("label"),
                            "text": src[s["start"] : s["end"]],
                            "start": s["start"],
                            "end": s["end"],
                        }
                        for s in sorted(r.spans, key=lambda s: s["start"])
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
    if r.sentences:
        with st.expander("Gate decision per sentence", expanded=not r.masker_ran):
            st.caption(
                f"{GATE_INFO[last['gate']][0]} · a sentence is flagged when its score ≥ "
                f"threshold {last['thr']:.3f} (set low on purpose, to miss little PII)"
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {"sentence": s, "score": p, "decision": "🚩 flagged" if f else "clean"}
                        for s, p, f in r.sentences
                    ]
                ),
                column_config={
                    "score": st.column_config.ProgressColumn(
                        "PII score", min_value=0.0, max_value=1.0, format="%.3f"
                    )
                },
                hide_index=True,
                width="stretch",
            )


def tab_compare(maskers: dict[str, Path]) -> None:
    st.markdown(
        "Run several maskers on the **same text** in *whole text* mode and compare what each one "
        "hides. Uses the text from the **Mask text** tab."
    )
    default = [m for m in maskers if not is_llm(maskers[m])][:1] or list(maskers)[:1]
    chosen = st.multiselect("Maskers", list(maskers), default=default, format_func=masker_label)
    text = st.session_state.get("text", "")
    if any(is_llm(maskers[m]) for m in chosen) and device_name() == "cpu":
        st.caption("🐢 Language-model maskers take several seconds each on a CPU.")
    if not st.button("Compare", type="primary", disabled=not (chosen and text.strip())):
        if not text.strip():
            st.info("Enter some text in the **Mask text** tab first.")
        return
    rows, results = [], []
    for name in chosen:
        try:
            with st.spinner(f"Running {masker_label(name)}..."):
                guard = PiiGuardrail(load_masker(str(maskers[name])))
                t0 = time.perf_counter()
                r = guard.mask(text)
                ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            st.error(f"{masker_label(name)}: {e}")
            continue
        results.append((name, r, ms))
        rows.append({"masker": masker_label(name), "PII found": len(r.spans), "latency ms": ms})
    if not results:
        return
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        column_config={"latency ms": st.column_config.NumberColumn(format="%.0f")},
    )
    for col, (name, r, ms) in zip(st.columns(len(results)), results, strict=True):
        with col, st.container(border=True):
            st.markdown(f"**{masker_label(name)}** · {len(r.spans)} spans · {ms:,.0f} ms")
            st.markdown(highlight(text, r.spans), unsafe_allow_html=True)
            st.code(r.masked, language=None, wrap_lines=True)


def tab_scores() -> None:
    st.markdown(
        "Measured on the held-out test sets during the research runs (means over seeds). "
        "Read live from `results/`; see `reports/report.md` for confidence intervals."
    )
    df = read_summary("masker/summary.csv")
    if df is not None:
        st.markdown("##### Maskers")
        sets = {
            "clean": "clean",
            "obf_all": "disguised",
            "rare": "rare types",
            "cross_de": "German",
        }
        rows = []
        for key, (name, _) in MASKER_INFO.items():
            sub = df[df["model"] == key].set_index("set")
            if sub.empty:
                continue
            row: dict[str, Any] = {"masker": name, "seeds": int(sub["n_seeds"].iloc[0])}
            for s, lab in sets.items():
                if s in sub.index:
                    row[f"F1 {lab}"] = sub.loc[s, "entity_f1_mean"]
                    row[f"leak {lab}"] = sub.loc[s, "char_leak_rate_mean"] * 100
            rows.append(row)
        fmt = {c: st.column_config.NumberColumn(format="%.3f") for c in rows[0] if "F1" in c}
        fmt |= {c: st.column_config.NumberColumn(format="%.1f%%") for c in rows[0] if "leak" in c}
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config=fmt)
        st.caption("Leak = share of PII characters still readable after masking (lower is better).")
    gdf = read_summary("gate/summary.csv")
    if gdf is not None:
        st.markdown("##### Gates")
        rows = []
        for key, (m, s, n) in GATE_SUMMARY_ROW.items():
            r = gdf[(gdf["model"] == m) & (gdf["setting"] == s) & (gdf["size"] == n)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append(
                {
                    "gate": GATE_INFO[key][0],
                    "ROC-AUC": r["roc_auc_mean"],
                    "F1": r["f1_mean"],
                    "ms / sentence": r["latency_ms_single_mean"],
                    "parameters": int(r["params_mean"]),
                    "seeds": int(r["n_seeds"]),
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
                "F1": st.column_config.NumberColumn(format="%.3f"),
                "ms / sentence": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption("Quantum (VQC) latency is simulator time, not quantum-hardware time.")


def tab_about() -> None:
    st.markdown(
        """
**Two stages.** The **gate** is a quick yes/no check per sentence: *might this contain personal
data?* The **masker** then finds the exact pieces (names, emails, IDs, ...) and replaces each with a
label such as `[EMAIL]`.

**Modes**

| Mode | What the masker sees | When to use |
|---|---|---|
| Whole text | everything, always | safest; the default |
| Gate, then whole text | everything, unless no sentence is flagged | saves time on clean text |
| Flagged sentences only | only sentences the gate flags | fastest, but measured to leak much more |

**Scores**
- **F1**: did the model find exactly the right pieces?
- **Leak**: how much private text is still readable after masking (the privacy number).

**Limits.** The models were trained on a synthetic dataset and can miss PII, especially when it is
disguised (`jane at example dot com`). Treat the output as a safety net, not a guarantee.
"""
    )


def main() -> None:
    st.set_page_config(page_title="PII Guardrail", page_icon="🛡️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    root = checkpoints_dir()
    maskers = find_maskers(root)
    gates_dir, gates = find_gates(root)

    st.title("🛡️ PII Guardrail")
    st.caption("Find and hide personal data in text before it reaches an AI system.")
    if not maskers:
        st.error(
            f"No trained maskers found in `{root / 'masker'}`. Build the folders with "
            "`scripts/package_checkpoints.py` (see app/README.md), or start the app with "
            "`-- --checkpoints <dir>`."
        )
        st.stop()

    masker, gate, mode = sidebar(maskers, gates, root)
    st.markdown(flow_strip(gate, masker, mode), unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(
        ["🛡️ Mask text", "⚖️ Compare maskers", "📊 Model scores", "ℹ️ How it works"]
    )
    with t1:
        tab_mask(maskers, gates_dir, masker, gate, mode)
    with t2:
        tab_compare(maskers)
    with t3:
        tab_scores()
    with t4:
        tab_about()


main()
