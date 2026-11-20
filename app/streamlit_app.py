"""Streamlit app: hide personal data in your own text with the trained PII guardrail.

Run from the repository root:
    streamlit run app/streamlit_app.py
    streamlit run app/streamlit_app.py -- --checkpoints /path/to/checkpoints

The app only uses the public ``piiguard.inference`` API; it does not train or evaluate anything.
Scores shown in the app are read from ``results/`` (never typed by hand).
"""

from __future__ import annotations

import argparse
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

# ------------------------------------------------------------------ static text
MASKERS = {
    "deberta": ("DeBERTa-v3-small", "Fastest and most accurate. Recommended."),
    "qwen_lora": ("Qwen2.5-0.5B + LoRA", "Small chat model. Slower on a CPU."),
    "smollm_lora": ("SmolLM2-360M + LoRA", "Smallest chat model. Slower on a CPU."),
}
GATES = {
    "none": "Off (check every text)",
    "logreg": "Logistic regression",
    "mlp": "Small neural net (MLP)",
    "vqc": "Quantum circuit (VQC, simulated)",
    "minilm_ft": "Fine-tuned MiniLM",
}
GATE_SUMMARY_ROW = {
    "logreg": ("logreg", "classical", "n2000"),
    "mlp": ("mlp", "classical", "n2000"),
    "vqc": ("vqc", "exact", "n2000"),
    "minilm_ft": ("minilm_finetuned", "raw_text", "full"),
}
MODES = {
    "document": "Whole text (safest)",
    "doc_gate": "Skip text the gate finds clean",
    "sentence": "Flagged sentences only (leaks more)",
}
EXAMPLES = {
    "📅 Appointment": (
        "Dear Mr. Thompson, your appointment is confirmed for 14/03/2025 at 10:30. "
        "Please bring your passport (ID: 548293761) and send any questions to "
        "michael.thompson@outlook.com."
    ),
    "🏥 Patient record": (
        "Patient Sarah Johnson, born 1987-05-22, lives at 17 Elm Road, Springfield, IL 62704. "
        "Contact: +1 217 555 0198, username sjohnson87."
    ),
    "💻 Login log": (
        "Login from IP 192.168.24.17 by user alex_k92 at 08:45. "
        "Driver license DL-4471-9920 was updated."
    ),
    "🕵️ Disguised": "Reach john at gmail dot com, phone five five five 0142.",
    "✅ No PII": "The meeting moved to the second floor. Please bring the quarterly slides.",
}

# Friendly names and a colour group for every label in the dataset.
GROUPS = {
    "person": ("#7c3aed", "Person"),
    "contact": ("#2563eb", "Contact"),
    "place": ("#ea580c", "Place"),
    "id": ("#dc2626", "ID"),
    "time": ("#059669", "Date and time"),
}
LABELS = {
    "GIVENNAME1": ("First name", "person"),
    "GIVENNAME2": ("Middle name", "person"),
    "LASTNAME1": ("Last name", "person"),
    "LASTNAME2": ("Last name", "person"),
    "LASTNAME3": ("Last name", "person"),
    "TITLE": ("Title", "person"),
    "SEX": ("Gender", "person"),
    "EMAIL": ("Email", "contact"),
    "TEL": ("Phone", "contact"),
    "USERNAME": ("Username", "contact"),
    "IP": ("IP address", "contact"),
    "PASS": ("Password", "contact"),
    "STREET": ("Street", "place"),
    "BUILDING": ("House number", "place"),
    "SECADDRESS": ("Address line", "place"),
    "CITY": ("City", "place"),
    "STATE": ("State", "place"),
    "POSTCODE": ("Postcode", "place"),
    "COUNTRY": ("Country", "place"),
    "GEOCOORD": ("Coordinates", "place"),
    "IDCARD": ("ID card", "id"),
    "PASSPORT": ("Passport", "id"),
    "DRIVERLICENSE": ("Driver's license", "id"),
    "SOCIALNUMBER": ("Social security no.", "id"),
    "CARDISSUER": ("Card issuer", "id"),
    "DATE": ("Date", "time"),
    "BOD": ("Date of birth", "time"),
    "TIME": ("Time", "time"),
}


def label_name(label: str) -> str:
    return LABELS.get(label, (label.title(), ""))[0]


def label_color(label: str) -> str:
    return GROUPS.get(LABELS.get(label, ("", "id"))[1], GROUPS["id"])[0]


# ------------------------------------------------------------------ discovery and loading
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
    mdir = root / "masker"
    if not mdir.is_dir():
        return {}
    return {
        d.name: d
        for d in sorted(p for p in mdir.iterdir() if p.is_dir())
        if (d / "config.json").exists() or (d / "adapter_config.json").exists()
    }


def find_gates(root: Path) -> tuple[Path | None, list[str]]:
    d = root / "gate"
    for meta in (d / "shared" / "meta.json", d / "meta.json"):
        if meta.exists():
            names = json.loads(meta.read_text())["gates"]
            order = list(GATES)
            return d, sorted(names, key=lambda g: order.index(g) if g in order else len(order))
    return None, []


def masker_name(folder: str) -> str:
    return MASKERS.get(model_key(folder), (folder, ""))[0]


def is_llm(path: Path) -> bool:
    return (path / "adapter_config.json").exists() or is_causal_lm(path)


@st.cache_resource(show_spinner="Loading the model (first time only)...")
def load_masker(path: str) -> Any:
    return PiiGuardrail.from_dir(path).masker


@st.cache_resource(show_spinner="Loading the gate (first time only)...")
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


def clean_scores(key: str) -> tuple[float, float] | None:
    """(F1, leak) on the clean test set, from results/."""
    df = read_summary("masker/summary.csv")
    if df is None:
        return None
    r = df[(df["model"] == key) & (df["set"] == "clean")]
    return None if r.empty else (r["entity_f1_mean"].iloc[0], r["char_leak_rate_mean"].iloc[0])


# ------------------------------------------------------------------ look and feel
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextArea textarea, button {
  font-family: 'Inter', system-ui, sans-serif !important;
}
.block-container {max-width: 1080px; padding-top: 2.2rem;}
[data-testid="stHeader"], #MainMenu, footer {display: none;}

.hero {text-align: center; padding: 30px 20px 24px; border-radius: 22px; margin-bottom: 18px;
  background: radial-gradient(1200px 300px at 50% -40%, rgba(37,99,235,.18), transparent),
              linear-gradient(180deg, rgba(37,99,235,.06), rgba(124,58,237,.04));
  border: 1px solid rgba(37,99,235,.15);}
.hero .logo {font-size: 46px; line-height: 1;}
.hero h1 {font-size: 2.3rem; font-weight: 800; letter-spacing: -.03em; margin: 8px 0 4px;
  padding: 0; background: linear-gradient(90deg,#1d4ed8,#7c3aed);
  -webkit-background-clip: text; background-clip: text; color: transparent;}
.hero p {font-size: 1.08rem; opacity: .75; margin: 0 0 14px;}
.badge {display: inline-block; padding: 5px 12px; margin: 3px; border-radius: 999px;
  font-size: .82rem; font-weight: 600; background: rgba(255,255,255,.65);
  border: 1px solid rgba(37,99,235,.18);}

.banner {display: flex; align-items: center; gap: 12px; padding: 14px 18px; border-radius: 14px;
  margin: 18px 0 14px; font-size: 1.02rem; font-weight: 600;}
.banner.ok {background: rgba(5,150,105,.09); border: 1px solid rgba(5,150,105,.35); color: #047857;}
.banner.none {background: rgba(100,116,139,.08); border: 1px solid rgba(100,116,139,.3);}
.banner .meta {margin-left: auto; font-weight: 500; font-size: .88rem; opacity: .7;}

.card-title {font-weight: 700; font-size: .95rem; margin-bottom: 8px; opacity: .85;}
.pii-text {line-height: 2.2; font-size: 1.02rem; white-space: pre-wrap;}
.pii {border-radius: 7px; padding: 2px 6px; border: 1.5px solid; font-weight: 500;}
.pii small {font-size: .66em; font-weight: 700; margin-left: 5px; text-transform: uppercase;
  letter-spacing: .04em;}
.safe {line-height: 2.2; font-size: 1.02rem; white-space: pre-wrap;}
.mask {border-radius: 6px; padding: 2px 7px; font-weight: 700; font-size: .85em; color: #fff;}
.chip {display: inline-block; border-radius: 999px; padding: 4px 12px; margin: 0 6px 6px 0;
  font-size: .84rem; font-weight: 600; border: 1.5px solid;}
.foot {text-align: center; opacity: .55; font-size: .82rem; margin-top: 28px;}
</style>
"""


def rgba(hex_color: str, a: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{a})"


def highlight(text: str, spans: list[dict[str, Any]]) -> str:
    """The original text with every detected span highlighted and named."""
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: s["start"]):
        if s["start"] < pos:
            continue
        c = label_color(s["label"])
        out.append(html.escape(text[pos : s["start"]]))
        out.append(
            f"<span class='pii' style='background:{rgba(c, 0.12)};border-color:{c}'>"
            f"{html.escape(text[s['start'] : s['end']])}"
            f"<small style='color:{c}'>{html.escape(label_name(s['label']))}</small></span>"
        )
        pos = s["end"]
    out.append(html.escape(text[pos:]))
    return f"<div class='pii-text'>{''.join(out)}</div>"


def safe_view(masked: str) -> str:
    """The masked text with each [LABEL] drawn as a coloured block."""

    def block(m: re.Match[str]) -> str:
        lab = m.group(1)
        name = html.escape(label_name(lab))
        return f"<span class='mask' style='background:{label_color(lab)}'>{name}</span>"

    parts = re.split(r"(\[[A-Z0-9_]+\])", masked)
    out = [
        block(re.match(r"\[([A-Z0-9_]+)\]", p))
        if re.fullmatch(r"\[[A-Z0-9_]+\]", p)
        else html.escape(p)
        for p in parts
    ]
    return f"<div class='safe'>{''.join(out)}</div>"


def chips(spans: list[dict[str, Any]]) -> str:
    counts = Counter(label_name(s["label"]) for s in spans)
    colors = {label_name(s["label"]): label_color(s["label"]) for s in spans}
    return "".join(
        f"<span class='chip' style='background:{rgba(colors[n], 0.1)};border-color:{colors[n]};"
        f"color:{colors[n]}'>{html.escape(n)} · {k}</span>"
        for n, k in counts.most_common()
    )


# ------------------------------------------------------------------ pages
def hero() -> None:
    sc = clean_scores("deberta")
    badges = ["🔒 Runs on your machine", "🧠 Trained on 28 PII types"]
    if sc:
        badges.append(f"📊 {sc[1]:.1%} leak on the test set")
    st.markdown(
        "<div class='hero'><div class='logo'>🛡️</div><h1>PII Guardrail</h1>"
        "<p>Hide personal data before it reaches an LLM.</p>"
        + "".join(f"<span class='badge'>{b}</span>" for b in badges)
        + "</div>",
        unsafe_allow_html=True,
    )


def set_example() -> None:
    choice = st.session_state.get("example")
    if choice:
        st.session_state["text"] = EXAMPLES[choice]
        st.session_state["autorun"] = True


def clear_text() -> None:
    st.session_state["text"] = ""
    st.session_state.pop("example", None)
    st.session_state.pop("last", None)


def settings(maskers: dict[str, Path], gates: list[str]) -> tuple[str, str, str]:
    with st.popover("⚙️ Settings", width="stretch"):
        masker = st.selectbox("Model", list(maskers), format_func=masker_name)
        st.caption(MASKERS.get(model_key(masker), ("", ""))[1])
        if is_llm(maskers[masker]) and device_name() == "cpu":
            st.caption("🐢 This model takes a few seconds per text on a CPU.")
        gate = st.selectbox(
            "Gate (quick *is there PII?* check)", ["none", *gates], format_func=GATES.get
        )
        options = ["document"] if gate == "none" else list(MODES)
        mode = st.radio("Mode", options, format_func=MODES.get, index=0)
    return masker, gate, mode


def page_try(maskers: dict[str, Path], gates_dir: Path | None, gates: list[str]) -> None:
    st.pills(
        "Try an example",
        list(EXAMPLES),
        selection_mode="single",
        key="example",
        on_change=set_example,
    )
    text = st.text_area(
        "Your text",
        key="text",
        height=130,
        placeholder="Paste an email, a chat message or a note...",
        label_visibility="collapsed",
    )
    c1, c2, c3 = st.columns([2, 1, 1.4])
    run = c1.button(
        "🛡️  Hide personal data", type="primary", width="stretch", disabled=not text.strip()
    )
    c2.button("Clear", on_click=clear_text, width="stretch")
    with c3:
        masker, gate, mode = settings(maskers, gates)

    if (run or st.session_state.pop("autorun", False)) and text.strip():
        try:
            m = load_masker(str(maskers[masker]))
            g = load_gates(str(gates_dir), gate) if gate != "none" else None
            guard = PiiGuardrail(m, g, gate if g is not None else None)
            t0 = time.perf_counter()
            r = guard.mask(text, mode=mode)
            ms = (time.perf_counter() - t0) * 1000
        except Exception as e:  # show the problem instead of a stack trace
            st.error(f"Could not run the guardrail: {e}")
            return
        st.session_state["last"] = {"text": text, "r": r, "ms": ms, "masker": masker}

    last = st.session_state.get("last")
    if not last:
        st.caption("Pick an example above, or paste your own text and press the button.")
        return
    r, src = last["r"], last["text"]
    meta = f"{masker_name(last['masker'])} · {last['ms']:,.0f} ms"
    if r.spans:
        n = len(r.spans)
        st.markdown(
            f"<div class='banner ok'>✅ Hid {n} piece{'s' if n != 1 else ''} of personal data"
            f"<span class='meta'>{meta}</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='banner none'>🔎 No personal data found"
            f"<span class='meta'>{meta}</span></div>",
            unsafe_allow_html=True,
        )
    if src != text:
        st.caption("Showing the result for the previous text. Press the button to update.")

    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        st.markdown("<div class='card-title'>🔍 What we found</div>", unsafe_allow_html=True)
        st.markdown(highlight(src, r.spans), unsafe_allow_html=True)
    with right, st.container(border=True):
        st.markdown("<div class='card-title'>🛡️ Safe to share</div>", unsafe_allow_html=True)
        st.markdown(safe_view(r.masked), unsafe_allow_html=True)
    if r.spans:
        st.markdown(chips(r.spans), unsafe_allow_html=True)
    b1, b2, _ = st.columns([1, 1, 2])
    b1.download_button("⬇️ Download", r.masked, file_name="masked.txt", width="stretch")
    with b2.popover("📋 Copy text", width="stretch"):
        st.code(r.masked, language=None, wrap_lines=True)
    if r.sentences:
        with st.expander("Gate decision per sentence"):
            st.dataframe(
                pd.DataFrame(
                    [{"sentence": s, "PII score": p, "flagged": f} for s, p, f in r.sentences]
                ),
                hide_index=True,
                width="stretch",
            )


def page_results() -> None:
    st.markdown(
        "Measured on the held-out English test set (means over seeds), read from `results/`."
    )
    df = read_summary("masker/summary.csv")
    if df is not None:
        rows = []
        for key, (name, _) in MASKERS.items():
            sub = df[df["model"] == key].set_index("set")
            if "clean" not in sub.index:
                continue
            rows.append(
                {
                    "Model": name,
                    "F1": sub.loc["clean", "entity_f1_mean"],
                    "Leak": sub.loc["clean", "char_leak_rate_mean"] * 100,
                    "Leak (disguised)": sub.loc["obf_all", "char_leak_rate_mean"] * 100
                    if "obf_all" in sub.index
                    else None,
                }
            )
        st.markdown("##### 🛡️ Maskers")
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "F1": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                "Leak": st.column_config.NumberColumn(format="%.1f%%"),
                "Leak (disguised)": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        st.caption("Leak = share of personal data still readable after masking. Lower is better.")
    gdf = read_summary("gate/summary.csv")
    if gdf is not None:
        rows = []
        for key, (m, s, n) in GATE_SUMMARY_ROW.items():
            r = gdf[(gdf["model"] == m) & (gdf["setting"] == s) & (gdf["size"] == n)]
            if not r.empty:
                rows.append(
                    {
                        "Gate": GATES[key],
                        "ROC-AUC": r["roc_auc_mean"].iloc[0],
                        "ms / sentence": r["latency_ms_single_mean"].iloc[0],
                    }
                )
        st.markdown("##### 🚦 Gates")
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "ROC-AUC": st.column_config.ProgressColumn(
                    min_value=0.5, max_value=1, format="%.3f"
                ),
                "ms / sentence": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption(
            "The quantum circuit runs on a simulator. "
            "It matches the classical gates, but doesn't beat them."
        )


def page_how() -> None:
    st.markdown(
        """
| | Step | What happens |
|:-:|---|---|
| 1 | **Gate** *(optional)* | A quick check per sentence: *might this contain personal data?* |
| 2 | **Masker** | Finds the exact pieces (names, emails, IDs...) |
| 3 | **Replace** | Each piece becomes a label such as `[EMAIL]` |

**Colours:** <span style='color:#7c3aed'>■ person</span> ·
<span style='color:#2563eb'>■ contact</span> · <span style='color:#ea580c'>■ place</span> ·
<span style='color:#dc2626'>■ ID</span> · <span style='color:#059669'>■ date and time</span>

**Limits:** the models were trained on synthetic data and can miss personal data, especially when
it's disguised (`jane at example dot com`). Treat the result as a safety net, not a guarantee.
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="PII Guardrail", page_icon="🛡️", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    root = checkpoints_dir()
    maskers = find_maskers(root)
    gates_dir, gates = find_gates(root)

    hero()
    if not maskers:
        st.error(
            f"No trained models found in `{root / 'masker'}`. Download them (see the README) "
            "or start the app with `-- --checkpoints <dir>`."
        )
        st.stop()

    t1, t2, t3 = st.tabs(["🛡️ Try it", "📊 Results", "⚙️ How it works"])
    with t1:
        page_try(maskers, gates_dir, gates)
    with t2:
        page_results()
    with t3:
        page_how()
    st.markdown(
        f"<div class='foot'>Runs on {device_name().upper()} · "
        "your text never leaves this app</div>",
        unsafe_allow_html=True,
    )


main()
