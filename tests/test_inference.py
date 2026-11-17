import json
import re
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from piiguard.inference import ChunkedEncoderMasker, PiiGuardrail
from piiguard.serve import make_server

PII = re.compile(r"\S+@\S+|\d{3}-\d{4}")


class FakeMasker:
    """Hides emails and phone-like numbers; records what it was shown."""

    def __init__(self):
        self.seen = []

    def predict(self, texts):
        self.seen += list(texts)
        return [
            [
                {"start": m.start(), "end": m.end(), "label": "PII", "value": m.group()}
                for m in PII.finditer(t)
            ]
            for t in texts
        ]


class FakeGates:
    models = {"g": object()}
    thresholds = {"g": 0.5}

    def score(self, name, texts):
        return np.array([0.9 if PII.search(t) else 0.1 for t in texts])


def guard(with_gate=True):
    m = FakeMasker()
    return PiiGuardrail(m, FakeGates() if with_gate else None, "g" if with_gate else None), m


TEXT = "Hello there. Call 555-0142 now. Bye."


def test_document_mode_masks_the_whole_text_in_one_call():
    g, m = guard(with_gate=False)
    r = g.mask(TEXT)
    assert r.masked == "Hello there. Call [PII] now. Bye." and r.masker_ran
    assert m.seen == [TEXT]


def test_doc_gate_skips_the_masker_only_when_nothing_is_flagged():
    g, m = guard()
    clean = g.mask("Nice weather today. See you soon.", "doc_gate")
    assert (
        not clean.masker_ran
        and m.seen == []
        and clean.masked == "Nice weather today. See you soon."
    )
    r = g.mask(TEXT, "doc_gate")
    assert r.masker_ran and m.seen == [TEXT] and "555-0142" not in r.masked  # whole document, once


def test_sentence_mode_only_shows_flagged_sentences_and_keeps_offsets():
    g, m = guard()
    r = g.mask(TEXT, "sentence")
    assert m.seen == ["Call 555-0142 now."]
    assert r.masked == "Hello there. Call [PII] now. Bye."
    assert [f for _, _, f in r.sentences] == [False, True, False]


def test_errors_and_edge_cases():
    g, _ = guard(with_gate=False)
    with pytest.raises(ValueError):
        g.mask(TEXT, "doc_gate")  # no gate loaded
    with pytest.raises(ValueError):
        g.mask(TEXT, "nonsense")
    assert g.mask("   ").masked == "   " and not g.mask("").masker_ran


class FakeEncoder:
    """Whitespace 'tokenizer' and a masker that hides emails; counts calls."""

    def __init__(self):
        self.calls = 0

    def tok(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def predict(self, texts):
        self.calls += len(texts)
        return [
            [
                {"start": m.start(), "end": m.end(), "label": "EMAIL", "value": m.group()}
                for m in re.finditer(r"\S+@\S+", t)
            ]
            for t in texts
        ]


def test_long_text_is_chunked_and_late_pii_is_still_masked():
    enc = FakeEncoder()
    wrapped = ChunkedEncoderMasker(enc, max_tokens=40)  # budget = 32 tokens
    sentences = [f"This is filler sentence number {i} here." for i in range(20)]
    text = " ".join(sentences) + " Write to bob@site.com today."
    assert len(text.split()) > 100
    spans = wrapped.predict([text])[0]
    assert enc.calls > 1
    assert [text[s["start"] : s["end"]] for s in spans] == ["bob@site.com"]  # offsets shifted back


def test_short_text_is_not_chunked():
    enc = FakeEncoder()
    ChunkedEncoderMasker(enc, max_tokens=512).predict(["mail a@b.com"])
    assert enc.calls == 1


def _post(url, body, raw=False):
    data = body if raw else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_service_round_trip():
    g, _ = guard()
    srv = make_server(g, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/health") as r:
            assert json.loads(r.read())["status"] == "ok"
        code, out = _post(base + "/mask", {"text": TEXT})
        assert code == 200 and out["masked"] == "Hello there. Call [PII] now. Bye."
        assert _post(base + "/mask", {"text": TEXT, "mode": "sentence"})[0] == 200
        assert _post(base + "/mask", {"text": 5})[0] == 400
        assert _post(base + "/mask", {"text": "x", "mode": "bad"})[0] == 400
        assert _post(base + "/mask", b"not json", raw=True)[0] == 400
        assert _post(base + "/other", {"text": "x"})[0] == 404
    finally:
        srv.shutdown()
        srv.server_close()


def test_gates_save_and_load_round_trip(tmp_path):
    from piiguard.gate import classical
    from piiguard.gate.features import AngleScaler
    from piiguard.gate.vqc import VQC
    from piiguard.pipeline.gates import Gates

    rng = np.random.default_rng(0)
    emb = rng.normal(size=(80, 16)).astype(np.float32)
    y = (emb[:, 0] > 0).astype(int)
    cfg = {
        "gate": {"embed_model": "dummy", "embed_batch_size": 8, "n_features": 8},
        "pipeline": {"seed": 0, "target_recall": 0.99},
    }
    g = Gates(cfg, 0)
    g.scaler = AngleScaler(8, 0).fit(emb)
    X = g.scaler.transform(emb)
    g.models["logreg"] = classical.make_model("logreg", 0).fit(X, y)
    g.models["mlp"] = classical.make_model("mlp", 0, hidden=7, max_iter=50).fit(X, y)
    g.models["vqc"] = VQC(8, 1, seed=0)
    g.models["vqc"].fit(X[:16], y[:16], epochs=1, batch_size=8)
    g.thresholds = {"logreg": 0.3, "mlp": 0.4, "vqc": 0.5}
    g.info = {"logreg": {}, "mlp": {}, "vqc": {}}
    g.save(tmp_path)
    h = Gates.load(tmp_path)
    assert h.thresholds == g.thresholds and sorted(h.models) == ["logreg", "mlp", "vqc"]
    assert np.allclose(h.scaler.transform(emb), X)
    for k in ("logreg", "mlp"):
        assert np.allclose(h.models[k].predict_proba(X), g.models[k].predict_proba(X))
    assert np.allclose(h.models["vqc"].predict_proba(X), g.models["vqc"].predict_proba(X))
    assert sorted(Gates.load(tmp_path, only=["logreg"]).models) == ["logreg"]
