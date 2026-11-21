"""API and model tests. Run from the backend folder:  pytest -q

Works with both the original (v1) weights and weights produced by
training/run_all.py (v2). Set WEIGHTS_DIR to test another weights folder.
"""

import io
import os
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402
from caption_model import CLASS_NAMES, EncoderCNN, transform  # noqa: E402

SAMPLES = ROOT / "test_images"
REAL_WEIGHTS = "WEIGHTS_DIR" not in os.environ  # synthetic smoke weights are random


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


def upload(client, path, **params):
    with open(path, "rb") as f:
        return client.post("/analyze", files={"file": (path.name, f, "image/jpeg")}, params=params)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["vocab_size"] > 4
    assert body["shared_backbone"] is True


@pytest.mark.parametrize("name,expected", [("1.jpeg", "Earring"), ("12.jpg", "Earring"),
                                           ("3.jpg", "Necklace"), ("2.jpeg", "Necklace")])
def test_analyze(client, name, expected):
    r = upload(client, SAMPLES / name)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] in CLASS_NAMES
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["inference_ms"] > 0
    assert "generated_description" in body
    if REAL_WEIGHTS:
        assert body["type"] == expected
        assert len(body["description"].split()) >= 3
        assert body["description"][0].isupper() and body["description"].endswith((".", "!", "?"))


def test_generate_mode(client):
    r = upload(client, SAMPLES / "1.jpeg", mode="generate")
    assert r.status_code == 200
    assert r.json()["method"] == "generated"


def test_rejects_non_images(client):
    r = client.post("/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_transparent_png(client):
    img = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    buf = io.BytesIO(); img.save(buf, "PNG")
    r = client.post("/analyze", files={"file": ("t.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    assert r.json()["type"] in CLASS_NAMES


def _v1(client):
    if main.analyzer.version != 1:
        pytest.skip("only applies to the original v1 weights")
    return main.analyzer


def test_v1_incremental_decoder_matches_original(client):
    """Carrying the GRU state forward must give the same greedy caption as
    re-running the decoder over the whole prefix (the original implementation)."""
    a = _v1(client)
    start, end = a.vocab.stoi["<START>"], a.vocab.stoi["<END>"]
    x = transform(Image.open(SAMPLES / "1.jpeg").convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        h0 = a.encoder.head(a.classifier.features(x))
        caption = [start]
        for _ in range(30):
            nxt = a.decoder(h0, torch.tensor([caption])).argmax(2)[:, -1].item()
            caption.append(nxt)
            if nxt == end:
                break
        hidden, tok, fast = h0.unsqueeze(0), torch.tensor([start]), [start]
        for _ in range(30):
            logits, hidden = a.decoder.step(tok, hidden)
            tok = logits.argmax(-1)
            fast.append(tok.item())
            if tok.item() == end:
                break
    assert fast == caption


def test_v1_shared_backbone_matches_separate_encoder(client):
    a = _v1(client)
    full = EncoderCNN()
    full.load_state_dict(torch.load(main.WEIGHTS_DIR / "encoder.pth", map_location="cpu", weights_only=True))
    full.eval()
    x = transform(Image.open(SAMPLES / "3.jpg").convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        assert torch.allclose(full(x), a.encoder.head(a.classifier.features(x)), atol=1e-5)
