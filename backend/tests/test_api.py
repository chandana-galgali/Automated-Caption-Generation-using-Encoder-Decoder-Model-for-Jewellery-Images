"""API and model tests. Run from the backend folder:  pytest -q"""

import io
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
from caption_model import CLASS_NAMES, transform  # noqa: E402

SAMPLES = ROOT / "test_images"


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:  # runs the lifespan -> loads the models
        yield c


def upload(client, path, **params):
    with open(path, "rb") as f:
        return client.post("/analyze", files={"file": (path.name, f, "image/jpeg")}, params=params)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["vocab_size"] == 2158
    assert body["shared_backbone"] is True


@pytest.mark.parametrize("name,expected", [("1.jpeg", "Earring"), ("12.jpg", "Earring"),
                                           ("3.jpg", "Necklace"), ("2.jpeg", "Necklace")])
def test_classification(client, name, expected):
    r = upload(client, SAMPLES / name)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == expected
    assert 0.5 <= body["confidence"] <= 1.0
    assert len(body["description"].split()) >= 3
    assert body["description"][0].isupper() and body["description"].endswith((".", "!", "?"))
    assert body["inference_ms"] > 0


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


def test_optimised_decoder_matches_original(client):
    """Incremental decoding must give exactly the same greedy caption as the
    original implementation that re-ran the GRU over the whole prefix."""
    a = main.analyzer
    x = transform(Image.open(SAMPLES / "1.jpeg").convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        h0 = a.encoder.head(a.classifier.features(x))
        caption = [a.start]  # original algorithm
        for _ in range(30):
            nxt = a.decoder(h0, torch.tensor([caption])).argmax(2)[:, -1].item()
            caption.append(nxt)
            if nxt == a.end:
                break
        hidden, tok, fast = h0.unsqueeze(0), torch.tensor([a.start]), [a.start]
        for _ in range(30):
            logits, hidden = a.decoder.step(tok, hidden)
            tok = logits.argmax(-1)
            fast.append(tok.item())
            if tok.item() == a.end:
                break
    assert fast == caption


def test_shared_backbone_matches_separate_encoder(client):
    """Running VGG-16 once must equal the original separate encoder."""
    from caption_model import EncoderCNN
    a = main.analyzer
    full = EncoderCNN()
    full.load_state_dict(torch.load(BACKEND / "weights" / "encoder.pth", map_location="cpu", weights_only=True))
    full.eval()
    x = transform(Image.open(SAMPLES / "3.jpg").convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        assert torch.allclose(full(x), a.encoder.head(a.classifier.features(x)), atol=1e-5)
