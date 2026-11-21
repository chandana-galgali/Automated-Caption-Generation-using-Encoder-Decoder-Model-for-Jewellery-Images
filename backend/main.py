"""
FastAPI backend for the Jewellery Analyser.

Endpoints
---------
POST /analyze   multipart image upload (field name ``file``) ->
                {"type", "confidence", "description", "inference_ms"}
GET  /health    readiness check

If a production build of the React frontend exists (``frontend/build`` or the
directory in $STATIC_DIR) it is served at ``/``, so the whole app runs from a
single URL.

Run locally:  uvicorn main:app --reload --port 8000
"""

import io
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from caption_model import (
    CLASS_NAMES, EMBED_SIZE, HIDDEN_SIZE,
    DecoderRNN, EncoderCNN, VGG16_GRU_Classifier,
    generate_caption, load_vocab, transform,
)

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", BASE_DIR / "weights"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", BASE_DIR.parent / "frontend" / "build"))
MAX_UPLOAD_MB = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load(module, filename):
    state = torch.load(WEIGHTS_DIR / filename, map_location=device, weights_only=True)
    module.load_state_dict(state)
    return module.to(device).eval()


print(f"Loading models on {device} ...")
vocab = load_vocab(WEIGHTS_DIR / "vocab.pkl")
classifier = _load(VGG16_GRU_Classifier(num_classes=len(CLASS_NAMES)), "classifier.pth")
encoder = _load(EncoderCNN(HIDDEN_SIZE), "encoder.pth")
decoder = _load(DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab)), "decoder.pth")
print(f"Models ready (vocabulary size {len(vocab)}).")

app = FastAPI(title="Jewellery Analyser API",
              description="VGG-16 + GRU jewellery classification and description generation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "vocab_size": len(vocab)}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image must be smaller than {MAX_UPLOAD_MB} MB")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Please upload a valid image file (JPG, PNG or WEBP).")

    start = time.perf_counter()
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(classifier(tensor), dim=1)[0]
    idx = int(probs.argmax())
    description = generate_caption(tensor, encoder, decoder, vocab)
    elapsed_ms = round((time.perf_counter() - start) * 1000)

    return {
        "type": CLASS_NAMES[idx],
        "confidence": round(float(probs[idx]), 4),
        "description": description,
        "inference_ms": elapsed_ms,
    }


# Backwards-compatible alias for the original Flask route
app.add_api_route("/predict", analyze, methods=["POST"], include_in_schema=False)

# Serve the built React app (must be mounted last so API routes take priority)
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
