"""
FastAPI backend for the Jewellery Analyzer.

Endpoints
---------
POST /analyze   multipart image upload (field ``file``) ->
                {"type", "confidence", "description", "method", "inference_ms"}
                optional ``mode``: auto (default chosen on the validation set),
                generate (GRU decoder) or retrieve (nearest training description)
GET  /health    readiness check + model info

If a production build of the React frontend exists (``frontend/build`` or
$STATIC_DIR) it is served at ``/`` so the whole app runs from a single URL.

Run locally:  uvicorn main:app --reload --port 8000
"""

import asyncio
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from caption_model import JewelleryAnalyzer

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", BASE_DIR / "weights"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", BASE_DIR.parent / "frontend" / "build"))
CAPTION_INDEX = os.getenv("CAPTION_INDEX", str(WEIGHTS_DIR / "caption_index.npz"))
MAX_UPLOAD_MB = 10
Image.MAX_IMAGE_PIXELS = 40_000_000  # reject decompression bombs

analyzer: JewelleryAnalyzer | None = None
_lock = asyncio.Lock()  # one inference at a time keeps CPU usage predictable


@asynccontextmanager
async def lifespan(_app):
    global analyzer
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = time.perf_counter()
    analyzer = JewelleryAnalyzer(WEIGHTS_DIR, device=device,
                                 caption_index=CAPTION_INDEX if Path(CAPTION_INDEX).exists() else None)
    analyzer.analyze(Image.new("RGB", (224, 224), "white"))  # warm-up
    print(f"Models ready on {device} in {time.perf_counter() - t:.1f}s "
          f"(model v{analyzer.version}, default caption mode: {analyzer.default_mode})")
    yield


app = FastAPI(title="Jewellery Analyzer API", lifespan=lifespan,
              description="VGG-16 + GRU jewellery classification and description generation")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    if analyzer is None:
        raise HTTPException(503, "Model is loading")
    return {
        "status": "ok",
        "device": str(analyzer.device),
        "vocab_size": len(analyzer.vocab),
        "model_version": analyzer.version,
        "default_caption_mode": analyzer.default_mode,
        "shared_backbone": analyzer.shared_backbone,
        "caption_index_size": len(analyzer.index) if analyzer.index else 0,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...),
                  mode: str = Query("auto", pattern="^(auto|generate|retrieve)$")):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image must be smaller than {MAX_UPLOAD_MB} MB.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        image = ImageOps.exif_transpose(image)  # phone photos: honour the camera rotation
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "Please upload a valid image file (JPG, PNG or WEBP).")

    if image.mode in ("RGBA", "LA", "P"):  # flatten transparency onto white like the training photos
        rgba = image.convert("RGBA")
        bg = Image.new("RGB", rgba.size, "white")
        bg.paste(rgba, mask=rgba.split()[3])
        image = bg

    async with _lock:
        start = time.perf_counter()
        result = await asyncio.to_thread(analyzer.analyze, image, mode)
        result["inference_ms"] = round((time.perf_counter() - start) * 1000)
    return result


# Backwards-compatible alias for the original Flask route
app.add_api_route("/predict", analyze, methods=["POST"], include_in_schema=False)

# Serve the built React app (mounted last so API routes take priority)
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
