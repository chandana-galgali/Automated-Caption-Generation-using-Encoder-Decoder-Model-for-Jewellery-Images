# 💎 Automated Description Generation for Jewellery Images using Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-VGG--16%20%2B%20GRU-ee4c2c)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![React](https://img.shields.io/badge/Frontend-React-61dafb)
![CI](https://github.com/chandana-galgali/Automated-Caption-Generation-using-Encoder-Decoder-Model-for-Jewellery-Images/actions/workflows/ci.yml/badge.svg)

**Live demo:** _add your Hugging Face Space link here once deployed (see [Deployment](#-deployment))_

An end-to-end computer-vision and NLP application. Upload a jewellery photo and it **classifies** the item (earring or necklace) and returns a **natural-language description** of it, covering metal, stones, shape and style. The system is built on a VGG-16 image encoder with GRU recurrent networks. It is served by a FastAPI backend with a React frontend.

B.Tech (Information Technology) final-year project, K. J. Somaiya College of Engineering, Somaiya Vidyavihar University (2025). The report, papers, poster and all presentations are in [`Project Documents/`](Project%20Documents).

## ✨ Results

**From the project report** (validation split of the augmented dataset):

| Metric | Value |
|---|---|
| Dataset size (after augmentation) | **32,190 images** |
| Jewellery type classification accuracy | **94.62%** |
| Caption model next-word accuracy (teacher-forced) | **90.44%** |

**On 53 unseen web images** ([`dataset/test_set`](dataset/test_set), run with `python training/evaluate.py`):

| Metric | Value |
|---|---|
| Classification accuracy | **96.2%** (51/53) |
| Description BLEU-4: feature-matched catalogue description (default) | **0.045** |
| Description BLEU-4: free GRU generation | 0.011 |
| Description names the correct jewellery type: feature-matched / GRU generation | 94% / 13% |

The GRU decoder reaches high next-word accuracy on the validation split, but it generalises poorly to new photos. One reason is that the original notebooks split augmented copies of the same photo across train and validation. So by default, the app describes an image with the reference description of the most similar training photo of the predicted type, matched on the same VGG-16 encoder features. The GRU's own caption is still returned (`generated_description`) and shown in the UI, and `?mode=generate` makes it the main output. See [Limitations](#-limitations-and-future-work).

## 🏗️ Architecture

![Architecture](Project%20Documents/System%20Workflow%20Diagram/Architecture.jpg)

1. **Shared encoder.** The image is resized to 224×224, ImageNet-normalised and passed **once** through a frozen VGG-16, giving a 7×7×512 feature map.
2. **Classifier (VGG-16 → GRU).** The 49 spatial feature vectors are read as a sequence by a GRU (hidden 256), and a linear layer predicts *Earring* or *Necklace*.
3. **Caption decoder (VGG-16 → GRU).** Average-pooled features are projected to 512-d and initialise a GRU language model (embedding 256, hidden 512, vocabulary 2,158). It writes a description using beam search (k = 3, with no repeated 3-grams).
4. **Description matching.** The pooled features are compared by cosine similarity with those of 5,180 original training photos of the predicted type, and the closest item's description is returned.

### Optimisations in this version

- **One VGG-16 pass instead of two.** The classifier and the caption encoder were trained with the same frozen ImageNet backbone. All 26 convolution tensors are identical, so the backbone runs once and feeds both heads. This roughly halves CPU time and saves about 60 MB of RAM. A unit test checks that the outputs match the original encoder.
- **Incremental decoding.** The GRU hidden state is carried forward rather than re-running the decoder over the whole prefix for every new word, so decoding cost grows with caption length instead of its square. A unit test checks that it produces the original greedy caption exactly.
- **Batched beam search** with blocking of special tokens and repeated n-grams.
- **No ImageNet download at start-up**, because the backbone weights are already inside the saved checkpoints. The models are loaded and warmed up once, when the server starts.
- **Frontend:** images are downscaled in the browser before upload, and transparent PNGs are flattened onto white.

## 📂 Repository structure

```
backend/                FastAPI app + model code
  main.py               API: POST /analyze, GET /health; also serves the built frontend
  caption_model.py      Model definitions + optimised inference pipeline
  weights/              classifier.pth, encoder.pth, decoder.pth, vocab.pkl, caption_index.npz
  tests/                pytest suite (API, accuracy checks, optimisation equivalence)
frontend/               React + Tailwind UI (drag & drop, paste, sample images)
training/               Reproducible training / evaluation scripts
notebooks/              Original Kaggle/Colab notebooks (dataset, augmentation, training)
dataset/                Label spreadsheets, augmented dataset index, 53-image test set
Project Documents/      Report, synopsis, technical paper, poster, PPTs, diagrams, metrics, screenshots
experiments/            Earlier Gemini LLM baseline (not used by the app)
test_images/            Extra sample images
deploy/huggingface/     Hugging Face Space files (live demo)
Dockerfile              One-container build (frontend + backend)
```

## 🚀 Run locally

**Requirements:** Python 3.10+ and Node.js 18+.

```bash
# 1. Backend (http://localhost:8000, interactive API docs at /docs)
cd backend
python -m venv .venv
.venv\Scripts\activate             # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 2. Frontend (new terminal, http://localhost:3000)
cd frontend
npm install
npm start
```

**Or with Docker:** `docker build -t jewellery-analyzer .` then `docker run -p 7860:7860 jewellery-analyzer`, and open http://localhost:7860.

### API

```bash
curl -F "file=@test_images/1.jpeg" http://localhost:8000/analyze
```
Example response:

```json
{
  "type": "Earring",
  "confidence": 0.9999,
  "description": "Pair of gold earrings with dangling diamond accents in a crescent moon design.",
  "method": "retrieved",
  "similarity": 0.89,
  "generated_description": "…",
  "inference_ms": 420
}
```

Add `?mode=generate` to get the GRU decoder's caption as the main description.

### Tests

```bash
cd backend && pip install pytest httpx && pytest -q
cd frontend && npm test
```

The tests also run automatically on every push through GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## 🧪 Training and evaluation

The original notebooks are kept in [`notebooks/`](notebooks). [`training/`](training) contains the same models as command-line scripts, with two fixes: validation is split by **original photo**, so augmented copies don't leak across splits, and VGG-16 features are cached so caption training takes minutes.

```bash
pip install -r training/requirements.txt
python training/train_classifier.py   --csv dataset/labels/augmented_dataset.csv --images <augmented image folder> --out backend/weights/classifier.pth
python training/train_captioning.py   --csv dataset/labels/augmented_dataset.csv --images <augmented image folder> --out-dir backend/weights
python training/build_caption_index.py --csv dataset/labels/augmented_dataset.csv --images <augmented image folder> --out backend/weights/caption_index.npz
python training/evaluate.py           # accuracy, BLEU-4 and latency on dataset/test_set
```

## 🌐 Deployment

The live demo runs as a free Hugging Face Space (Docker, CPU), which builds straight from this GitHub repository:

1. Create a Space at https://huggingface.co/new-space, choosing **Docker → Blank** and **CPU basic** (free).
2. Upload the two files from [`deploy/huggingface/`](deploy/huggingface): `Dockerfile` and `README.md`.
3. Wait for the build (~10 min). The app is then live at `https://huggingface.co/spaces/<username>/<space-name>`.
4. After pushing to GitHub, use **Settings → Factory rebuild** on the Space to update it.

## 📊 Dataset

The dataset combines open Hugging Face jewellery datasets (*AI Tool Pool Jewelry Vision*, `imnaveenk/necklace`, `Bhargav1127/necklace-dataset-with-captions`) with manually collected images. 5,307 captioned originals were augmented about 6× (flips, rotations, colour shifts). See [`dataset/README.md`](dataset/README.md). The raw image folders (~450 MB) are not stored in this repository.

## ⚠️ Limitations and future work

- **Only two classes.** Earrings and necklaces only; rings, bracelets and so on will be forced into one of the two (the UI flags low-confidence results).
- **Background sensitivity.** Most training photos are product shots on plain backgrounds. Earrings photographed while worn, or on busy backgrounds, are sometimes classified as necklaces.
- **Free-form captions.** The GRU decoder tends to reproduce fragments of training captions on unseen images. Retraining with the group-aware split in `training/`, adding attention over the 7×7 feature map, or fine-tuning the upper VGG layers are the natural next steps.

## 🛠️ Tech stack

PyTorch · Torchvision · FastAPI · Uvicorn · React · Tailwind CSS · Axios · NumPy · Pandas · NLTK · Pillow · Docker · GitHub Actions

## 👥 Team

Tanisha Mangaonkar · Prachi Gandhi · Chandana Galgali · Mahek Thakkar
Guide: Prof. Avani Sakhapara, Department of Information Technology, KJSCE

## 📄 License

[MIT](LICENSE)
