# 💎 Automated Description Generation for Jewellery Images using Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-VGG--16%20%2B%20GRU-ee4c2c)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![React](https://img.shields.io/badge/Frontend-React-61dafb)

**Live demo:** _add your Hugging Face Space link here once deployed (see [Deployment](#-deployment))_

An end-to-end computer-vision and NLP application that takes a jewellery photo, **classifies** it (earring or necklace) and **generates a natural-language description** of it, for example its metal, stones, shape and style. Both tasks use an encoder-decoder design built on a VGG-16 image encoder and a GRU.

B.Tech (Information Technology) final-year project, K. J. Somaiya College of Engineering, Somaiya Vidyavihar University (2025).

## ✨ Results

| Metric | Value |
|---|---|
| Dataset size (after augmentation) | **32,190 images** |
| Jewellery type classification accuracy (validation) | **94.62%** |
| Caption generation accuracy (validation, next-word accuracy) | **90.44%** |
| Inference time per image | ~1–2 s on CPU |

The 90.44% figure is token-level accuracy on the 6,318-image validation split (see `notebooks/03_captioning_vgg16_gru_training.ipynb`). The same notebook also evaluates on 53 extra web images in [`dataset/test_set`](dataset/test_set): there the type accuracy is 81.13% and the BLEU-4 score is 0.04. In other words, the descriptions generalise less well to images unlike the training data. Classification metrics and plots are in [`docs/Classification Model Metrics`](docs/Classification%20Model%20Metrics).

## 🏗️ Architecture

![Architecture](docs/System%20Workflow%20Diagram/Architecture.jpg)

**1. Classifier: VGG-16 → GRU → Linear.** A frozen VGG-16 backbone produces a 7×7×512 feature map. The model reads it as a sequence of 49 spatial vectors and passes them through a GRU (hidden size 256), whose last hidden state is classified as *Earring* or *Necklace*.

**2. Captioning: VGG-16 encoder → GRU decoder.** The VGG-16 features are average-pooled and projected to 512-d. That vector initialises a GRU language model (embedding size 256, hidden size 512, vocabulary of 2,158 words), which generates the description word by word using greedy decoding.

**Preprocessing:** resize to 224×224, tensor conversion and ImageNet normalisation (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

## 📂 Repository structure

```
backend/               FastAPI app + model code
  main.py              API (/analyze, /health); also serves the built frontend
  caption_model.py     Model definitions, vocabulary loader, caption decoding
  weights/             Trained weights (classifier.pth, encoder.pth, decoder.pth, vocab.pkl)
frontend/              React + Tailwind UI
notebooks/             Dataset building, augmentation and training notebooks (Kaggle / Colab)
dataset/               Label spreadsheets, dataset info and the 53-image description test set
docs/                  Project report, technical paper, poster, PPTs, synopsis, diagrams, metrics
experiments/           Earlier LLM (Gemini) baseline, not used by the app
test_images/           Sample images to try in the demo
deploy/huggingface/    Files for the Hugging Face Space (live demo)
Dockerfile             One-container build (frontend + backend)
```

## 🚀 Run locally

**Requirements:** Python 3.10+ and Node.js 18+.

```bash
# 1. Backend (http://localhost:8000, API docs at /docs)
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 2. Frontend (new terminal, http://localhost:3000)
cd frontend
npm install
npm start
```

Upload any image from `test_images/` and click **Identify Item**.

**Or with Docker:** `docker build -t jewellery-analyzer .` then `docker run -p 7860:7860 jewellery-analyzer`, and open http://localhost:7860.

### API

```bash
curl -F "file=@test_images/1.jpeg" http://localhost:8000/analyze
```
```json
{ "type": "Earring", "confidence": 0.9987, "description": "A pair of gold ...", "inference_ms": 850 }
```

## 🌐 Deployment

The live demo runs as a free Hugging Face Space (Docker, CPU). The Space builds directly from this GitHub repo:

1. Create a new Space at https://huggingface.co/new-space, choose **Docker → Blank** and the free **CPU basic** hardware.
2. Upload the two files from [`deploy/huggingface/`](deploy/huggingface) (`Dockerfile` and `README.md`) to the Space.
3. Wait for the build (~5–10 min). The app is then live at `https://huggingface.co/spaces/<username>/<space-name>`.
4. After pushing changes to GitHub, use **Settings → Factory rebuild** on the Space.

## 📊 Dataset

The dataset was built from open Hugging Face jewellery datasets (*AI Tool Pool Jewelry Vision*, `imnaveenk/necklace`, `Bhargav1127/necklace-dataset-with-captions`) plus manually collected images. 5,407 captioned images were augmented about 6× (flips, rotations, colour shifts), giving 32,190 images. See [`dataset/README.md`](dataset/README.md). The full image folders (~450 MB) are not stored in this repo.

## 🛠️ Tech stack

PyTorch · Torchvision · FastAPI · Uvicorn · React · Tailwind CSS · Axios · Pandas · NLTK · Pillow

## 👥 Team

Tanisha Mangaonkar · Prachi Gandhi · Chandana Galgali · Mahek Thakkar
Guide: Prof. Avani Sakhapara, Department of Information Technology, KJSCE

## 📄 License

[MIT](LICENSE)
