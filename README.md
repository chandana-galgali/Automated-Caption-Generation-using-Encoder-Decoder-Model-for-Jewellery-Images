# 💎 Automated Description Generation for Jewellery Images using Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-VGG--16%20%2B%20GRU-ee4c2c)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![React](https://img.shields.io/badge/Frontend-React-61dafb)
![CI](https://github.com/chandana-galgali/Automated-Caption-Generation-using-Encoder-Decoder-Model-for-Jewellery-Images/actions/workflows/ci.yml/badge.svg)

**Live demo:** _add your Hugging Face Space link here once deployed (see [Deployment](#-deployment))_

An end-to-end computer-vision and NLP application. Upload a photo of a piece of jewellery and it **classifies** the item (earring or necklace) and writes a **natural-language description** of it, covering metal, stones, shape and style. It uses a VGG-16 encoder with GRU-based decoders, served by a FastAPI backend with a React frontend.

B.Tech (Information Technology) final-year project, K. J. Somaiya College of Engineering, Somaiya Vidyavihar University (2025). The report, synopsis, technical paper, poster and all presentations are in [`project documents/`](project%20documents).

## 📊 Results

<!-- METRICS:START -->
_Metrics are generated automatically by `training/run_all.py` and inserted here by `training/install_results.py` (see [Training and evaluation](#-training-and-evaluation))._
<!-- METRICS:END -->

## 🏗️ Architecture

![Architecture](project%20documents/System%20Architecture.jpg)

1. **Encoder.** The image is resized to 224×224, ImageNet-normalised and passed **once** through a frozen, ImageNet-pretrained VGG-16, giving a 7×7×512 feature map (49 image regions).
2. **Classifier (VGG-16 → GRU).** The 49 region vectors are read as a sequence by a GRU (hidden 256), and a linear layer predicts *Earring* or *Necklace*.
3. **Description decoder (VGG-16 → attention GRU).** A GRU language model (embedding 256, hidden 512) with soft attention over the 49 regions ("Show, Attend and Tell"). At every word it looks at the relevant part of the image. Captions are decoded with beam search (k = 3).
4. **Description retrieval (baseline).** The most similar training image of the predicted type, by cosine similarity of the same VGG-16 features, provides a reference description. Generation or retrieval becomes the app's default, whichever scores higher on the **validation** set.

### Engineering details

- **One VGG-16 pass per image**, shared by the classifier and the description decoder.
- **Feature caching during training.** The frozen VGG-16 features of every image (plus 5 augmented views per training image) are computed once, so all heads train in minutes on a single GPU.
- **Leakage-free splits.** Train, validation and test are split by photo, with duplicate photos and identical descriptions grouped together. Augmentation is applied only to training images, on the fly.
- **Incremental, batched decoding** with the GRU state carried forward.
- The models load once at server start-up, and the frontend downscales images before upload.

## 📂 Repository structure

```
backend/                FastAPI app + model code
  main.py               API: POST /analyze, GET /health; also serves the built frontend
  caption_model.py      Model definitions + inference pipeline
  weights/              Trained weights
  tests/                pytest suite
frontend/               React + Tailwind UI (drag & drop, paste, sample images)
training/               Dataset preparation, training + evaluation pipeline, metrics
results/                Metrics, plots and test-set predictions of the latest training run
notebooks/              Original notebooks + Kaggle training notebook
dataset/                Label files and the 53-image external test set
project documents/      Report, technical paper, poster, presentations, architecture, screenshots
experiments/            Earlier Gemini LLM baseline (not used by the app)
test_images/            Sample images
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

Response fields: `type`, `confidence`, `description`, `method` (`generated` or `retrieved`), `generated_description` and `inference_ms`. Use `?mode=generate` or `?mode=retrieve` to choose the description method explicitly.

## 🧪 Training and evaluation

Everything (training, model selection and evaluation) is one command. The test sets are used only for the final evaluation.

```bash
# 1. Build the dataset package from the raw image folders (once)
python training/prepare_dataset.py --raw ../Dataset --labels dataset/labels/master_labels.csv \
    --external ../Dataset/DESCRIPTION_TEST_DATA --external-labels dataset/test_set/labels.csv \
    --out ../jewellery-dataset

# 2. Train + evaluate (GPU recommended; on Kaggle use notebooks/04_kaggle_train_and_evaluate.ipynb)
python training/run_all.py --data ../jewellery-dataset --out run

# 3. Install the trained weights, metrics and plots, and update this README
python training/install_results.py run.zip
```

Reported metrics:

- **Classification:** accuracy, precision, recall and F1 per class and macro-averaged, ROC-AUC and the confusion matrix.
- **Descriptions:** BLEU-1 to BLEU-4, METEOR, ROUGE-L and CIDEr-D.
- **Speed:** CPU latency.

All are computed on a held-out test split and on 53 external web images. Training curves and per-image test predictions are saved to [`results/`](results).

## 🌐 Deployment

The live demo runs as a free Hugging Face Space (Docker, CPU), which builds straight from this GitHub repository:

1. Create a Space at https://huggingface.co/new-space, choosing **Docker → Blank** and **CPU basic** (free).
2. Upload the two files from [`deploy/huggingface/`](deploy/huggingface): `Dockerfile` and `README.md`.
3. Wait for the build (~10 min). The app is then live at `https://huggingface.co/spaces/<username>/<space-name>`.
4. After pushing to GitHub, use **Settings → Factory rebuild** on the Space to update it.

## 📁 Dataset

The dataset was built from open Hugging Face jewellery datasets (*AI Tool Pool Jewelry Vision*, `imnaveenk/necklace`, `Bhargav1127/necklace-dataset-with-captions`) plus manually collected images: 5,406 photos (3,548 earrings and 1,858 necklaces), 5,306 of them with a description. See [`dataset/README.md`](dataset/README.md). The image folders are not stored in this repository.

## ⚠️ Limitations

- Only two classes: earrings and necklaces. Other jewellery is forced into one of the two, and the UI flags low-confidence results.
- Most training photos are product shots on plain backgrounds, so performance on worn or busy-background photos is lower. The external test set measures this.

## 🛠️ Tech stack

PyTorch · Torchvision · FastAPI · Uvicorn · React · Tailwind CSS · Axios · NumPy · Matplotlib · NLTK · Pillow · Docker · GitHub Actions

## 👥 Team

Tanisha Mangaonkar · Prachi Gandhi · Chandana Galgali · Mahek Thakkar
Guide: Prof. Avani Sakhapara, Department of Information Technology, KJSCE

## 📄 License

[MIT](LICENSE)
