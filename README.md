# 💎 Automated Caption Generation using Encoder Decoder Model for Jewellery Images

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-VGG--16%20%2B%20GRU-ee4c2c)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![React](https://img.shields.io/badge/Frontend-React-61dafb)
![CI](https://github.com/chandana-galgali/Automated-Caption-Generation-using-Encoder-Decoder-Model-for-Jewellery-Images/actions/workflows/ci.yml/badge.svg)

**🔗 Live demo: [chandana-galgali.github.io/Automated-Caption-Generation-using-Encoder-Decoder-Model-for-Jewellery-Images](https://chandana-galgali.github.io/Automated-Caption-Generation-using-Encoder-Decoder-Model-for-Jewellery-Images/)**: the trained models run directly in your browser, with no server and no sign-up.

An end-to-end computer-vision and NLP application. Upload a photo of a piece of jewellery and it **classifies** the item (earring or necklace) and writes a **natural-language description** of it, covering metal, stones, shape and style. It uses a VGG-16 encoder with GRU-based decoders, served by a FastAPI backend with a React frontend.

B.Tech (Information Technology) final-year project, K. J. Somaiya College of Engineering, Somaiya Vidyavihar University (2025). The report, synopsis, technical paper, poster and all presentations are in [`project documents/`](project%20documents).

## 📊 Results

<!-- METRICS:START -->
_All numbers below were produced by `training/run_all.py` on 2026-11-21 (Tesla T4, seed 42). Test sets were not used for any training or model-selection decision._

#### Dataset

| Split | Earrings | Necklaces | Total | Used for |
|---|---|---|---|---|
| train | 2484 | 1301 | 3785 | training (+ 5 augmented views per image) |
| val | 532 | 279 | 811 | model selection / early stopping |
| test | 532 | 278 | 810 | final evaluation only |
| external | 23 | 30 | 53 | unseen web images, final evaluation only |

Splits are grouped so that duplicate photos and identical descriptions never appear in two splits.

#### Jewellery type classification (VGG-16 + GRU)

| Test set | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC |
|---|---|---|---|---|---|
| Held-out test (n=810) | **98.64%** | 98.38% | 98.62% | 98.50% | 0.9986 |
| External web images (n=53) | **100.00%** | 100.00% | 100.00% | 100.00% | 1.0000 |

| Class (held-out test) | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Earring | 99.24% | 98.68% | 98.96% | 532 |
| Necklace | 97.51% | 98.56% | 98.03% | 278 |

Validation accuracy at the selected epoch (2): 99.01%. Confusion matrix: [confusion matrix](results/confusion_matrix.png).

#### Description generation

| Method | Test set | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr-D |
|---|---|---|---|---|---|---|---|---|
| VGG-16 + attention GRU (beam 3) | held-out | 0.5273 | 0.3728 | 0.2878 | **0.2273** | 0.4964 | 0.5154 | 0.8755 |
| VGG-16 + attention GRU (beam 3) | external web | 0.3334 | 0.1583 | 0.0909 | **0.0421** | 0.2477 | 0.3476 | 0.1669 |
| Nearest training description | held-out | 0.4978 | 0.3369 | 0.2470 | **0.1864** | 0.4601 | 0.4794 | 0.7103 |
| Nearest training description | external web | 0.2948 | 0.1470 | 0.0907 | **0.0482** | 0.2263 | 0.3174 | 0.2058 |

Captioner selected at epoch 23 by validation CIDEr-D. Default description mode in the app: **generate** (validation CIDEr-D: generated 1.0360, retrieved 0.7767).

#### Inference speed

End-to-end (classification + description) on CPU with 2 threads: median **352 ms**, 95th percentile 986 ms over 30 test images.

Training time: 7.2 min on Tesla T4.

![Training curves](results/training_curves.png)
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
frontend/               React + Tailwind UI; runs the models in the browser on GitHub Pages
training/               Dataset preparation, training + evaluation pipeline, metrics
results/                Metrics, plots and test-set predictions of the latest training run
notebooks/              Original notebooks + Kaggle training notebook
dataset/                Label files and the 53-image external test set
project documents/      Report, technical paper, poster, presentations, architecture, screenshots
experiments/            Earlier Gemini LLM baseline (not used by the app)
test_images/            Sample images
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

**Or with Docker:** `docker build -t jewellery-analyzer .` then `docker run -p 8000:8000 jewellery-analyzer`, and open http://localhost:8000.

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

The live demo is a static website on **GitHub Pages**. The trained PyTorch models are exported to ONNX and run in the visitor's browser with ONNX Runtime Web, so the demo is free, always on and needs no server. On every push to `main`, [`.github/workflows/pages.yml`](.github/workflows/pages.yml):

1. exports `backend/weights` to ONNX (`training/export_onnx.py`) and checks that ONNX reproduces the PyTorch predictions and descriptions exactly;
2. builds the React app in browser mode (`REACT_APP_INFERENCE=browser`);
3. opens the built site in headless Chromium and analyses a sample image and an uploaded image (end-to-end test);
4. publishes the site only if all of the above pass.

Image decoding and resizing in the browser reproduce Pillow bit for bit (`frontend/src/browserModel.js`), so the website gives the same results as the Python backend. The model (~75 MB) is downloaded once and then cached by the browser.

The FastAPI backend can also be self-hosted on any server with Docker: `docker build -t jewellery-analyzer .` then `docker run -p 8000:8000 jewellery-analyzer`.

## 📁 Dataset

The dataset was built from open Hugging Face jewellery datasets (*AI Tool Pool Jewelry Vision*, `imnaveenk/necklace`, `Bhargav1127/necklace-dataset-with-captions`) plus manually collected images: 5,406 photos (3,548 earrings and 1,858 necklaces), 5,306 of them with a description. See [`dataset/README.md`](dataset/README.md). The image folders are not stored in this repository.

## ⚠️ Limitations

- Only two classes: earrings and necklaces. Other jewellery is forced into one of the two, and the UI flags low-confidence results.
- Most training photos are product shots on plain backgrounds, so performance on worn or busy-background photos is lower. The external test set measures this.

## 🛠️ Tech stack

PyTorch · Torchvision · ONNX / ONNX Runtime Web · FastAPI · Uvicorn · React · Tailwind CSS · NumPy · Matplotlib · NLTK · Pillow · Docker · GitHub Actions · GitHub Pages

## 👥 Team

Tanisha Mangaonkar · Prachi Gandhi · Chandana Galgali · Mahek Thakkar
Guide: Prof. Avani Sakhapara, Department of Information Technology, KJSCE

## 📄 License

[MIT](LICENSE)