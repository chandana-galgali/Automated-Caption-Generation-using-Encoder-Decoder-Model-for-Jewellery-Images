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

Validation accuracy at the selected epoch (2): 99.01%. Confusion matrix: `results/confusion_matrix.png`.

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
