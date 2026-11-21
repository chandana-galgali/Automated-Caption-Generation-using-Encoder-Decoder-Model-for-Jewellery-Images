"""
One command to train and evaluate the complete system and produce every number
shown in the README.

    python training/run_all.py --data /path/to/jewellery-dataset --out run

The dataset folder is produced by training/prepare_dataset.py (labels.csv with a
fixed, leakage-free train / val / test split + external_test/).

Pipeline
 1. VGG-16 (ImageNet, frozen) feature maps (7x7x512) for every image: the clean
    view plus N augmented views per training image (random crop, flip, rotation,
    colour jitter) - computed once, so the heads train in minutes on a GPU.
 2. Classifier: GRU over the 49 VGG regions -> Earring / Necklace.
    Class-weighted cross-entropy; best epoch chosen by validation macro-F1.
 3. Captioner: GRU decoder with soft attention over the 49 regions.
    Label smoothing + doubly-stochastic attention regularisation; best epoch
    chosen by validation CIDEr-D (early stopping).
 4. Retrieval baseline: description of the most similar training image.
    Whichever of generation / retrieval scores higher CIDEr-D on VALIDATION
    becomes the app's default (decided before looking at the test set).
 5. Held-out TEST set + external web test set: accuracy, precision, recall, F1,
    ROC-AUC, confusion matrix, BLEU-1..4, METEOR, ROUGE-L, CIDEr-D; CPU latency.
Outputs: <out>/weights (drop-in for backend/weights), metrics.json, METRICS.md,
plots, per-image predictions, and <out>.zip containing all of it.
"""

import argparse
import copy
import csv
import json
import pickle
import platform
import random
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from caption_model import (  # noqa: E402
    CLASS_NAMES, AttentionCaptioner, JewelleryAnalyzer, VGG16_GRU_Classifier, Vocabulary,
    beam_search, detokenize, greedy_batch, tokenize, transform as EVAL_TF,
)
from metrics import caption_metrics, classification_metrics  # noqa: E402

AUG_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.RandomRotation(15, fill=(255, 255, 255))], p=0.5),
    transforms.ColorJitter(0.25, 0.25, 0.25, 0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

START_TIME = time.time()


def log(*args):
    print(f"[{time.time() - START_TIME:7.1f}s]", *args, flush=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------- data
def load_data(root):
    rows = list(csv.DictReader(open(root / "labels.csv", encoding="utf-8")))
    for r in rows:
        r["label"] = CLASS_NAMES.index(r["type"].capitalize())
    splits = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    ext_csv = root / "external_test" / "labels.csv"
    external = []
    if ext_csv.exists():
        for r in csv.DictReader(open(ext_csv, encoding="utf-8")):
            external.append({"file": "external_test/" + r["file"], "description": r["description"],
                             "type": r["type"], "label": CLASS_NAMES.index(r["type"].capitalize())})
    return splits, external


class Views(Dataset):
    def __init__(self, root, rows, n_views, augment):
        self.root, self.rows, self.n, self.augment = root, rows, n_views, augment

    def __len__(self):
        return len(self.rows) * self.n

    def __getitem__(self, i):
        r, v = self.rows[i // self.n], i % self.n
        im = Image.open(self.root / r["file"]).convert("RGB")
        return EVAL_TF(im) if v == 0 or not self.augment else AUG_TF(im)


@torch.no_grad()
def extract(backbone, root, rows, n_views, augment, device, batch, workers):
    """Returns float16 CPU tensor (len(rows), n_views, 512, 7, 7)."""
    loader = DataLoader(Views(root, rows, n_views, augment), batch_size=batch, num_workers=workers,
                        pin_memory=device.type == "cuda")
    out = []
    for x in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = backbone(x.to(device, non_blocking=True))
        out.append(f.half().cpu())
    return torch.cat(out).view(len(rows), n_views, 512, 7, 7)


# ------------------------------------------------------------- classifier
@torch.no_grad()
def classify(clf, feats, device, batch=256):
    clf.eval()
    probs = []
    for i in range(0, len(feats), batch):
        probs.append(torch.softmax(clf.head(feats[i:i + batch].to(device).float()), 1).cpu())
    return torch.cat(probs)


def train_classifier(clf, ftr, ytr, fva, yva, device, args):
    params = list(clf.gru.parameters()) + list(clf.fc.parameters())
    opt = torch.optim.Adam(params, lr=args.cls_lr, weight_decay=1e-4)
    counts = torch.bincount(ytr, minlength=len(CLASS_NAMES)).float()
    loss_fn = nn.CrossEntropyLoss(weight=(counts.sum() / (len(counts) * counts)).to(device))
    n, v = ftr.shape[:2]
    best, best_state, history = -1.0, None, []
    for epoch in range(1, args.cls_epochs + 1):
        clf.train()
        idx = torch.randperm(n * v)
        total = 0.0
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            x = ftr[b // v, b % v].to(device).float()
            y = ytr[b // v].to(device)
            loss = loss_fn(clf.head(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        p = classify(clf, fva, device)
        m = classification_metrics(yva.numpy(), p.argmax(1).numpy(), p[:, 1].numpy(), CLASS_NAMES)
        history.append({"epoch": epoch, "train_loss": total / len(idx), "val_accuracy": m["accuracy"],
                        "val_macro_f1": m["macro_f1"]})
        log(f"classifier epoch {epoch:2d}  loss {total / len(idx):.4f}  "
            f"val acc {m['accuracy']:.4f}  val macro-F1 {m['macro_f1']:.4f}")
        if m["macro_f1"] > best:
            best, best_state = m["macro_f1"], copy.deepcopy(clf.state_dict())
    clf.load_state_dict(best_state)
    best_epoch = max(history, key=lambda h: h["val_macro_f1"])["epoch"]
    return history, best_epoch


# -------------------------------------------------------------- captioner
def build_vocab(captions, threshold):
    counts = Counter(w for c in captions for w in tokenize(c))
    vocab = Vocabulary(threshold)
    vocab.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
    for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if c >= threshold:
            vocab.itos[len(vocab.itos)] = w
    vocab.stoi = {w: i for i, w in vocab.itos.items()}
    return vocab


def numericalize(vocab, text, max_len):
    ids = [vocab.stoi.get(w, vocab.stoi["<UNK>"]) for w in tokenize(text)][:max_len]
    return torch.tensor([vocab.stoi["<START>"]] + ids + [vocab.stoi["<END>"]])


def step_fn(cap):
    return lambda tok, st: cap.step(tok, st)[:2]


@torch.no_grad()
def decode_greedy(cap, feats, vocab, device, batch=256):
    cap.eval()
    out = []
    for i in range(0, len(feats), batch):
        f = feats[i:i + batch].to(device).float()
        out += greedy_batch(step_fn(cap), cap.encode(f), vocab, len(f))
    return out


@torch.no_grad()
def decode_beam(cap, feats, vocab, device, k):
    cap.eval()
    return [beam_search(step_fn(cap), cap.encode(feats[i:i + 1].to(device).float()), vocab, k)
            for i in range(len(feats))]


def train_captioner(cap, vocab, ftr, ctr, fva, refs_va, device, args):
    pad = vocab.stoi["<PAD>"]
    opt = torch.optim.Adam(cap.parameters(), lr=args.cap_lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad, label_smoothing=0.1)
    n, v = ftr.shape[:2]
    best, best_state, history, bad = -1.0, None, [], 0
    for epoch in range(1, args.cap_epochs + 1):
        cap.train()
        order = torch.randperm(n)
        views = torch.randint(0, v, (n,))  # a different augmented view every epoch
        total = words = 0
        for i in range(0, n, args.batch):
            b = order[i:i + args.batch]
            x = ftr[b, views[b]].to(device).float()
            c = pad_sequence([ctr[j] for j in b.tolist()], batch_first=True, padding_value=pad).to(device)
            logits, alphas = cap(x, c[:, :-1])
            target = c[:, 1:]
            mask = (target != pad).float()
            ce = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            att = (((1.0 - (alphas * mask.unsqueeze(-1)).sum(1)) ** 2).mean())
            loss = ce + args.att_reg * att
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(cap.parameters(), 5.0)
            opt.step()
            total += ce.item() * mask.sum().item(); words += mask.sum().item()
        hyps = decode_greedy(cap, fva, vocab, device)
        m = caption_metrics(refs_va, hyps)
        sched.step(m["CIDEr-D"])
        history.append({"epoch": epoch, "train_loss": total / words, "val_BLEU-4": m["BLEU-4"],
                        "val_CIDEr-D": m["CIDEr-D"], "lr": opt.param_groups[0]["lr"]})
        log(f"captioner epoch {epoch:2d}  loss {total / words:.4f}  val BLEU-4 {m['BLEU-4']:.4f}  "
            f"val CIDEr-D {m['CIDEr-D']:.4f}  e.g. \"{detokenize(hyps[0])}\"")
        if m["CIDEr-D"] > best:
            best, best_state, bad = m["CIDEr-D"], copy.deepcopy(cap.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                log(f"early stopping (no validation improvement for {args.patience} epochs)")
                break
    cap.load_state_dict(best_state)
    best_epoch = max(history, key=lambda h: h["val_CIDEr-D"])["epoch"]
    return history, best_epoch


# -------------------------------------------------------------- retrieval
def retrieval(index_feats, index_labels, index_caps, query_feats, query_labels):
    """Description of the most similar index image of the same (predicted) type."""
    sims = F.normalize(query_feats, dim=1) @ F.normalize(index_feats, dim=1).T
    sims[query_labels.unsqueeze(1) != index_labels.unsqueeze(0)] = -2.0
    return [index_caps[i] for i in sims.argmax(1).tolist()]


# ---------------------------------------------------------------- reports
def pct(x):
    return "n/a" if x is None else f"{100 * x:.2f}%"


def num(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


def write_report(out, M):
    d, c, cap = M["dataset"], M["classification"], M["captioning"]
    L = []
    L.append(f"_All numbers below were produced by `training/run_all.py` on {M['run']['date']} "
             f"({M['run']['device']}, seed {M['run']['seed']}). Test sets were not used for any training "
             f"or model-selection decision._\n")
    L.append("#### Dataset\n")
    L.append("| Split | Earrings | Necklaces | Total | Used for |")
    L.append("|---|---|---|---|---|")
    use = {"train": "training (+ %d augmented views per image)" % (M["run"]["views"] - 1),
           "val": "model selection / early stopping", "test": "final evaluation only"}
    for s in ("train", "val", "test"):
        L.append(f"| {s} | {d[s]['Earring']} | {d[s]['Necklace']} | {d[s]['total']} | {use[s]} |")
    L.append(f"| external | {d['external']['Earring']} | {d['external']['Necklace']} | "
             f"{d['external']['total']} | unseen web images, final evaluation only |\n")
    L.append("Splits are grouped so that duplicate photos and identical descriptions never appear in two splits.\n")

    L.append("#### Jewellery type classification (VGG-16 + GRU)\n")
    L.append("| Test set | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC |")
    L.append("|---|---|---|---|---|---|")
    for name, key in (("Held-out test", "test"), ("External web images", "external")):
        if key in c:
            m = c[key]
            L.append(f"| {name} (n={sum(map(sum, m['confusion_matrix']))}) | **{pct(m['accuracy'])}** | "
                     f"{pct(m['macro_precision'])} | {pct(m['macro_recall'])} | {pct(m['macro_f1'])} | {num(m.get('roc_auc'))} |")
    L.append("")
    L.append("| Class (held-out test) | Precision | Recall | F1 | Support |")
    L.append("|---|---|---|---|---|")
    for k, v in c["test"]["per_class"].items():
        L.append(f"| {k} | {pct(v['precision'])} | {pct(v['recall'])} | {pct(v['f1'])} | {v['support']} |")
    L.append(f"\nValidation accuracy at the selected epoch ({c['best_epoch']}): {pct(c['val']['accuracy'])}. "
             "Confusion matrix: `results/confusion_matrix.png`.\n")

    L.append("#### Description generation\n")
    L.append("| Method | Test set | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr-D |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for method, label in (("generated", "VGG-16 + attention GRU (beam 3)"), ("retrieved", "Nearest training description")):
        for key, setname in (("test", "held-out"), ("external", "external web")):
            if key in cap[method]:
                m = cap[method][key]
                L.append(f"| {label} | {setname} | {num(m['BLEU-1'])} | {num(m['BLEU-2'])} | {num(m['BLEU-3'])} | "
                         f"**{num(m['BLEU-4'])}** | {num(m['METEOR'])} | {num(m['ROUGE-L'])} | {num(m['CIDEr-D'])} |")
    L.append(f"\nCaptioner selected at epoch {cap['best_epoch']} by validation CIDEr-D. "
             f"Default description mode in the app: **{cap['default_mode']}** "
             f"(validation CIDEr-D: generated {num(cap['val_cider']['generated'])}, "
             f"retrieved {num(cap['val_cider']['retrieved'])}).\n")

    lat = M.get("latency")
    if lat:
        L.append("#### Inference speed\n")
        L.append(f"End-to-end (classification + description) on CPU with {lat['threads']} threads: "
                 f"median **{lat['median_ms']:.0f} ms**, 95th percentile {lat['p95_ms']:.0f} ms "
                 f"over {lat['n']} test images.\n")
    L.append(f"Training time: {M['run']['minutes']:.1f} min on {M['run']['device']}.")
    (out / "METRICS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def plots(out, cls_hist, cap_hist, cm):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        log("matplotlib not available - skipping plots")
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    e = [h["epoch"] for h in cls_hist]
    ax[0].plot(e, [h["train_loss"] for h in cls_hist], label="train loss")
    ax[0].plot(e, [h["val_accuracy"] for h in cls_hist], label="val accuracy")
    ax[0].plot(e, [h["val_macro_f1"] for h in cls_hist], label="val macro-F1")
    ax[0].set_title("Classifier"); ax[0].set_xlabel("epoch"); ax[0].legend()
    e = [h["epoch"] for h in cap_hist]
    ax[1].plot(e, [h["train_loss"] for h in cap_hist], label="train loss")
    ax2 = ax[1].twinx()
    ax2.plot(e, [h["val_CIDEr-D"] for h in cap_hist], color="tab:green", label="val CIDEr-D")
    ax2.plot(e, [h["val_BLEU-4"] for h in cap_hist], color="tab:red", label="val BLEU-4")
    ax[1].set_title("Captioner"); ax[1].set_xlabel("epoch"); ax[1].legend(loc="upper left"); ax2.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Greens")
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=12)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES); ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Held-out test")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", default="run", type=Path)
    ap.add_argument("--views", type=int, default=6, help="1 clean + (views-1) augmented views per training image")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cls-epochs", type=int, default=15)
    ap.add_argument("--cls-lr", type=float, default=1e-3)
    ap.add_argument("--cap-epochs", type=int, default=40)
    ap.add_argument("--cap-lr", type=float, default=4e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--att-reg", type=float, default=1.0)
    ap.add_argument("--vocab-threshold", type=int, default=3)
    ap.add_argument("--max-caption-len", type=int, default=40)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run with random weights (pipeline test)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else f"CPU ({platform.processor() or 'unknown'})"
    out = args.out
    (out / "weights").mkdir(parents=True, exist_ok=True)
    log(f"device: {dev_name}")

    splits, external = load_data(args.data)
    if args.smoke:
        splits = {k: v[:{"train": 40, "val": 16, "test": 16}[k]] for k, v in splits.items()}
        external = external[:8]
        args.views, args.cls_epochs, args.cap_epochs, args.workers = 2, 1, 2, 0
    for s, rows in splits.items():
        log(f"{s}: {len(rows)} images")

    # 1. features -----------------------------------------------------------
    weights = None if args.smoke else models.VGG16_Weights.IMAGENET1K_V1
    backbone = models.vgg16(weights=weights).features.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    ftr = extract(backbone, args.data, splits["train"], args.views, True, device, args.batch, args.workers)
    log(f"train features {tuple(ftr.shape)}")
    fva = extract(backbone, args.data, splits["val"], 1, False, device, args.batch, args.workers)[:, 0]
    fte = extract(backbone, args.data, splits["test"], 1, False, device, args.batch, args.workers)[:, 0]
    fex = extract(backbone, args.data, external, 1, False, device, args.batch, args.workers)[:, 0] if external else None
    log("features done")

    lab = {s: torch.tensor([r["label"] for r in rows]) for s, rows in splits.items()}

    # 2. classifier ------------------------------------------------------------
    clf = VGG16_GRU_Classifier(len(CLASS_NAMES))
    clf.features = backbone
    clf.to(device)
    cls_hist, cls_best = train_classifier(clf, ftr, lab["train"], fva, lab["val"], device, args)
    cls_res = {"best_epoch": cls_best, "history": cls_hist}
    pred = {}
    for key, feats, y in (("val", fva, lab["val"]), ("test", fte, lab["test"]),
                          ("external", fex, torch.tensor([r["label"] for r in external]) if external else None)):
        if feats is None:
            continue
        p = classify(clf, feats, device)
        pred[key] = p
        cls_res[key] = classification_metrics(y.numpy(), p.argmax(1).numpy(), p[:, 1].numpy(), CLASS_NAMES)
    log(f"classifier TEST accuracy {cls_res['test']['accuracy']:.4f}  macro-F1 {cls_res['test']['macro_f1']:.4f}")

    # 3. captioner -------------------------------------------------------------
    cap_rows = {s: [i for i, r in enumerate(rows) if r["description"].strip()] for s, rows in splits.items()}
    tr_caps = [splits["train"][i]["description"] for i in cap_rows["train"]]
    vocab = build_vocab(tr_caps, args.vocab_threshold)
    log(f"vocabulary: {len(vocab)} words (threshold {args.vocab_threshold})")
    ctr = [numericalize(vocab, c, args.max_caption_len) for c in tr_caps]
    refs = {s: [[tokenize(splits[s][i]["description"])] for i in cap_rows[s]] for s in ("val", "test")}
    refs["external"] = [[tokenize(r["description"])] for r in external]
    cfeats = {"train": ftr[cap_rows["train"]], "val": fva[cap_rows["val"]], "test": fte[cap_rows["test"]],
              "external": fex}

    cap_cfg = {"embed_size": 256, "hidden_size": 512, "attn_size": 256, "dropout": 0.3}
    cap = AttentionCaptioner(len(vocab), **cap_cfg).to(device)
    cap_hist, cap_best = train_captioner(cap, vocab, cfeats["train"], ctr, cfeats["val"], refs["val"], device, args)

    # 4. retrieval index (training images only) + choose default on VALIDATION ----
    idx_feats = ftr[cap_rows["train"], 0].float().mean(dim=(2, 3))
    idx_labels = lab["train"][cap_rows["train"]]
    idx_caps = tr_caps
    gen, ret, cap_res = {}, {}, {"generated": {}, "retrieved": {}, "best_epoch": cap_best, "history": cap_hist}
    for key in ("val", "test", "external"):
        if cfeats[key] is None:
            continue
        sel = cap_rows[key] if key != "external" else list(range(len(external)))
        q_labels = pred[key].argmax(1)[sel]  # the app uses the PREDICTED type
        gen[key] = decode_beam(cap, cfeats[key], vocab, device, args.beam)
        ret[key] = retrieval(idx_feats, idx_labels, idx_caps, cfeats[key].float().mean(dim=(2, 3)), q_labels)
        cap_res["generated"][key] = caption_metrics(refs[key], gen[key])
        cap_res["retrieved"][key] = caption_metrics(refs[key], [tokenize(c) for c in ret[key]])
        log(f"{key}: generated BLEU-4 {cap_res['generated'][key]['BLEU-4']:.4f} CIDEr-D "
            f"{cap_res['generated'][key]['CIDEr-D']:.4f} | retrieved BLEU-4 {cap_res['retrieved'][key]['BLEU-4']:.4f} "
            f"CIDEr-D {cap_res['retrieved'][key]['CIDEr-D']:.4f}")
    val_cider = {m: cap_res[m]["val"]["CIDEr-D"] for m in ("generated", "retrieved")}
    default_mode = "retrieve" if val_cider["retrieved"] > val_cider["generated"] else "generate"
    cap_res.update(default_mode=default_mode, val_cider=val_cider)
    log(f"default description mode (chosen on validation): {default_mode}")

    # 5. save weights in backend format ------------------------------------------
    W = out / "weights"
    torch.save({k: v.cpu() for k, v in clf.state_dict().items()}, W / "classifier.pth")
    torch.save({k: v.cpu() for k, v in cap.state_dict().items()}, W / "captioner.pth")
    with open(W / "vocab.pkl", "wb") as fh:
        pickle.dump(vocab, fh)
    np.savez_compressed(W / "caption_index.npz",
                        features=F.normalize(idx_feats, dim=1).numpy().astype(np.float16),
                        labels=idx_labels.numpy().astype(np.int8), captions=np.array(idx_caps))
    (W / "model_config.json").write_text(json.dumps({
        "version": 2, "captioner": cap_cfg, "default_caption_mode": default_mode,
        "trained_on": time.strftime("%Y-%m-%d"), "vocab_size": len(vocab)}, indent=2))

    # 6. CPU latency of the deployed pipeline ---------------------------------------
    latency = None
    try:
        torch.set_num_threads(2)
        analyzer = JewelleryAnalyzer(W, device="cpu")
        imgs = [Image.open(args.data / r["file"]).convert("RGB") for r in splits["test"][:30]]
        analyzer.analyze(imgs[0])
        times = []
        for im in imgs:
            t = time.perf_counter(); analyzer.analyze(im); times.append((time.perf_counter() - t) * 1000)
        times.sort()
        latency = {"threads": 2, "n": len(times), "median_ms": statistics.median(times),
                   "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))]}
        log(f"CPU latency median {latency['median_ms']:.0f} ms")
    except Exception as e:  # never lose the training results because of the benchmark
        log(f"latency benchmark skipped: {e}")

    # 7. reports ------------------------------------------------------------------
    def counts(rows):
        c = Counter(r["type"].capitalize() for r in rows)
        return {"Earring": c["Earring"], "Necklace": c["Necklace"], "total": len(rows)}

    M = {
        "run": {"date": time.strftime("%Y-%m-%d"), "device": dev_name, "seed": args.seed, "views": args.views,
                "minutes": (time.time() - START_TIME) / 60, "smoke": args.smoke, "args": {k: str(v) for k, v in vars(args).items()}},
        "dataset": {**{s: counts(r) for s, r in splits.items()}, "external": counts(external)},
        "classification": cls_res, "captioning": cap_res, "latency": latency,
    }
    (out / "metrics.json").write_text(json.dumps(M, indent=2, default=float))
    write_report(out, M)
    plots(out, cls_hist, cap_hist, cls_res["test"]["confusion_matrix"])

    with open(out / "predictions_test.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "true_type", "predicted_type", "confidence", "reference", "generated", "retrieved"])
        k = 0
        for i, r in enumerate(splits["test"]):
            p = pred["test"][i]
            has = i in set(cap_rows["test"])
            w.writerow([r["file"], r["type"], CLASS_NAMES[int(p.argmax())], f"{float(p.max()):.4f}", r["description"],
                        detokenize(gen["test"][k]) if has else "", ret["test"][k] if has else ""])
            k += has

    shutil.make_archive(str(out), "zip", out)
    log(f"DONE in {(time.time() - START_TIME) / 60:.1f} min -> {out}.zip")
    print("\n" + (out / "METRICS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
