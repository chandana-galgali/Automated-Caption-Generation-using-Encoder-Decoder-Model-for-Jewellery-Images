"""Train the VGG-16 encoder + GRU decoder description model.

    python training/train_captioning.py --csv dataset/labels/augmented_dataset.csv \
        --images /path/to/augment_images --out-dir backend/weights

Same architecture and hyper-parameters as
notebooks/03_captioning_vgg16_gru_training.ipynb (embed 256, hidden 512,
vocab threshold 5, Adam 1e-3, 20 epochs, batch 32). Improvements:

* the frozen VGG-16 features are computed ONCE and cached, so each epoch only
  trains the small Linear + GRU layers (minutes instead of hours);
* validation uses held-out original photos (no augmented near-duplicates);
* the best epoch by validation loss is saved, together with vocab.pkl;
* reports teacher-forced next-word accuracy (the metric used in the report).
"""

import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

from common import ImageTable, build_vocab, group_split, load_table, numericalize, device, workers
from caption_model import EMBED_SIZE, HIDDEN_SIZE, DecoderRNN, EncoderCNN


@torch.no_grad()
def pooled_features(df, images, backbone, dev, batch=64):
    loader = DataLoader(ImageTable(df, images), batch, num_workers=workers())
    out = []
    for x, _ in loader:
        out.append(backbone(x.to(dev)).mean(dim=(2, 3)).cpu())
    return torch.cat(out)


def caption_batches(feats, caps, pad, batch, shuffle):
    order = torch.randperm(len(caps)) if shuffle else torch.arange(len(caps))
    for i in range(0, len(caps), batch):
        idx = order[i:i + batch]
        yield feats[idx], pad_sequence([caps[j] for j in idx], batch_first=True, padding_value=pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = device()

    df = load_table(args.csv)
    vocab = build_vocab(df["description"].tolist())
    print(f"vocabulary size {len(vocab)}")
    train_df, val_df = group_split(df)

    backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(dev).eval()
    print("caching VGG-16 features ...")
    ftr, fva = pooled_features(train_df, args.images, backbone, dev), pooled_features(val_df, args.images, backbone, dev)
    ctr = [torch.tensor(numericalize(vocab, d)) for d in train_df["description"]]
    cva = [torch.tensor(numericalize(vocab, d)) for d in val_df["description"]]

    encoder = EncoderCNN(HIDDEN_SIZE)
    encoder.features = backbone.cpu()  # saved inside encoder.pth, as before
    encoder.to(dev)
    decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab)).to(dev)
    pad = vocab.stoi["<PAD>"]
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad)
    opt = torch.optim.Adam(list(decoder.parameters()) + list(encoder.fc.parameters()), lr=args.lr)

    def run(feats, caps, train):
        decoder.train(train); encoder.train(train)
        tot = words = correct = 0
        with torch.set_grad_enabled(train):
            for f, c in caption_batches(feats, caps, pad, args.batch_size, train):
                f, c = f.to(dev), c.to(dev)
                logits = decoder(encoder.fc(f), c[:, :-1])
                target = c[:, 1:]
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
                mask = target != pad
                tot += loss.item() * mask.sum().item(); words += mask.sum().item()
                correct += ((logits.argmax(-1) == target) & mask).sum().item()
        return tot / words, 100 * correct / words

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, _ = run(ftr, ctr, True)
        va_loss, va_acc = run(fva, cva, False)
        print(f"epoch {epoch}/{args.epochs}  train loss {tr_loss:.4f}  val loss {va_loss:.4f}  val word acc {va_acc:.2f}%")
        if va_loss < best:
            best = va_loss
            torch.save(encoder.state_dict(), out / "encoder.pth")
            torch.save(decoder.state_dict(), out / "decoder.pth")
            with open(out / "vocab.pkl", "wb") as fh:
                pickle.dump(vocab, fh)
    print(f"saved best model (val loss {best:.4f}) to {out}")


if __name__ == "__main__":
    main()
