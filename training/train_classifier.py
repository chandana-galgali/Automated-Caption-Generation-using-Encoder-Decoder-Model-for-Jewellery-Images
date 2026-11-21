"""Train the VGG-16 + GRU jewellery-type classifier.

    python training/train_classifier.py --csv dataset/labels/augmented_dataset.csv \
        --images /path/to/augment_images --out backend/weights/classifier.pth

Matches notebooks/02_classification_vgg16_gru_training.ipynb (frozen ImageNet
VGG-16, GRU hidden 256, Adam lr 1e-3, batch 16, 10 epochs) but validates on
held-out ORIGINAL images and keeps the best epoch.
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from common import ImageTable, group_split, load_table, device, workers
from caption_model import CLASS_NAMES, VGG16_GRU_Classifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True, help="root folder that image_path is relative to")
    ap.add_argument("--out", default="classifier.pth")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    dev = device()
    train_df, val_df = group_split(load_table(args.csv))
    print(f"train {len(train_df)} images, validation {len(val_df)} images (grouped by original photo)")
    train = DataLoader(ImageTable(train_df, args.images), args.batch_size, shuffle=True, num_workers=workers())
    val = DataLoader(ImageTable(val_df, args.images), args.batch_size, num_workers=workers())

    model = VGG16_GRU_Classifier(len(CLASS_NAMES))
    model.features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
    for p in model.features.parameters():
        p.requires_grad = False
    model.to(dev)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loss_fn, best = nn.CrossEntropyLoss(), 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)

        model.eval()
        correct = n = 0
        with torch.no_grad():
            for x, y in val:
                correct += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
                n += len(y)
        acc = 100 * correct / n
        print(f"epoch {epoch}/{args.epochs}  loss {total / len(train.dataset):.4f}  val acc {acc:.2f}%")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), args.out)
    print(f"best validation accuracy {best:.2f}% -> {args.out}")


if __name__ == "__main__":
    main()
