"""Build backend/weights/caption_index.npz used for caption re-ranking.

    python training/build_caption_index.py --csv dataset/labels/augmented_dataset.csv \
        --images /path/to/augment_images --out backend/weights/caption_index.npz

Stores, for every ORIGINAL (non-augmented) training image, its L2-normalised
VGG-16 pooled feature (float16), its class label and its reference description.
"""

import argparse

import numpy as np
import torch.nn.functional as F
from torchvision import models

from common import load_table, device
from train_captioning import pooled_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="caption_index.npz")
    args = ap.parse_args()

    df = load_table(args.csv)
    if "is_augmented" in df:
        df = df[~df["is_augmented"].astype(str).str.lower().eq("true")].reset_index(drop=True)
    dev = device()
    backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(dev).eval()
    feats = F.normalize(pooled_features(df, args.images, backbone, dev), dim=1)
    np.savez_compressed(args.out, features=feats.numpy().astype(np.float16),
                        labels=df["label"].to_numpy(np.int8), captions=df["description"].to_numpy(str))
    print(f"indexed {len(df)} images -> {args.out}")


if __name__ == "__main__":
    main()
