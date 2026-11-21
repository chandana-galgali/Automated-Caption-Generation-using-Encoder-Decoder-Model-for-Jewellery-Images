"""Shared data utilities for the training scripts.

The augmented CSV (dataset/labels/augmented_dataset.csv) has one row per image
(original + augmented copies) with columns:
    image_path, description, jewelry_type, is_augmented, original_image, source_folder
"""

import os
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from caption_model import CLASS_NAMES, Vocabulary, transform  # noqa: E402,F401

try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    word_tokenize = nltk.word_tokenize
except Exception:  # pragma: no cover - fall back to the backend tokenizer
    from caption_model import tokenize as word_tokenize


def build_vocab(sentences, freq_threshold=5):
    """Same algorithm as the original notebook (frequency threshold, insertion order)."""
    vocab = Vocabulary(freq_threshold)
    vocab.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
    vocab.stoi = {v: k for k, v in vocab.itos.items()}
    counts, idx = Counter(), 4
    for s in sentences:
        for w in word_tokenize(s.lower()):
            counts[w] += 1
            if counts[w] == freq_threshold:
                vocab.stoi[w], vocab.itos[idx] = idx, w
                idx += 1
    return vocab


def numericalize(vocab, text):
    unk = vocab.stoi["<UNK>"]
    return [vocab.stoi["<START>"]] + [vocab.stoi.get(w, unk) for w in word_tokenize(text.lower())] + [vocab.stoi["<END>"]]


def load_table(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=["description"])
    df["description"] = df["description"].str.strip()
    df["label"] = df["jewelry_type"].str.lower().map({c.lower(): i for i, c in enumerate(CLASS_NAMES)})
    group = df["original_image"] if "original_image" in df else df["image_path"]
    folder = df["source_folder"].astype(str) if "source_folder" in df else ""
    df["group"] = folder + "/" + group.astype(str)
    return df


def group_split(df, val_fraction=0.2, seed=42):
    """Split by ORIGINAL image so augmented copies never leak between train and validation.

    (The original notebooks used random_split over augmented rows, which puts
    near-duplicates of validation images in the training set and inflates accuracy.)
    """
    groups = sorted(df["group"].unique())
    random.Random(seed).shuffle(groups)
    val_groups = set(groups[: int(len(groups) * val_fraction)])
    is_val = df["group"].isin(val_groups)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


class ImageTable(Dataset):
    def __init__(self, df, image_root, vocab=None):
        self.df, self.root, self.vocab = df, Path(image_root), vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = transform(Image.open(self.root / row["image_path"]).convert("RGB"))
        if self.vocab is None:
            return image, int(row["label"])
        return image, torch.tensor(numericalize(self.vocab, row["description"]))


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def workers():
    return min(4, os.cpu_count() or 1)
