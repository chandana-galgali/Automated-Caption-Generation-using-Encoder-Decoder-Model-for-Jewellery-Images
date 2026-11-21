"""Create a tiny fake dataset in the prepare_dataset.py format (used by CI to test run_all.py).

    python training/make_synthetic_dataset.py /tmp/synthetic
"""

import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

METALS = ["gold", "silver", "rose gold"]
STONES = ["diamond", "ruby", "emerald", "pearl", "sapphire"]
SHAPES = ["round", "teardrop", "floral", "heart-shaped"]


def make(path, kind, seed):
    rng = random.Random(seed)
    im = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(im)
    col = tuple(rng.randint(60, 230) for _ in range(3))
    if kind == "earring":
        for x in (80, 176):
            d.ellipse([x - 25, 90, x + 25, 170], outline=col, width=6)
    else:
        d.arc([40, 20, 216, 200], 20, 160, fill=col, width=8)
        d.ellipse([110, 170, 146, 210], fill=col)
    im.save(path, "JPEG")


def main(out):
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "external_test").mkdir(exist_ok=True)
    rng = random.Random(0)
    rows = []
    for i in range(90):
        kind = "earring" if i % 3 else "necklace"
        noun = "pair of earrings" if kind == "earring" else "necklace"
        desc = f"A {rng.choice(METALS)} {noun} with {rng.choice(SHAPES)} {rng.choice(STONES)} stones."
        make(out / "images" / f"{i:05d}.jpg", kind, i)
        split = "train" if i < 60 else "val" if i < 75 else "test"
        rows.append([i, f"images/{i:05d}.jpg", desc, kind, split, i, "synthetic"])
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "file", "description", "type", "split", "group", "source"])
        w.writerows(rows)
    with open(out / "external_test" / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "description", "type"])
        for i in range(6):
            kind = "earring" if i % 2 else "necklace"
            make(out / "external_test" / f"{i}.jpg", kind, 1000 + i)
            w.writerow([f"{i}.jpg", f"A gold {kind} with diamond stones.", kind])
    print(f"synthetic dataset written to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic-dataset")
