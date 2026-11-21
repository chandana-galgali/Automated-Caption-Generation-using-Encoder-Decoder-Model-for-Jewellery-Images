"""Build the clean training package from the raw Dataset/ folder.

    python training/prepare_dataset.py --raw "../Dataset" --labels dataset/labels/master_labels.csv \
        --external dataset/test_set --out jewellery-dataset

Steps
1. read the master label file (one row per ORIGINAL photo, no augmented copies)
2. skip missing / unreadable images, convert to RGB on white, resize so the
   longer side is 512 px (the models use 224x224), save as JPEG
3. find exact duplicates (same pixels after resizing) and identical captions
   and put them in the same group, so a photo can never appear in two splits
4. stratified (by type) group split: 70% train / 15% validation / 15% test
5. add the 53-image external test set
Outputs <out>/images/*.jpg, <out>/labels.csv, <out>/external_test/*, <out>/README.txt
Only Pillow + NumPy are required.
"""

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

MAX_SIDE = 512


def load_rgb(path):
    im = Image.open(path)
    im.load()
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def resize(im):
    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
    return im


def fingerprint(im):
    small = np.asarray(im.convert("L").resize((16, 16), Image.BILINEAR), dtype=np.uint8)
    return hashlib.md5(small.tobytes()).hexdigest()


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw Dataset folder (earring_best/, necklace/, ...)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--external", default=None, help="folder with the external test images")
    ap.add_argument("--external-labels", default=None)
    ap.add_argument("--out", default="jewellery-dataset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    rows = list(csv.DictReader(open(args.labels, encoding="utf-8")))
    cache_path = os.path.join(args.out, ".progress.json")  # lets an interrupted run resume quickly
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    new = 0
    items, skipped = [], Counter()
    for r in rows:
        src = os.path.join(args.raw, r["image_path"])
        desc = " ".join(r["description"].split()) if r["description"].strip().lower() not in ("", "nan") else ""
        jtype = r["jewelry_type"].strip().lower()
        if jtype not in ("earring", "necklace"):  # images without a description are kept for classification
            skipped["bad label"] += 1
            continue
        if not os.path.exists(src):
            skipped["missing file"] += 1
            continue
        idx = len(items)
        name = f"{idx:05d}.jpg"
        dst = os.path.join(args.out, "images", name)
        if name in cache and os.path.exists(dst):  # resume: already converted
            items.append(dict(id=idx, file=f"images/{name}", description=desc, type=jtype,
                              source=r["image_path"], fp=cache[name]))
            continue
        im = None
        if os.path.exists(dst):
            try:
                im = Image.open(dst).convert("RGB")
            except Exception:  # truncated file from an interrupted run
                im = None
        if im is None:
            try:
                im = resize(load_rgb(src))
            except Exception:
                skipped["unreadable"] += 1
                continue
            im.save(dst + ".tmp", "JPEG", quality=92)
            os.replace(dst + ".tmp", dst)
        items.append(dict(id=idx, file=f"images/{name}", description=desc, type=jtype,
                          source=r["image_path"], fp=fingerprint(im)))
        cache[name] = items[-1]["fp"]
        new += 1
        if new % 100 == 0:
            json.dump(cache, open(cache_path, "w"))
            print(f"  processed {idx + 1} images", flush=True)
    json.dump(cache, open(cache_path, "w"))

    # group exact duplicate photos and identical descriptions
    dsu = DSU(len(items))
    for key in ("fp", "description"):
        first = {}
        for it in items:
            k = it[key].lower() if key == "description" else it[key]
            if not k:
                continue
            if k in first:
                dsu.union(it["id"], first[k])
            else:
                first[k] = it["id"]
    groups = defaultdict(list)
    for it in items:
        groups[dsu.find(it["id"])].append(it["id"])

    # stratified group split 70/15/15
    rng = random.Random(args.seed)
    split = {}
    for jtype in ("earring", "necklace"):
        gs = [g for g, ids in groups.items() if Counter(items[i]["type"] for i in ids).most_common(1)[0][0] == jtype]
        rng.shuffle(gs)
        n = sum(len(groups[g]) for g in gs)
        acc = 0
        for g in gs:
            frac = acc / n
            s = "train" if frac < 0.70 else "val" if frac < 0.85 else "test"
            for i in groups[g]:
                split[i] = s
            acc += len(groups[g])

    with open(os.path.join(args.out, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "file", "description", "type", "split", "group", "source"])
        for it in items:
            w.writerow([it["id"], it["file"], it["description"], it["type"], split[it["id"]],
                        dsu.find(it["id"]), it["source"]])

    ext = 0
    if args.external and args.external_labels:
        os.makedirs(os.path.join(args.out, "external_test"), exist_ok=True)
        files = {os.path.splitext(f)[0]: f for f in os.listdir(args.external)}
        with open(os.path.join(args.out, "external_test", "labels.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "description", "type"])
            for r in csv.DictReader(open(args.external_labels, encoding="utf-8")):
                stem = os.path.splitext(os.path.basename(r["image_path"]))[0]
                if stem not in files:
                    continue
                im = resize(load_rgb(os.path.join(args.external, files[stem])))
                im.save(os.path.join(args.out, "external_test", stem + ".jpg"), "JPEG", quality=92)
                w.writerow([stem + ".jpg", " ".join(r["description"].split()), r["jewelry_type"].strip().lower()])
                ext += 1

    c = Counter((split[i["id"]], i["type"]) for i in items)
    summary = [f"images: {len(items)}  (skipped: {dict(skipped)})",
               f"duplicate groups merged: {len(items) - len(groups)} images share a group",
               *(f"{s:5s}: earring {c[(s, 'earring')]:5d}  necklace {c[(s, 'necklace')]:5d}" for s in ("train", "val", "test")),
               f"images without a description (classification only): {sum(1 for i in items if not i['description'])}",
               f"external test images: {ext}"]
    open(os.path.join(args.out, "README.txt"), "w").write("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
