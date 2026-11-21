"""Evaluate the deployed pipeline on the external test set (dataset/test_set).

    python training/evaluate.py                       # both caption modes
    python training/evaluate.py --mode generate       # GRU generation only

Reports classifier accuracy, BLEU-4 (nltk, smoothing method1) and latency,
and writes per-image results to evaluation_results.csv.
"""

import argparse
import statistics
import time
from pathlib import Path

import pandas as pd
from PIL import Image

from common import word_tokenize
from caption_model import JewelleryAnalyzer

ROOT = Path(__file__).resolve().parents[1]


def bleu4(ref, hyp):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    return sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default=ROOT / "dataset" / "test_set", type=Path)
    ap.add_argument("--weights", default=ROOT / "backend" / "weights", type=Path)
    ap.add_argument("--mode", choices=["auto", "generate", "both"], default="both")
    args = ap.parse_args()

    analyzer = JewelleryAnalyzer(args.weights, caption_index=args.weights / "caption_index.npz")
    table = pd.read_excel(args.test_dir / "test_descriptions.xlsx")
    files = {p.stem: p for p in args.test_dir.iterdir() if p.suffix.lower() != ".xlsx"}
    modes = ["generate", "auto"] if args.mode == "both" else [args.mode]

    rows = []
    for _, r in table.iterrows():
        path = files[Path(str(r["image_path"])).stem]
        image = Image.open(path).convert("RGB")
        ref = word_tokenize(str(r["description"]).lower())
        row = {"image": path.name, "true_type": str(r["jewelry_type"]).strip().capitalize()}
        for mode in modes:
            t = time.perf_counter()
            out = analyzer.analyze(image, mode)
            row[f"{mode}_ms"] = (time.perf_counter() - t) * 1000
            row["pred_type"] = out["type"]
            row[f"{mode}_description"] = out["description"]
            row[f"{mode}_bleu4"] = bleu4(ref, word_tokenize(out["description"].lower()))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("evaluation_results.csv", index=False)
    print(f"images: {len(df)}")
    print(f"classification accuracy: {(df.pred_type == df.true_type).mean():.2%}")
    for mode in modes:
        print(f"[{mode}] BLEU-4 {df[f'{mode}_bleu4'].mean():.4f}   "
              f"median latency {statistics.median(df[f'{mode}_ms']):.0f} ms")


if __name__ == "__main__":
    main()
