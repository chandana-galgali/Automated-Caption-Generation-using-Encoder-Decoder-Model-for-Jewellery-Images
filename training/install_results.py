"""Install a finished training run into the repository in one step.

    python training/install_results.py path/to/run.zip      (or an unzipped run folder)

* copies the new weights into backend/weights/ (the original v1 weights are
  moved to backend/weights/v1_original/, nothing is deleted)
* copies metrics.json, METRICS.md, plots and predictions into results/
* replaces the metrics section of README.md (between the METRICS markers)
"""

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "backend" / "weights"
RESULTS = ROOT / "results"
START, END = "<!-- METRICS:START -->", "<!-- METRICS:END -->"


def main(src):
    src = Path(src)
    tmp = None
    if src.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp())
        zipfile.ZipFile(src).extractall(tmp)
        src = tmp
    if (src / "metrics.json").exists() is False:  # zip may contain a top-level folder
        hits = list(src.rglob("metrics.json"))
        if not hits:
            sys.exit("metrics.json not found - is this a run_all.py output?")
        src = hits[0].parent

    if (src / "metrics.json").read_text().find('"smoke": true') != -1:
        sys.exit("This is a --smoke test run; refusing to install its numbers.")

    legacy = WEIGHTS / "v1_original"
    for name in ("encoder.pth", "decoder.pth", "classifier.pth", "vocab.pkl", "caption_index.npz"):
        if (WEIGHTS / name).exists() and not (legacy / name).exists() and not (WEIGHTS / "captioner.pth").exists():
            legacy.mkdir(exist_ok=True)
            shutil.move(str(WEIGHTS / name), legacy / name)
    for f in (src / "weights").iterdir():
        shutil.copy2(f, WEIGHTS / f.name)
        print("weights  ->", (WEIGHTS / f.name).relative_to(ROOT))

    RESULTS.mkdir(exist_ok=True)
    for name in ("metrics.json", "METRICS.md", "training_curves.png", "confusion_matrix.png", "predictions_test.csv"):
        if (src / name).exists():
            shutil.copy2(src / name, RESULTS / name)
            print("results  ->", (RESULTS / name).relative_to(ROOT))

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    metrics = (src / "METRICS.md").read_text(encoding="utf-8").replace("`results/confusion_matrix.png`",
                                                                       "[confusion matrix](results/confusion_matrix.png)")
    metrics += "\n![Training curves](results/training_curves.png)\n"
    if START in readme and END in readme:
        a, b = readme.index(START) + len(START), readme.index(END)
        readme = readme[:a] + "\n" + metrics + readme[b:]
        (ROOT / "README.md").write_text(readme, encoding="utf-8")
        print("README.md metrics section updated")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nDone. Check `git status`, run the backend tests (cd backend && pytest -q), then commit and push.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
