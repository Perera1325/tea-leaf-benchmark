"""Step 1: inventory every image under a root folder.

Usage:
  python src/scan_datasets.py --root /kaggle/input --out results/inventory.csv

Layout assumed: <root>/<dataset_folder>/.../<class_name>/<image>.jpg
dataset_id = first folder under root; label = image's parent folder.
Paths containing 'aug' (case-insensitive) are flagged is_augmented=True and
are EXCLUDED later (original images only).
"""
import argparse
from pathlib import Path
import pandas as pd

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in EXT or not p.is_file():
            continue
        rel = p.relative_to(root)
        parts = rel.parts
        rows.append({
            "dataset": parts[0],
            "label": p.parent.name,
            "path": str(p),
            "is_augmented": any("aug" in x.lower() for x in parts[1:-1]) or "aug" in p.stem.lower(),
        })
    return pd.DataFrame(rows, columns=["dataset", "label", "path", "is_augmented"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/inventory.csv")
    a = ap.parse_args()
    df = scan(Path(a.root))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.groupby(["dataset", "is_augmented"]).size().unstack(fill_value=0))
    print("classes per dataset:\n", df.groupby("dataset")["label"].nunique())
