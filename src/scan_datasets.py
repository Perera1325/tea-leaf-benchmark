"""Step 1: inventory every image under a root folder.

Usage:
  python src/scan_datasets.py --root /kaggle/input --out results/inventory.csv

Works with both layouts:
  <root>/<dataset>/.../<class>/<img>                       (local)
  <root>/datasets/<owner>/<dataset>/.../<class>/<img>      (Kaggle mount)
dataset = the dataset folder (e.g. tea-d9); label = the image's parent folder.

Images that are NOT original are flagged is_augmented=True with an exclude_reason
and are removed later (original images only): folder/file names containing
'aug' (augmented) or 'annotat' (annotated copies, e.g. D4 'Annotated data').
macOS junk (__MACOSX, ._files) is skipped entirely.
"""
import argparse
from pathlib import Path
import pandas as pd

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_parts(rel_parts):
    """Return (dataset, path parts below the dataset folder)."""
    if rel_parts[0] == "datasets" and len(rel_parts) >= 4:
        return rel_parts[2], rel_parts[3:]
    return rel_parts[0], rel_parts[1:]


def exclude_reason(below):
    folders = [x.lower() for x in below[:-1]]
    name = Path(below[-1]).stem.lower()
    if any("aug" in x for x in folders) or "aug" in name:
        return "augmented"
       if any("annotat" in x for x in folders):
        return "annotated"
    if "pipeline" in name or "folder_structure" in name or "folder structure" in name:
        return "diagram"
    return ""


def scan(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in EXT or not p.is_file():
            continue
        parts = p.relative_to(root).parts
        if "__MACOSX" in parts or p.name.startswith("._"):
            continue
        if len(parts) < 2:
            continue
        ds, below = split_parts(parts)
        reason = exclude_reason(below) if below else ""
        rows.append({
            "dataset": ds,
            "label": p.parent.name,
            "path": str(p),
            "is_augmented": bool(reason),
            "exclude_reason": reason,
        })
    return pd.DataFrame(rows, columns=["dataset", "label", "path", "is_augmented", "exclude_reason"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/inventory.csv")
    a = ap.parse_args()
    df = scan(Path(a.root))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    orig = df[~df["is_augmented"]]
    summary = pd.DataFrame({
        "original": orig.groupby("dataset").size(),
        "excluded": df[df["is_augmented"]].groupby("dataset").size(),
        "classes": orig.groupby("dataset")["label"].nunique(),
    }).fillna(0).astype(int)
    print(summary.to_string())
    print("\nTOTAL original:", len(orig), "| excluded:", int(df["is_augmented"].sum()))
