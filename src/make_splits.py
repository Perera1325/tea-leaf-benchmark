"""Step 3: grouped 70/15/15 split and leave-one-dataset-out (LODO) folds.

Usage:
  python src/make_splits.py --groups results/groups.csv --out results/splits.csv --seed 0
  python src/make_splits.py --groups results/groups.csv --holdout D5 --out results/lodo_D5.csv

Rules enforced:
* the split unit is group_id (near-duplicates never straddle train/val/test);
* LODO: every group containing ANY image of the held-out dataset is removed from
  train/val, so no near-duplicate of a test image can be trained on.
"""
import argparse
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def grouped_split(df: pd.DataFrame, seed=0, fractions=(0.70, 0.15, 0.15)) -> pd.Series:
    """Return a Series with 'train'/'val'/'test' for each row, split by group_id."""
    tr, va, te = fractions
    gss = GroupShuffleSplit(n_splits=1, test_size=1 - tr, random_state=seed)
    idx_tr, idx_rest = next(gss.split(df, groups=df["group_id"]))
    rest = df.iloc[idx_rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=te / (va + te), random_state=seed)
    i_va, i_te = next(gss2.split(rest, groups=rest["group_id"]))
    out = pd.Series("", index=df.index)
    out.iloc[idx_tr] = "train"
    out.loc[rest.index[i_va]] = "val"
    out.loc[rest.index[i_te]] = "test"
    return out


def lodo_split(df: pd.DataFrame, holdout: str, seed=0) -> pd.Series:
    """Held-out dataset -> 'test'. Others -> train/val (85/15) with leaked groups dropped."""
    held_groups = set(df.loc[df["dataset"] == holdout, "group_id"])
    out = pd.Series("drop", index=df.index)
    out[df["dataset"] == holdout] = "test"
    others = df[(df["dataset"] != holdout) & (~df["group_id"].isin(held_groups))]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    i_tr, i_va = next(gss.split(others, groups=others["group_id"]))
    out.loc[others.index[i_tr]] = "train"
    out.loc[others.index[i_va]] = "val"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="results/groups.csv")
    ap.add_argument("--out", default="results/splits.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", default=None, help="dataset id/folder to hold out (LODO)")
    a = ap.parse_args()
    df = pd.read_csv(a.groups)
    df["split"] = lodo_split(df, a.holdout, a.seed) if a.holdout else grouped_split(df, a.seed)
    df.to_csv(a.out, index=False)
    print(df["split"].value_counts())
