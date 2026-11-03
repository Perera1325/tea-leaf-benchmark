import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from make_splits import grouped_split, lodo_split  # noqa: E402
from dhash_groups import group_hashes  # noqa: E402


def make_df(n_groups=300, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        ds = f"D{rng.integers(1, 5)}"
        for _ in range(rng.integers(1, 4)):
            rows.append({"dataset": ds, "label": "x", "path": f"{g}_{len(rows)}.jpg", "group_id": g})
    # make some groups span two datasets
    df = pd.DataFrame(rows)
    df.loc[df["group_id"] % 10 == 0, "dataset"] = df["dataset"].where(df.index % 2 == 0, "D1")
    return df


def test_no_group_crosses_splits():
    df = make_df()
    df["split"] = grouped_split(df, seed=0)
    assert (df["split"] != "").all()
    assert df.groupby("group_id")["split"].nunique().max() == 1


def test_proportions_roughly_70_15_15():
    df = make_df(1000)
    s = grouped_split(df, seed=0).value_counts(normalize=True)
    assert abs(s["train"] - 0.70) < 0.05 and abs(s["val"] - 0.15) < 0.05


def test_lodo_has_no_leak():
    df = make_df()
    df["split"] = lodo_split(df, "D2", seed=0)
    test_groups = set(df.loc[df.split == "test", "group_id"])
    trainval_groups = set(df.loc[df.split.isin(["train", "val"]), "group_id"])
    assert not (test_groups & trainval_groups)
    assert (df.loc[df.dataset == "D2", "split"] == "test").all()


def test_dhash_grouping_hamming():
    a = 0b1010101010101010101010101010101010101010101010101010101010101010
    near = a ^ 0b11          # 2 bits different
    far = a ^ ((1 << 20) - 1)  # 20 bits different
    g = group_hashes([a, near, far], max_dist=4)
    assert g[0] == g[1] and g[0] != g[2]
