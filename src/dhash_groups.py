"""Step 2: perceptual-hash (dHash) near-duplicate grouping.

Usage:
  python src/dhash_groups.py --inventory results/inventory.csv --out results/groups.csv

Only non-augmented images are hashed. Images whose 64-bit dHash differs by at most
--max-dist bits (default 4) are put in the same group, ACROSS datasets too, so a
group never straddles train and test. Also writes results/duplicate_report.csv
(number of cross-dataset duplicate pairs per dataset pair).
"""
import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import imagehash


def dhash_int(path: str) -> int:
    with Image.open(path) as im:
        h = imagehash.dhash(im.convert("RGB"), hash_size=8)
    return int("".join("1" if b else "0" for b in h.hash.flatten()), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def group_hashes(hashes, max_dist=4):
    """Return group id per hash. Pigeonhole: if distance<=3 some 16-bit chunk is equal;
    for max_dist up to 7 we use 8 chunks of 8 bits (distance<=7 -> some chunk equal)."""
    n = len(hashes)
    dsu = DSU(n)
    chunks = 8
    width = 64 // chunks
    for c in range(chunks):
        shift = c * width
        buckets = defaultdict(list)
        for i, h in enumerate(hashes):
            buckets[(h >> shift) & ((1 << width) - 1)].append(i)
        for idx in buckets.values():
            if len(idx) < 2 or len(idx) > 5000:
                continue
            for i, j in combinations(idx, 2):
                if hamming(hashes[i], hashes[j]) <= max_dist:
                    dsu.union(i, j)
    roots = {}
    return np.array([roots.setdefault(dsu.find(i), len(roots)) for i in range(n)])


def main(inv_path, out_path, max_dist):
    inv = pd.read_csv(inv_path)
    inv = inv[~inv["is_augmented"]].reset_index(drop=True)
    hashes = []
    ok = []
    for p in inv["path"]:
        try:
            hashes.append(dhash_int(p))
            ok.append(True)
        except Exception as e:  # unreadable file -> reported, dropped
            print("unreadable:", p, e)
            hashes.append(0)
            ok.append(False)
    inv["readable"] = ok
    inv["dhash"] = hashes
    inv = inv[inv["readable"]].reset_index(drop=True)
    inv["group_id"] = group_hashes(inv["dhash"].tolist(), max_dist)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    inv.drop(columns=["readable"]).to_csv(out_path, index=False)

    # cross-dataset duplicate report
    ds_per_group = inv.groupby("group_id")["dataset"].agg(lambda s: sorted(set(s)))
    rep = defaultdict(int)
    for ds in ds_per_group:
        for a, b in combinations(ds, 2):
            rep[(a, b)] += 1
    rep_df = pd.DataFrame([{"dataset_a": a, "dataset_b": b, "shared_groups": n} for (a, b), n in rep.items()])
    rep_df.to_csv(Path(out_path).with_name("duplicate_report.csv"), index=False)
    print(f"{len(inv)} images -> {inv.group_id.nunique()} groups; "
          f"{int((ds_per_group.map(len) > 1).sum())} groups span >1 dataset")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="results/inventory.csv")
    ap.add_argument("--out", default="results/groups.csv")
    ap.add_argument("--max-dist", type=int, default=4)
    a = ap.parse_args()
    main(a.inventory, a.out, a.max_dist)
