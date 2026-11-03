# tea-leaf-benchmark

Offline tea leaf disease detection: a leakage-free cross-dataset benchmark, quantised TFLite models and an Android app.

Final-year project (BSc (Hons) Computer Science, University of Bedfordshire / SLIIT).
Title: *Offline Tea Leaf Disease Detection for Sri Lankan Smallholders: A Cross-Dataset Generalisation Benchmark*

## Research questions
1. How large is the accuracy drop on an unseen dataset?
2. Which strategies (multi-source training, stronger backbone, background removal) restore robustness?
3. What does quantisation cost on a real phone?
4. Can temperature scaling and a validation-chosen reject threshold make errors visible?

## Pipeline (reproducible, fixed seeds)
```
python src/scan_datasets.py --root data --out results/inventory.csv      # 1. inventory (flags augmented copies)
python src/dhash_groups.py  --inventory results/inventory.csv --out results/groups.csv   # 2. near-duplicate groups
python src/make_splits.py   --groups results/groups.csv --out results/splits.csv --seed 0   # 3. grouped 70/15/15
python src/make_splits.py   --groups results/groups.csv --holdout D5 --out results/lodo_D5.csv   # leave-one-dataset-out
pytest -q
```
On Kaggle use `--root /kaggle/input`.

## Rules
- Original images only (augmented copies excluded).
- Split unit = near-duplicate group; no group crosses train/val/test.
- LODO removes every group touching the held-out dataset from train/val.
- One frozen final-test dataset never influences model selection, calibration or the reject threshold.

## Datasets
See `dataset_registry.csv` (version, DOI, licence). Data is not stored in this repository.

## Status
- [x] Repo scaffold, dataset registry, dHash grouping, grouped and LODO splits, tests
- [ ] Download datasets and verify counts/licences
- [ ] Label map
- [ ] Baselines (MobileNetV3-Large, EfficientNet-B0, 3 seeds)
- [ ] Calibration and reject threshold
- [ ] TFLite fp32/fp16/int8 and Android app
