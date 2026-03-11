#!/usr/bin/env python3
"""
PHASE 1D - Stratified Train / Val / Test Split
3-Class Marine Biodiversity Project
Classes: 0=Butterflyfish, 1=Parrotfish, 2=Angelfish

Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Usage:
    python src/utils/phase1d_split_dataset.py \
        --input_dir  dataset/augmented \
        --output_dir dataset/processed
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

CLASS_NAMES = {0: "Butterflyfish", 1: "Parrotfish", 2: "Angelfish"}

TRAIN_RATIO = 0.85
VAL_RATIO   = 0.08
TEST_RATIO  = 0.07


def get_primary_class(label_path):
    counts = defaultdict(int)
    try:
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        counts[int(parts[0])] += 1
                    except ValueError:
                        pass
    except Exception:
        return None
    return max(counts, key=lambda k: counts[k]) if counts else None


def find_image(label_path):
    for ext in (".jpg", ".jpeg", ".png"):
        img = Path(
            str(label_path)
            .replace("/labels/", "/images/")
            .replace("\\labels\\", "\\images\\")
        ).with_suffix(ext)
        if img.exists():
            return img
    return None


def stratified_split(input_dir, seed=42):
    random.seed(seed)
    by_class  = defaultdict(list)
    unclassed = []

    label_paths = list(Path(input_dir).rglob("*.txt"))
    print(f"\nFound {len(label_paths)} label files")

    for lp in label_paths:
        cid = get_primary_class(lp)
        if cid is not None:
            by_class[cid].append(lp)
        else:
            unclassed.append(lp)

    train_list, val_list, test_list = [], [], []

    for cid, paths in sorted(by_class.items()):
        random.shuffle(paths)
        n       = len(paths)
        n_val   = max(1, round(n * VAL_RATIO))
        n_test  = max(1, round(n * TEST_RATIO))
        n_train = n - n_val - n_test
        train_list.extend(paths[:n_train])
        val_list.extend(paths[n_train:n_train + n_val])
        test_list.extend(paths[n_train + n_val:])

    # Distribute unclassified
    random.shuffle(unclassed)
    n_u = len(unclassed)
    u_v = round(n_u * VAL_RATIO)
    u_t = round(n_u * TEST_RATIO)
    train_list.extend(unclassed[u_v + u_t:])
    val_list.extend(unclassed[:u_v])
    test_list.extend(unclassed[u_v:u_v + u_t])

    return train_list, val_list, test_list


def copy_split(label_paths, split_dir, split_name):
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    copied = missing = 0
    for lp in tqdm(label_paths, desc=f"  Copying {split_name}"):
        img = find_image(lp)
        if img is None:
            missing += 1
            continue
        shutil.copy2(lp,  split_dir / "labels" / lp.name)
        shutil.copy2(img, split_dir / "images"  / img.name)
        copied += 1
    if missing:
        print(f"  WARNING: {missing} labels had no matching image")
    return copied


def write_data_yaml(output_dir):
    yaml_content = f"""path: {Path(output_dir).resolve()}
train: train/images
val: val/images
test: test/images

nc: 3
names: ['Butterflyfish', 'Parrotfish', 'Angelfish']
"""
    yaml_path = Path(output_dir) / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\ndata.yaml written: {yaml_path}")


def save_report(train, val, test, out_path):
    total = len(train) + len(val) + len(test)
    by_class = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}

    for split_name, paths in [("train", train), ("val", val), ("test", test)]:
        for lp in paths:
            cid = get_primary_class(lp)
            if cid is not None:
                by_class[split_name][CLASS_NAMES.get(cid, str(cid))] += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("DATASET SPLIT STATISTICS — 3-CLASS PROJECT\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Total  : {total}\n")
        f.write(f"Train  : {len(train)} ({len(train)/max(total,1)*100:.1f}%)\n")
        f.write(f"Val    : {len(val)}   ({len(val)/max(total,1)*100:.1f}%)\n")
        f.write(f"Test   : {len(test)}  ({len(test)/max(total,1)*100:.1f}%)\n\n")
        f.write(f"{'Class':20s}  {'Train':>7}  {'Val':>6}  {'Test':>6}\n")
        f.write("-" * 48 + "\n")
        for name in CLASS_NAMES.values():
            tr = by_class["train"].get(name, 0)
            vl = by_class["val"].get(name, 0)
            ts = by_class["test"].get(name, 0)
            f.write(f"  {name:18s}  {tr:7d}  {vl:6d}  {ts:6d}\n")

    print(f"Split report saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  default="dataset/augmented")
    parser.add_argument("--output_dir", default="dataset/processed")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: {input_dir} not found. Run phase1d_augment_dataset.py first.")
        return

    train, val, test = stratified_split(input_dir, seed=args.seed)
    total = len(train) + len(val) + len(test)

    print(f"\nSplit plan:")
    print(f"  Train : {len(train):5d}  ({len(train)/total*100:.1f}%)")
    print(f"  Val   : {len(val):5d}  ({len(val)/total*100:.1f}%)")
    print(f"  Test  : {len(test):5d}  ({len(test)/total*100:.1f}%)")

    tr = copy_split(train, output_dir / "train", "train")
    vl = copy_split(val,   output_dir / "val",   "val")
    ts = copy_split(test,  output_dir / "test",  "test")

    write_data_yaml(output_dir)
    save_report(train, val, test, "results/cleaning/split_statistics.txt")

    print("\n" + "=" * 55)
    print(f"  Train: {tr}  |  Val: {vl}  |  Test: {ts}")
    print(f"  Output: {output_dir.resolve()}")
    print("=" * 55)
    print("\nPHASE 1 COMPLETE! Ready for Phase 2 (Enhancement)")
    print("Next: src/enhancement/phase2_ancuti_fusion.py")


if __name__ == "__main__":
    main()