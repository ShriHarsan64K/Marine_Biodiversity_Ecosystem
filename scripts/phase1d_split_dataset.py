#!/usr/bin/env python3
"""
PHASE 1D — Stratified Train / Val / Test Split
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Splits the augmented dataset into Train (85%) / Val (8%) / Test (7%)
with stratified sampling so each split has proportional class coverage.

Usage:
    python src/utils/phase1d_split_dataset.py \
        --input_dir  dataset/augmented  \
        --output_dir dataset/processed

Output:
    dataset/processed/
        train/images/   train/labels/
        val/images/     val/labels/
        test/images/    test/labels/
    results/cleaning/split_statistics.txt
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STANDARD_CLASSES: dict[int, str] = {
    0: "Butterflyfish",
    1: "Grouper",
    2: "Parrotfish",
    3: "Surgeonfish",
    4: "Damselfish",
    5: "Wrasse",
    6: "Triggerfish",
    7: "Angelfish",
}

TRAIN_RATIO = 0.85
VAL_RATIO   = 0.08
TEST_RATIO  = 0.07


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_primary_class(label_path: Path) -> int | None:
    """Return the most frequent class ID in a label file."""
    counts: dict[int, int] = defaultdict(int)
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    counts[int(parts[0])] += 1
    except Exception:
        return None
    return max(counts, key=lambda k: counts[k]) if counts else None


def find_image(label_path: Path) -> Path | None:
    """Find the image corresponding to a label file."""
    for ext in (".jpg", ".jpeg", ".png"):
        img = Path(
            str(label_path)
            .replace("/labels/", "/images/")
            .replace("\\labels\\", "\\images\\")
        ).with_suffix(ext)
        if img.exists():
            return img
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Stratified split
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(
    input_dir: Path,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    seed:        int   = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Build stratified train / val / test lists of label paths.

    Returns (train_labels, val_labels, test_labels)
    """
    random.seed(seed)

    # Group labels by primary class
    by_class: dict[int, list[Path]] = defaultdict(list)
    unclassified: list[Path] = []

    label_paths = list(input_dir.rglob("*.txt"))
    print(f"\nFound {len(label_paths)} label files in {input_dir}")

    for lp in label_paths:
        cid = get_primary_class(lp)
        if cid is not None:
            by_class[cid].append(lp)
        else:
            unclassified.append(lp)

    train_list: list[Path] = []
    val_list:   list[Path] = []
    test_list:  list[Path] = []

    for cid, paths in sorted(by_class.items()):
        random.shuffle(paths)
        n      = len(paths)
        n_val  = max(1, round(n * val_ratio))
        n_test = max(1, round(n * TEST_RATIO))
        n_train = n - n_val - n_test

        train_list.extend(paths[:n_train])
        val_list.extend(paths[n_train:n_train + n_val])
        test_list.extend(paths[n_train + n_val:])

    # Distribute unclassified proportionally
    random.shuffle(unclassified)
    n_u = len(unclassified)
    u_v = round(n_u * val_ratio)
    u_t = round(n_u * TEST_RATIO)
    train_list.extend(unclassified[u_v + u_t:])
    val_list.extend(unclassified[:u_v])
    test_list.extend(unclassified[u_v:u_v + u_t])

    return train_list, val_list, test_list


# ─────────────────────────────────────────────────────────────────────────────
# Copy split to output directories
# ─────────────────────────────────────────────────────────────────────────────

def copy_split(
    label_paths: list[Path],
    split_dir: Path,
    split_name: str,
) -> int:
    """Copy labels and images to split_dir/{labels,images}."""
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_images = 0

    for lp in tqdm(label_paths, desc=f"Copying {split_name}"):
        img = find_image(lp)
        if img is None:
            missing_images += 1
            continue
        shutil.copy2(lp,  split_dir / "labels" / lp.name)
        shutil.copy2(img, split_dir / "images"  / img.name)
        copied += 1

    if missing_images:
        print(f"  ⚠️  {missing_images} labels had no matching image")

    return copied


# ─────────────────────────────────────────────────────────────────────────────
# data.yaml
# ─────────────────────────────────────────────────────────────────────────────

def create_data_yaml(output_dir: Path) -> None:
    """Write a YOLOv8-compatible data.yaml to output_dir."""
    import yaml  # type: ignore

    data = {
        "path":  str(output_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(STANDARD_CLASSES),
        "names": {k: v for k, v in STANDARD_CLASSES.items()},
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ data.yaml written: {yaml_path}")
    print("   (Update 'path' if you move the dataset folder)")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics report
# ─────────────────────────────────────────────────────────────────────────────

def save_split_report(
    train: list[Path],
    val:   list[Path],
    test:  list[Path],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(train) + len(val) + len(test)

    by_class: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "val":   defaultdict(int),
        "test":  defaultdict(int),
    }
    for split_name, paths in [("train", train), ("val", val), ("test", test)]:
        for lp in paths:
            cid = get_primary_class(lp)
            if cid is not None:
                by_class[split_name][STANDARD_CLASSES.get(cid, str(cid))] += 1

    with open(output_path, "w") as f:
        f.write("DATASET SPLIT STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total images : {total}\n")
        f.write(f"Train        : {len(train)} ({len(train)/max(total,1)*100:.1f}%)\n")
        f.write(f"Val          : {len(val)}   ({len(val)/max(total,1)*100:.1f}%)\n")
        f.write(f"Test         : {len(test)}  ({len(test)/max(total,1)*100:.1f}%)\n\n")
        f.write("Per-class distribution:\n")
        f.write(f"{'Class':20s}  {'Train':>7}  {'Val':>6}  {'Test':>6}\n")
        f.write("-" * 50 + "\n")
        for name in STANDARD_CLASSES.values():
            tr = by_class["train"].get(name, 0)
            vl = by_class["val"].get(name, 0)
            ts = by_class["test"].get(name, 0)
            f.write(f"  {name:18s}  {tr:7d}  {vl:6d}  {ts:6d}\n")

    print(f"✅ Split statistics saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified train/val/test split")
    p.add_argument("--input_dir",  default="dataset/augmented")
    p.add_argument("--output_dir", default="dataset/processed")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║         PHASE 1D — STRATIFIED TRAIN / VAL / TEST SPLIT          ║")
    print("╚" + "═" * 68 + "╝")

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"\n❌ Input directory not found: {input_dir}")
        print("   Run phase1d_augment_dataset.py first")
        return

    train, val, test = stratified_split(input_dir, seed=args.seed)
    total = len(train) + len(val) + len(test)

    print(f"\nSplit summary:")
    print(f"  Train : {len(train):5d}  ({len(train)/total*100:.1f}%)")
    print(f"  Val   : {len(val):5d}  ({len(val)/total*100:.1f}%)")
    print(f"  Test  : {len(test):5d}  ({len(test)/total*100:.1f}%)")

    tr = copy_split(train, output_dir / "train", "train")
    vl = copy_split(val,   output_dir / "val",   "val")
    ts = copy_split(test,  output_dir / "test",  "test")

    create_data_yaml(output_dir)
    save_split_report(train, val, test,
                      Path("results/cleaning/split_statistics.txt"))

    print("\n" + "=" * 70)
    print(f"  ✅ Dataset split complete!")
    print(f"  Train: {tr}  |  Val: {vl}  |  Test: {ts}")
    print(f"  Output: {output_dir.resolve()}")
    print("=" * 70)
    print("\n🎉 PHASE 1 COMPLETE — Dataset ready for Phase 2 (Enhancement)")
    print("   Next: src/enhancement/phase2_ancuti_fusion.py")


if __name__ == "__main__":
    main()
