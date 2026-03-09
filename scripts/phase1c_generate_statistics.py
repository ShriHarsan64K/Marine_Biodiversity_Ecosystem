#!/usr/bin/env python3
"""
PHASE 1C — Dataset Statistics Report
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Generates a comprehensive statistics report and class-distribution plot
for the cleaned, standardised dataset.

Usage:
    python src/utils/phase1c_generate_statistics.py \
        --dataset_dir dataset/standardized

Output:
    results/cleaning/statistics.json
    results/cleaning/class_distribution.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import seaborn as sns             # type: ignore


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


# ─────────────────────────────────────────────────────────────────────────────
# Statistics collection
# ─────────────────────────────────────────────────────────────────────────────

def count_class_distribution(
    dataset_dir: Path,
) -> tuple[dict[int, int], int, int]:
    """
    Count bounding-box instances and unique images per class.

    Returns:
        box_counts     {class_id: total_box_count}
        total_boxes    total bounding boxes
        total_images   total label files processed
    """
    box_counts: dict[int, int] = defaultdict(int)
    total_images = 0

    label_paths = list(dataset_dir.rglob("*.txt"))
    for label_path in label_paths:
        total_images += 1
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        box_counts[int(parts[0])] += 1
                    except ValueError:
                        pass

    return dict(box_counts), sum(box_counts.values()), total_images


def count_image_distribution(
    dataset_dir: Path,
) -> dict[int, int]:
    """
    Count unique images that contain at least one instance of each class.
    (Different from box_counts when images have multiple objects.)
    """
    image_counts: dict[int, int] = defaultdict(int)

    for label_path in dataset_dir.rglob("*.txt"):
        classes_in_image: set[int] = set()
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        classes_in_image.add(int(parts[0]))
                    except ValueError:
                        pass
        for cid in classes_in_image:
            image_counts[cid] += 1

    return dict(image_counts)


def compute_bbox_statistics(dataset_dir: Path) -> dict:
    """Compute bounding-box size statistics (width, height in normalised coords)."""
    widths:  list[float] = []
    heights: list[float] = []

    for label_path in dataset_dir.rglob("*.txt"):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        widths.append(float(parts[3]))
                        heights.append(float(parts[4]))
                    except ValueError:
                        pass

    if not widths:
        return {}

    return {
        "width":  {"mean": float(np.mean(widths)),
                   "std":  float(np.std(widths)),
                   "min":  float(np.min(widths)),
                   "max":  float(np.max(widths))},
        "height": {"mean": float(np.mean(heights)),
                   "std":  float(np.std(heights)),
                   "min":  float(np.min(heights)),
                   "max":  float(np.max(heights))},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    box_counts: dict[int, int],
    image_counts: dict[int, int],
    output_path: Path,
) -> None:
    """Create a side-by-side bar chart of box counts and image counts per class."""
    class_ids   = sorted(STANDARD_CLASSES.keys())
    class_names = [STANDARD_CLASSES[i] for i in class_ids]
    boxes       = [box_counts.get(i, 0)   for i in class_ids]
    images      = [image_counts.get(i, 0) for i in class_ids]

    x = np.arange(len(class_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))

    bars1 = ax.bar(x - width / 2, boxes,  width, label="Bounding Boxes",
                   color="#2196F3", alpha=0.87)
    bars2 = ax.bar(x + width / 2, images, width, label="Images",
                   color="#4CAF50", alpha=0.87)

    # Value labels on top of bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)

    # Target line
    target = 500
    ax.axhline(y=target, color="red", linestyle="--", linewidth=1.2,
               label=f"Min target ({target})")

    ax.set_xlabel("Indicator Family", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Phase 1C — Dataset Class Distribution (After Cleaning)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.legend(fontsize=10)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Class distribution plot saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate dataset statistics report")
    p.add_argument("--dataset_dir", default="dataset/standardized",
                   help="Standardised dataset directory (default: dataset/standardized)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║          PHASE 1C — DATASET STATISTICS REPORT                   ║")
    print("╚" + "═" * 68 + "╝")

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"\n❌ Directory not found: {dataset_dir}")
        return

    print(f"\nAnalysing: {dataset_dir} …")

    box_counts, total_boxes, total_images = count_class_distribution(dataset_dir)
    image_counts = count_image_distribution(dataset_dir)
    bbox_stats   = compute_bbox_statistics(dataset_dir)

    # Console summary
    print("\n" + "=" * 60)
    print("DATASET STATISTICS (After Cleaning)")
    print("=" * 60)
    print(f"{'Total label files':25s}: {total_images}")
    print(f"{'Total bounding boxes':25s}: {total_boxes}\n")
    print(f"{'Class':20s}  {'Boxes':>6}  {'Images':>7}  {'%':>5}")
    print("-" * 50)

    for cid in sorted(STANDARD_CLASSES.keys()):
        name   = STANDARD_CLASSES[cid]
        boxes  = box_counts.get(cid, 0)
        imgs   = image_counts.get(cid, 0)
        pct    = boxes / max(total_boxes, 1) * 100
        status = "✅" if imgs >= 500 else "⚠️ "
        print(f"  {name:18s}  {boxes:6d}  {imgs:7d}  {pct:5.1f}%  {status}")

    if box_counts:
        max_c = max(box_counts.values())
        min_c = min(box_counts.values())
        ratio = max_c / max(min_c, 1)
        print("-" * 50)
        print(f"  Balance ratio: {ratio:.2f}:1", end="  ")
        if ratio < 3:
            print("✅ Well balanced")
        elif ratio < 5:
            print("⚠️  Moderate imbalance — augmentation needed")
        else:
            print("❌ Severe imbalance — prioritise augmentation")

    print("=" * 60)

    # Save JSON
    stats_json = {
        "total_label_files": total_images,
        "total_boxes":       total_boxes,
        "box_counts":        box_counts,
        "image_counts":      image_counts,
        "balance_ratio":     round(max(box_counts.values()) /
                                   max(min(box_counts.values()), 1), 2)
                             if box_counts else 0,
        "bbox_statistics":   bbox_stats,
    }

    out_json = Path("results/cleaning/statistics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(stats_json, indent=2))
    print(f"\n✅ Statistics JSON saved: {out_json}")

    # Plot
    plot_class_distribution(
        box_counts, image_counts,
        Path("results/cleaning/class_distribution.png")
    )

    print("\nNext: Phase 1D — python src/utils/phase1d_analyze_balance.py")


if __name__ == "__main__":
    main()
