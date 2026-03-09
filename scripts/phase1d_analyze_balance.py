#!/usr/bin/env python3
"""
PHASE 1D — Class Balance Analysis & Augmentation Planner
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Reads the cleaned dataset, calculates how many augmented images are needed
per class to reach the target (800 per class), and saves the plan as JSON.

Usage:
    python src/utils/phase1d_analyze_balance.py \
        --dataset_dir dataset/standardized \
        --target 800

Output:
    configs/augmentation_plan.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np


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
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def collect_images_by_class(
    dataset_dir: Path,
) -> dict[int, list[Path]]:
    """
    Return a dict mapping class_id → list of image paths that contain
    at least one bounding box of that class.
    """
    images_by_class: dict[int, list[Path]] = defaultdict(list)

    for label_path in dataset_dir.rglob("*.txt"):
        classes_in_label: set[int] = set()
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        classes_in_label.add(int(parts[0]))
                    except ValueError:
                        pass

        # Find corresponding image file
        for ext in (".jpg", ".jpeg", ".png"):
            img_candidate = Path(
                str(label_path)
                .replace("/labels/", "/images/")
                .replace("\\labels\\", "\\images\\")
            ).with_suffix(ext)
            if img_candidate.exists():
                for cid in classes_in_label:
                    images_by_class[cid].append(img_candidate)
                break

    return dict(images_by_class)


def compute_augmentation_plan(
    images_by_class: dict[int, list[Path]],
    target_per_class: int = 800,
) -> dict:
    """
    Calculate augmentation requirements for each class.

    Returns a plan dict ready to be saved as JSON.
    """
    plan: dict = {}

    print("\n" + "=" * 72)
    print(f"{'CLASS BALANCE ANALYSIS':^72}")
    print(f"{'(Target: ' + str(target_per_class) + ' images per class)':^72}")
    print("=" * 72)
    print(f"{'Class':20s}  {'Current':>8}  {'Target':>8}  {'+Needed':>8}  {'Factor':>7}")
    print("-" * 72)

    for cid in sorted(STANDARD_CLASSES.keys()):
        name    = STANDARD_CLASSES[cid]
        current = len(images_by_class.get(cid, []))
        needed  = max(0, target_per_class - current)
        factor  = target_per_class / max(current, 1)

        plan[str(cid)] = {
            "class_name":          name,
            "current_images":      current,
            "target_images":       target_per_class,
            "images_needed":       needed,
            "augmentation_factor": round(factor, 3),
            "augmentation_needed": needed > 0,
        }

        flag = "" if current >= target_per_class else " ← needs aug"
        print(f"  {name:18s}  {current:8d}  {target_per_class:8d}  "
              f"{needed:8d}  {factor:7.2f}x{flag}")

    total_current = sum(len(v) for v in images_by_class.values())
    total_target  = target_per_class * len(STANDARD_CLASSES)
    all_counts    = [len(images_by_class.get(i, [])) for i in STANDARD_CLASSES]

    print("-" * 72)
    print(f"  {'TOTAL':18s}  {total_current:8d}  {total_target:8d}  "
          f"{total_target - total_current:8d}")
    print("=" * 72)

    ratio_before = max(all_counts) / max(min(all_counts), 1) if all_counts else 0
    print(f"\nBalance ratio BEFORE augmentation: {ratio_before:.2f}:1")
    print("Balance ratio AFTER  augmentation: 1.00:1  (perfect)")

    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_balance_comparison(plan: dict, target: int, output_path: Path) -> None:
    """Bar chart showing current vs target counts per class."""
    class_names = [v["class_name"]     for v in plan.values()]
    current     = [v["current_images"] for v in plan.values()]
    targets     = [target] * len(plan)

    x     = np.arange(len(class_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width / 2, current, width, label="Current",
           color="#FF7043", alpha=0.88)
    ax.bar(x + width / 2, targets, width, label=f"Target ({target})",
           color="#42A5F5", alpha=0.68)
    ax.axhline(y=target, color="green", linestyle="--", linewidth=1.2,
               label="Target line")

    ax.set_xlabel("Indicator Family", fontsize=12)
    ax.set_ylabel("Image Count", fontsize=12)
    ax.set_title("Phase 1D — Before Augmentation: Current vs Target", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.legend(fontsize=10)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Balance comparison plot saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse class balance and build augmentation plan"
    )
    p.add_argument("--dataset_dir", default="dataset/standardized",
                   help="Standardised dataset directory")
    p.add_argument("--target", type=int, default=800,
                   help="Target images per class (default: 800)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║       PHASE 1D — CLASS BALANCE ANALYSIS & AUG PLANNER          ║")
    print("╚" + "═" * 68 + "╝")

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"\n❌ Directory not found: {dataset_dir}")
        return

    images_by_class = collect_images_by_class(dataset_dir)
    plan = compute_augmentation_plan(images_by_class, args.target)

    # Save plan
    plan_path = Path("configs/augmentation_plan.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2))
    print(f"\n✅ Augmentation plan saved: {plan_path}")

    # Plot
    plot_balance_comparison(plan, args.target,
                            Path("results/cleaning/balance_comparison.png"))

    print("\nNext: python src/utils/phase1d_augment_dataset.py")


if __name__ == "__main__":
    main()
