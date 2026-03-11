#!/usr/bin/env python3
"""
PHASE 1D - Class Balance Analysis & Augmentation Planner
3-Class Marine Biodiversity Project
Classes: 0=Butterflyfish, 1=Parrotfish, 2=Angelfish

Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Usage:
    python src/utils/phase1d_analyze_balance.py \
        --dataset_dir dataset/standardized \
        --target 800
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = {0: "Butterflyfish", 1: "Parrotfish", 2: "Angelfish"}


def collect_images_by_class(dataset_dir):
    images_by_class = defaultdict(list)
    for label_path in Path(dataset_dir).rglob("*.txt"):
        classes_in_label = set()
        try:
            with open(label_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            classes_in_label.add(int(parts[0]))
                        except ValueError:
                            pass
        except Exception:
            continue

        for ext in (".jpg", ".jpeg", ".png"):
            img = Path(
                str(label_path)
                .replace("/labels/", "/images/")
                .replace("\\labels\\", "\\images\\")
            ).with_suffix(ext)
            if img.exists():
                for cid in classes_in_label:
                    images_by_class[cid].append(img)
                break

    return dict(images_by_class)


def compute_plan(images_by_class, target):
    plan = {}
    print("\n" + "=" * 65)
    print(f"  CLASS BALANCE ANALYSIS  (Target: {target} per class)")
    print("=" * 65)
    print(f"  {'Class':18s}  {'Current':>8}  {'Target':>8}  {'Needed':>8}  {'Factor':>7}")
    print("-" * 65)

    for cid in range(3):
        name    = CLASS_NAMES[cid]
        current = len(images_by_class.get(cid, []))
        needed  = max(0, target - current)
        factor  = target / max(current, 1)
        flag    = "" if current >= target else "  <- needs aug"

        plan[str(cid)] = {
            "class_name":          name,
            "current_images":      current,
            "target_images":       target,
            "images_needed":       needed,
            "augmentation_factor": round(factor, 3),
            "augmentation_needed": needed > 0,
        }
        print(f"  {name:18s}  {current:8d}  {target:8d}  {needed:8d}  {factor:7.2f}x{flag}")

    counts = [len(images_by_class.get(i, [])) for i in range(3)]
    ratio  = max(counts) / max(min(counts), 1)
    print("-" * 65)
    print(f"\n  Imbalance ratio BEFORE: {ratio:.2f}x")
    print(f"  Imbalance ratio AFTER : 1.00x (perfect)")
    return plan


def plot_balance(plan, target, out_path):
    names   = [v["class_name"]     for v in plan.values()]
    current = [v["current_images"] for v in plan.values()]

    x, w = np.arange(len(names)), 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, current,        w, label="Current",           color="#FF7043", alpha=0.88)
    ax.bar(x + w/2, [target]*len(names), w, label=f"Target ({target})", color="#42A5F5", alpha=0.68)
    ax.axhline(target, color="green", linestyle="--", linewidth=1.2, label="Target line")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title("Phase 1D — Before Augmentation: Current vs Target")
    ax.set_ylabel("Image Count")
    ax.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="dataset/standardized")
    parser.add_argument("--target",      type=int, default=800)
    args = parser.parse_args()

    images_by_class = collect_images_by_class(args.dataset_dir)
    plan = compute_plan(images_by_class, args.target)

    plan_path = Path("configs/augmentation_plan.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    print(f"\nAugmentation plan saved: {plan_path}")

    plot_balance(plan, args.target, "results/cleaning/balance_comparison.png")
    print("\nNext: python src/utils/phase1d_augment_dataset.py")


if __name__ == "__main__":
    main()