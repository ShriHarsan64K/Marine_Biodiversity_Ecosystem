"""
Phase 1C: Generate Dataset Statistics
3-Class Marine Biodiversity Project
Classes: 0=Butterflyfish, 1=Parrotfish, 2=Angelfish

Author: Shri Harsan M | M.Tech Data Science | SRM Institute
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

CLASS_NAMES = {0: "Butterflyfish", 1: "Parrotfish", 2: "Angelfish"}
TARGET_PER_CLASS = 800


def generate_statistics(dataset_dir):
    dataset_dir = Path(dataset_dir)
    stats = {
        "splits": {},
        "class_totals": defaultdict(int),
        "total_images": 0,
        "total_boxes": 0,
    }

    for split in ["train", "valid", "test"]:
        lbl_dir = dataset_dir / split / "labels"
        img_dir = dataset_dir / split / "images"
        if not lbl_dir.exists():
            continue

        split_stats = {
            "images": len(list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png"))) if img_dir.exists() else 0,
            "boxes": 0,
            "class_counts": defaultdict(int),
        }

        for lbl_file in lbl_dir.rglob("*.txt"):
            try:
                with open(lbl_file, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls = int(parts[0])
                            split_stats["class_counts"][cls] += 1
                            split_stats["boxes"] += 1
                            stats["class_totals"][cls] += 1
                            stats["total_boxes"] += 1
            except Exception:
                continue

        stats["splits"][split] = split_stats
        stats["total_images"] += split_stats["images"]

    return stats


def print_report(stats):
    print("\n" + "=" * 60)
    print("DATASET STATISTICS REPORT — 3-CLASS PROJECT")
    print("=" * 60)
    print(f"\nTotal Images : {stats['total_images']}")
    print(f"Total Boxes  : {stats['total_boxes']}")

    print("\n--- Per Split ---")
    for split, s in stats["splits"].items():
        print(f"\n{split.upper()}")
        print(f"  Images : {s['images']}")
        print(f"  Boxes  : {s['boxes']}")
        for cid in range(3):
            cnt = s["class_counts"].get(cid, 0)
            print(f"    {cid} {CLASS_NAMES[cid]:15s}: {cnt}")

    print("\n--- Overall Class Totals ---")
    for cid in range(3):
        cnt = stats["class_totals"].get(cid, 0)
        pct = (cnt / stats["total_boxes"] * 100) if stats["total_boxes"] > 0 else 0
        gap = max(0, TARGET_PER_CLASS - cnt)
        bar = "#" * min(cnt // 10, 40)
        status = "OK" if cnt >= TARGET_PER_CLASS else f"NEED +{gap} more"
        print(f"  {cid} {CLASS_NAMES[cid]:15s}: {cnt:5d} ({pct:4.1f}%)  [{status}]  {bar}")

    counts = [stats["class_totals"].get(i, 0) for i in range(3)]
    max_c, min_c = max(counts), min(counts) if min(counts) > 0 else 1
    ratio = max_c / min_c
    print(f"\nImbalance ratio (max/min): {ratio:.2f}x")
    if ratio > 3:
        print("WARNING: Highly imbalanced. Phase 1D augmentation will fix this.")
    elif ratio > 1.5:
        print("NOTICE: Slight imbalance. Phase 1D will handle this.")
    else:
        print("Dataset is well balanced!")

    print("\n--- Phase 1D Readiness ---")
    for cid in range(3):
        cnt = stats["class_totals"].get(cid, 0)
        if cnt == 0:
            print(f"  Class {cid} ({CLASS_NAMES[cid]}): MISSING!")
        elif cnt < TARGET_PER_CLASS:
            print(f"  Class {cid} ({CLASS_NAMES[cid]}): {cnt} → augment to {TARGET_PER_CLASS}")
        else:
            print(f"  Class {cid} ({CLASS_NAMES[cid]}): {cnt} → READY")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="dataset/standardized")
    parser.add_argument("--target", type=int, default=800)
    args = parser.parse_args()

    global TARGET_PER_CLASS
    TARGET_PER_CLASS = args.target

    print(f"Analyzing: {args.dataset_dir}")
    stats = generate_statistics(args.dataset_dir)
    print_report(stats)

    out_path = Path("results") / "phase1c_statistics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_images": stats["total_images"],
            "total_boxes": stats["total_boxes"],
            "class_totals": {str(k): v for k, v in stats["class_totals"].items()},
        }, f, indent=2)
    print(f"\nStats saved to: {out_path}")


if __name__ == "__main__":
    main()