#!/usr/bin/env python3
"""
PHASE 1C — Duplicate Detection & Removal
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Uses perceptual hashing (pHash) to find near-duplicate images across all
collected sources and removes them to prevent data leakage.

Usage:
    # Preview only (safe — does NOT delete anything):
    python src/utils/phase1c_remove_duplicates.py --input_dir dataset/raw --dry_run

    # Actually remove duplicates:
    python src/utils/phase1c_remove_duplicates.py --input_dir dataset/raw

Output:
    results/cleaning/duplicate_report.txt
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image  # type: ignore
import imagehash      # type: ignore
from tqdm import tqdm  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Hash computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_phash(image_path: Path, hash_size: int = 8) -> imagehash.ImageHash | None:
    """
    Compute the perceptual hash of an image.

    Args:
        image_path: Path to JPG or PNG
        hash_size:  Hash grid size (higher = more discriminative)

    Returns:
        imagehash.ImageHash, or None on error
    """
    try:
        img = Image.open(image_path).convert("RGB")
        return imagehash.phash(img, hash_size=hash_size)
    except Exception as exc:
        print(f"  ⚠️  Cannot hash {image_path.name}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Find duplicates
# ─────────────────────────────────────────────────────────────────────────────

def find_duplicates(
    image_dir: Path,
    hash_size: int = 8,
    threshold: int = 5,
) -> dict[str, list[str]]:
    """
    Identify groups of near-duplicate images using Hamming distance on pHash.

    Args:
        image_dir:  Root directory to scan (recursive)
        hash_size:  Perceptual hash grid size
        threshold:  Max Hamming distance to consider images duplicates
                    (0 = exact bit-for-bit, 5 = near-duplicate)

    Returns:
        dict {original_path: [duplicate_path, ...]}
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths: list[Path] = []
    for ext in exts:
        image_paths.extend(image_dir.rglob(ext))

    print(f"\nScanning {len(image_paths)} images in {image_dir} …")

    # Compute all hashes
    hashes: dict[str, imagehash.ImageHash] = {}
    for img_path in tqdm(image_paths, desc="Computing pHash"):
        h = compute_phash(img_path, hash_size)
        if h is not None:
            hashes[str(img_path)] = h

    # Compare every pair — O(n²) but fast for <10k images
    duplicates: dict[str, list[str]] = defaultdict(list)
    marked: set[str] = set()
    paths = list(hashes.keys())

    for i, path1 in enumerate(tqdm(paths, desc="Comparing hashes")):
        if path1 in marked:
            continue
        for path2 in paths[i + 1:]:
            if path2 in marked:
                continue
            dist = hashes[path1] - hashes[path2]
            if dist <= threshold:
                duplicates[path1].append(path2)
                marked.add(path2)

    return dict(duplicates)


# ─────────────────────────────────────────────────────────────────────────────
# Remove duplicates
# ─────────────────────────────────────────────────────────────────────────────

def remove_duplicates(
    duplicates: dict[str, list[str]],
    dry_run: bool = True,
) -> int:
    """
    Delete duplicate images (and corresponding YOLO label files).

    Args:
        duplicates: Mapping from original → list of duplicate paths
        dry_run:    If True, only print — do NOT delete

    Returns:
        Number of files deleted (or that would be deleted)
    """
    total = sum(len(v) for v in duplicates.values())

    if total == 0:
        print("\n✅ No duplicates found!")
        return 0

    print(f"\nFound {len(duplicates)} duplicate groups ({total} duplicate files)")

    if dry_run:
        print("\n[DRY RUN] Would delete (showing first 5 groups):")
        for original, dups in list(duplicates.items())[:5]:
            print(f"  KEEP:   {original}")
            for dup in dups:
                print(f"  DELETE: {dup}")
            print()
        if len(duplicates) > 5:
            print(f"  … and {len(duplicates) - 5} more groups")
        print("\nRe-run WITHOUT --dry_run to actually delete.")
        return total

    deleted = 0
    for _, dups in tqdm(duplicates.items(), desc="Deleting duplicates"):
        for dup_str in dups:
            dup = Path(dup_str)

            # Delete image
            if dup.exists():
                dup.unlink()
                deleted += 1

            # Delete matching YOLO label (same stem, .txt extension)
            label = Path(
                str(dup)
                .replace("/images/", "/labels/")
                .replace("\\images\\", "\\labels\\")
            ).with_suffix(".txt")
            if label.exists():
                label.unlink()

    print(f"\n✅ Deleted {deleted} duplicate images")
    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def save_report(
    duplicates: dict[str, list[str]],
    output_path: Path,
    dry_run: bool,
) -> None:
    """Write a human-readable duplicate detection report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(len(v) for v in duplicates.values())

    with open(output_path, "w") as f:
        f.write("DUPLICATE DETECTION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Duplicate groups : {len(duplicates)}\n")
        f.write(f"Total duplicates : {total}\n")
        f.write(f"Dry run          : {dry_run}\n\n")
        f.write("DUPLICATE GROUPS (first 50):\n")
        f.write("-" * 60 + "\n")

        for i, (original, dups) in enumerate(list(duplicates.items())[:50]):
            f.write(f"\nGroup {i + 1}:\n")
            f.write(f"  KEEP:   {original}\n")
            for dup in dups:
                f.write(f"  DELETE: {dup}\n")

    print(f"✅ Report saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remove duplicate images using perceptual hashing"
    )
    p.add_argument("--input_dir", default="dataset/raw",
                   help="Root directory to scan (default: dataset/raw)")
    p.add_argument("--threshold", type=int, default=5,
                   help="Hamming distance threshold (0=exact, 5=near-dup, default:5)")
    p.add_argument("--hash_size", type=int, default=8,
                   help="pHash grid size (default: 8)")
    p.add_argument("--dry_run", action="store_true",
                   help="Preview only — do NOT delete any files")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║        PHASE 1C — DUPLICATE DETECTION & REMOVAL                 ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\nInput directory : {args.input_dir}")
    print(f"Hash threshold  : {args.threshold}")
    print(f"Mode            : {'DRY RUN (no deletions)' if args.dry_run else 'LIVE (will delete)'}")

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"\n❌ Directory not found: {input_dir}")
        return

    duplicates = find_duplicates(input_dir, args.hash_size, args.threshold)
    deleted = remove_duplicates(duplicates, dry_run=args.dry_run)
    save_report(duplicates, Path("results/cleaning/duplicate_report.txt"), args.dry_run)

    print("\n" + "=" * 70)
    print(f"  {'Would remove' if args.dry_run else 'Removed'}: {deleted} duplicate images")
    print("=" * 70)
    print("\nNext: python src/utils/phase1c_filter_quality.py --input_dir dataset/raw")


if __name__ == "__main__":
    main()
