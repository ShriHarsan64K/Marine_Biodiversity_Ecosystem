#!/usr/bin/env python3
"""
PHASE 1C — Image Quality Filtering
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Filters out:
  • Images below minimum resolution (480 × 480 px)
  • Blurry images (Laplacian variance < threshold)
  • Corrupt / unreadable files
  • Extreme aspect ratios (> 4:1)

Usage:
    # Preview (dry run):
    python src/utils/phase1c_filter_quality.py --input_dir dataset/raw --dry_run

    # Move rejected images to a quarantine folder:
    python src/utils/phase1c_filter_quality.py --input_dir dataset/raw

Output:
    results/cleaning/quality_report.txt
"""

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2           # type: ignore
import numpy as np
from tqdm import tqdm  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MIN_RESOLUTION: int   = 480    # Minimum short-edge in pixels
BLUR_THRESHOLD: float = 100.0  # Laplacian variance below this → blurry
MAX_ASPECT_RATIO: float = 4.0  # width/height or height/width > this → skip


# ─────────────────────────────────────────────────────────────────────────────
# Per-image quality checks
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QualityResult:
    path: str
    passed: bool
    reason: str = ""
    width: int = 0
    height: int = 0
    blur_score: float = 0.0


def check_image_quality(image_path: Path) -> QualityResult:
    """
    Run all quality checks on a single image.

    Returns a QualityResult — 'passed' is True only if ALL checks pass.
    """
    result = QualityResult(path=str(image_path), passed=False)

    # ── Load image ──────────────────────────────────────────────────────────
    img = cv2.imread(str(image_path))
    if img is None:
        result.reason = "Corrupt / unreadable"
        return result

    h, w = img.shape[:2]
    result.width  = w
    result.height = h

    # ── Resolution check ────────────────────────────────────────────────────
    if min(h, w) < MIN_RESOLUTION:
        result.reason = f"Low resolution: {w}×{h} (min {MIN_RESOLUTION}px)"
        return result

    # ── Aspect ratio check ──────────────────────────────────────────────────
    ratio = max(w, h) / max(min(w, h), 1)
    if ratio > MAX_ASPECT_RATIO:
        result.reason = f"Extreme aspect ratio: {ratio:.2f}:1 (max {MAX_ASPECT_RATIO}:1)"
        return result

    # ── Blur detection (Laplacian variance) ─────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    result.blur_score = lap_var

    if lap_var < BLUR_THRESHOLD:
        result.reason = f"Too blurry: Laplacian var={lap_var:.1f} (min {BLUR_THRESHOLD})"
        return result

    # ── All passed ──────────────────────────────────────────────────────────
    result.passed = True
    result.reason = "OK"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch filtering
# ─────────────────────────────────────────────────────────────────────────────

def filter_dataset(
    input_dir: Path,
    quarantine_dir: Path,
    dry_run: bool = True,
) -> tuple[list[QualityResult], list[QualityResult]]:
    """
    Scan all images, move failures to quarantine_dir.

    Args:
        input_dir:      Root directory to scan
        quarantine_dir: Where to move rejected images
        dry_run:        If True, only report — do NOT move files

    Returns:
        (passed_results, failed_results)
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    all_images: list[Path] = []
    for ext in exts:
        all_images.extend(input_dir.rglob(ext))

    print(f"\nScanning {len(all_images)} images…")

    passed: list[QualityResult] = []
    failed: list[QualityResult] = []

    for img_path in tqdm(all_images, desc="Quality check"):
        result = check_image_quality(img_path)
        if result.passed:
            passed.append(result)
        else:
            failed.append(result)

    # Print summary
    print(f"\n{'─' * 60}")
    print(f"  Passed : {len(passed)} images")
    print(f"  Failed : {len(failed)} images")
    print(f"{'─' * 60}")

    # Breakdown by failure reason
    reason_counts: dict[str, int] = {}
    for r in failed:
        key = r.reason.split(":")[0]
        reason_counts[key] = reason_counts.get(key, 0) + 1
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    # Move rejected images (unless dry run)
    if not dry_run and failed:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nMoving {len(failed)} rejected images to {quarantine_dir} …")
        for r in tqdm(failed, desc="Quarantining"):
            src = Path(r.path)
            dst = quarantine_dir / src.name
            if src.exists():
                shutil.move(str(src), str(dst))

                # Move matching label if exists
                label = Path(
                    str(src)
                    .replace("/images/", "/labels/")
                    .replace("\\images\\", "\\labels\\")
                ).with_suffix(".txt")
                if label.exists():
                    shutil.move(str(label), str(quarantine_dir / label.name))

        print(f"✅ {len(failed)} images quarantined")
    elif dry_run:
        print(f"\n[DRY RUN] Re-run WITHOUT --dry_run to move {len(failed)} rejected images.")

    return passed, failed


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def save_quality_report(
    passed: list[QualityResult],
    failed: list[QualityResult],
    output_path: Path,
    dry_run: bool,
) -> None:
    """Write a detailed quality report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("IMAGE QUALITY FILTER REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total scanned : {len(passed) + len(failed)}\n")
        f.write(f"Passed        : {len(passed)}\n")
        f.write(f"Failed        : {len(failed)}\n")
        f.write(f"Dry run       : {dry_run}\n\n")

        # Blur statistics on passed images
        if passed:
            blur_scores = [r.blur_score for r in passed]
            f.write(f"Blur score (passed) — "
                    f"min={min(blur_scores):.1f}  "
                    f"avg={np.mean(blur_scores):.1f}  "
                    f"max={max(blur_scores):.1f}\n\n")

        f.write("FAILED IMAGES:\n")
        f.write("-" * 60 + "\n")
        for r in failed:
            f.write(f"  [{r.reason}]  {r.path}\n")

    print(f"✅ Quality report saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter low-quality underwater images")
    p.add_argument("--input_dir", default="dataset/raw",
                   help="Directory to scan (default: dataset/raw)")
    p.add_argument("--quarantine_dir", default="dataset/quarantine",
                   help="Where to move rejected images (default: dataset/quarantine)")
    p.add_argument("--min_resolution", type=int, default=MIN_RESOLUTION,
                   help=f"Minimum short-edge pixels (default: {MIN_RESOLUTION})")
    p.add_argument("--blur_threshold", type=float, default=BLUR_THRESHOLD,
                   help=f"Minimum Laplacian variance (default: {BLUR_THRESHOLD})")
    p.add_argument("--dry_run", action="store_true",
                   help="Preview only — do NOT move any files")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║          PHASE 1C — IMAGE QUALITY FILTERING                     ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\nInput directory  : {args.input_dir}")
    print(f"Quarantine dir   : {args.quarantine_dir}")
    print(f"Min resolution   : {args.min_resolution} px")
    print(f"Blur threshold   : {args.blur_threshold}")
    print(f"Mode             : {'DRY RUN' if args.dry_run else 'LIVE'}")

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"\n❌ Directory not found: {input_dir}")
        return

    passed, failed = filter_dataset(
        input_dir,
        Path(args.quarantine_dir),
        dry_run=args.dry_run,
    )

    save_quality_report(
        passed, failed,
        Path("results/cleaning/quality_report.txt"),
        args.dry_run,
    )

    print("\n" + "=" * 70)
    print(f"  Retained {len(passed)} high-quality images")
    print("=" * 70)
    print("\nNext: python src/utils/phase1c_validate_annotations.py")


if __name__ == "__main__":
    main()
