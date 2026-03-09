#!/usr/bin/env python3
"""
PHASE 1D — Strategic Offline Augmentation
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Applies ecologically valid augmentations ONLY to under-represented classes
to reach the target image count (default 800 per class).

Augmentations used:
  1. HSV colour shift (simulates water depth)
  2. Horizontal flip (fish swim both ways)
  3. Rotation ±15° (natural orientation variation)
  4. Brightness / contrast (lighting variation)
  5. Gaussian noise (water turbidity)

Each augmented image gets a corresponding updated YOLO label file.

Usage:
    python src/utils/phase1d_augment_dataset.py \
        --input_dir  dataset/standardized \
        --output_dir dataset/augmented    \
        --plan       configs/augmentation_plan.json \
        --target     800

Output:
    dataset/augmented/   — original + augmented images with labels
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Optional

import cv2           # type: ignore
import numpy as np
from tqdm import tqdm  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation primitives
# ─────────────────────────────────────────────────────────────────────────────

def augment_hsv(
    image: np.ndarray,
    h_gain: float = 0.02,
    s_gain: float = 0.7,
    v_gain: float = 0.4,
) -> np.ndarray:
    """
    Randomise Hue / Saturation / Value channels.
    Simulates underwater depth colour shifts.
    """
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1.0
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    dtype = image.dtype
    x = np.arange(256, dtype=np.int16)
    lut_h = ((x * r[0]) % 180).astype(dtype)
    lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_v = np.clip(x * r[2], 0, 255).astype(dtype)

    merged = cv2.merge((
        cv2.LUT(hue, lut_h),
        cv2.LUT(sat, lut_s),
        cv2.LUT(val, lut_v),
    ))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


def augment_brightness_contrast(
    image: np.ndarray,
    brightness: float = 0.2,
    contrast: float = 0.2,
) -> np.ndarray:
    """Randomly adjust brightness and contrast."""
    alpha = 1.0 + random.uniform(-contrast, contrast)   # contrast
    beta  = random.uniform(-brightness, brightness) * 255  # brightness
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def augment_gaussian_noise(image: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Add Gaussian noise — simulates water turbidity."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def augment_flip_horizontal(
    image: np.ndarray,
    boxes: list[list[float]],
) -> tuple[np.ndarray, list[list[float]]]:
    """Flip image and update x-coordinate in YOLO boxes."""
    flipped = cv2.flip(image, 1)
    new_boxes = []
    for box in boxes:
        cid, xc, yc, bw, bh = box
        new_boxes.append([cid, 1.0 - xc, yc, bw, bh])
    return flipped, new_boxes


def augment_rotate(
    image: np.ndarray,
    boxes: list[list[float]],
    angle: float,
) -> tuple[np.ndarray, list[list[float]]]:
    """
    Rotate image by 'angle' degrees and update bounding boxes.
    Uses axis-aligned bounding box of the rotated corners.
    """
    h, w  = image.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)

    new_boxes: list[list[float]] = []
    for box in boxes:
        cid, xc, yc, bw, bh = box
        # Convert to pixel corners
        x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
        x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        # Rotate corners
        ones = np.ones((4, 1), dtype=np.float32)
        corners_h = np.hstack([corners, ones])
        rotated_corners = corners_h @ M.T

        # Axis-aligned bounding box
        rx1, ry1 = rotated_corners[:, 0].min(), rotated_corners[:, 1].min()
        rx2, ry2 = rotated_corners[:, 0].max(), rotated_corners[:, 1].max()

        # Clip and normalise
        rx1 = max(0.0, min(rx1, w)) / w
        ry1 = max(0.0, min(ry1, h)) / h
        rx2 = max(0.0, min(rx2, w)) / w
        ry2 = max(0.0, min(ry2, h)) / h

        new_xc = (rx1 + rx2) / 2
        new_yc = (ry1 + ry2) / 2
        new_bw = rx2 - rx1
        new_bh = ry2 - ry1

        if new_bw > 0 and new_bh > 0:
            new_boxes.append([cid, new_xc, new_yc, new_bw, new_bh])

    return rotated, new_boxes


# ─────────────────────────────────────────────────────────────────────────────
# Combined augmentation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def apply_random_augmentation(
    image: np.ndarray,
    boxes: list[list[float]],
) -> tuple[np.ndarray, list[list[float]]]:
    """
    Apply a random subset of augmentations.
    Always applies HSV + brightness/contrast (colour domain).
    Randomly applies flip (50%) and rotation.
    Always adds subtle Gaussian noise last.
    """
    # HSV colour shift (always)
    aug = augment_hsv(image)

    # Brightness / contrast (always)
    aug = augment_brightness_contrast(aug)

    # Horizontal flip (50%)
    if random.random() < 0.5:
        aug, boxes = augment_flip_horizontal(aug, boxes)

    # Rotation ±15° (50%)
    if random.random() < 0.5:
        angle = random.uniform(-15.0, 15.0)
        aug, boxes = augment_rotate(aug, boxes, angle)

    # Gaussian noise (always, subtle)
    aug = augment_gaussian_noise(aug, sigma=5.0)

    return aug, boxes


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_yolo_label(label_path: Path) -> list[list[float]]:
    """Read YOLO label file → list of [class_id, xc, yc, w, h]."""
    boxes: list[list[float]] = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    boxes.append([float(p) for p in parts])
                except ValueError:
                    pass
    return boxes


def write_yolo_label(label_path: Path, boxes: list[list[float]]) -> None:
    """Write boxes back to YOLO format."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n"
             for b in boxes]
    label_path.write_text("".join(lines))


def find_image_for_label(label_path: Path) -> Optional[Path]:
    """Find the corresponding image for a label file."""
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
# Augmentation orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def augment_class(
    source_images: list[Path],
    needed: int,
    output_dir: Path,
    class_name: str,
) -> int:
    """
    Generate 'needed' augmented images from 'source_images' pool.

    Returns number of images successfully generated.
    """
    if needed <= 0:
        return 0

    generated = 0
    images_labels: list[tuple[Path, Path]] = []

    for img_path in source_images:
        label_path = Path(
            str(img_path)
            .replace("/images/", "/labels/")
            .replace("\\images\\", "\\labels\\")
        ).with_suffix(".txt")
        if label_path.exists():
            images_labels.append((img_path, label_path))

    if not images_labels:
        print(f"  ⚠️  No labelled images found for {class_name}")
        return 0

    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=needed, desc=f"  Augmenting {class_name}", leave=False)

    attempt = 0
    while generated < needed and attempt < needed * 10:
        attempt += 1
        img_path, label_path = random.choice(images_labels)

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        boxes = read_yolo_label(label_path)
        if not boxes:
            continue

        aug_img, aug_boxes = apply_random_augmentation(img, boxes)
        if not aug_boxes:
            continue

        stem = f"{img_path.stem}_aug{generated:04d}"
        cv2.imwrite(str(out_images / f"{stem}.jpg"), aug_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        write_yolo_label(out_labels / f"{stem}.txt", aug_boxes)

        generated += 1
        pbar.update(1)

    pbar.close()
    return generated


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline augmentation for class balancing")
    p.add_argument("--input_dir",  default="dataset/standardized")
    p.add_argument("--output_dir", default="dataset/augmented")
    p.add_argument("--plan",       default="configs/augmentation_plan.json")
    p.add_argument("--target",     type=int, default=800)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║         PHASE 1D — STRATEGIC OFFLINE AUGMENTATION               ║")
    print("╚" + "═" * 68 + "╝")

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"\n❌ Input directory not found: {input_dir}")
        print("   Run phase1c_validate_annotations.py first")
        return

    # Load plan (or generate on the fly)
    plan_path = Path(args.plan)
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        print(f"\n✅ Loaded augmentation plan: {plan_path}")
    else:
        print(f"\n⚠️  Plan not found at {plan_path}")
        print("   Run phase1d_analyze_balance.py first, then re-run this script.")
        return

    # Step 1: Copy all original images to output_dir
    print("\nCopying original images to output directory…")
    total_copied = 0
    for img in input_dir.rglob("*.jpg"):
        dst = output_dir / "images" / img.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst)
        total_copied += 1
    for lbl in input_dir.rglob("*.txt"):
        dst = output_dir / "labels" / lbl.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lbl, dst)
    print(f"  Copied {total_copied} original images")

    # Step 2: Collect images per class from input_dir
    from collections import defaultdict
    images_by_class: dict[int, list[Path]] = defaultdict(list)
    for label_path in input_dir.rglob("*.txt"):
        classes_seen: set[int] = set()
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        classes_seen.add(int(parts[0]))
                    except ValueError:
                        pass
        img = find_image_for_label(label_path)
        if img:
            for cid in classes_seen:
                images_by_class[cid].append(img)

    # Step 3: Augment under-represented classes
    print("\nAugmenting under-represented classes…")
    total_generated = 0

    for cid_str, info in plan.items():
        cid     = int(cid_str)
        needed  = info.get("images_needed", 0)
        name    = info.get("class_name", f"class_{cid}")

        if needed <= 0:
            print(f"  ✅ {name}: already at target ({info['current_images']} images)")
            continue

        print(f"\n  {name}: generating {needed} augmented images…")
        generated = augment_class(
            source_images=images_by_class.get(cid, []),
            needed=needed,
            output_dir=output_dir,
            class_name=name,
        )
        print(f"    ✅ Generated {generated}/{needed}")
        total_generated += generated

    print("\n" + "=" * 70)
    total_images = len(list((output_dir / "images").rglob("*.jpg"))) if \
                   (output_dir / "images").exists() else 0
    print(f"  Original images   : {total_copied}")
    print(f"  Augmented images  : {total_generated}")
    print(f"  Total in output   : {total_images}")
    print("=" * 70)
    print("\nNext: python src/utils/phase1d_split_dataset.py")


if __name__ == "__main__":
    main()
