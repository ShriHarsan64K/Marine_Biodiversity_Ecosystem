#!/usr/bin/env python3
"""
PHASE 1D - Strategic Offline Augmentation
3-Class Marine Biodiversity Project
Classes: 0=Butterflyfish, 1=Parrotfish, 2=Angelfish

Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Usage:
    python src/utils/phase1d_augment_dataset.py \
        --input_dir  dataset/standardized \
        --output_dir dataset/augmented    \
        --plan       configs/augmentation_plan.json \
        --target     800
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

CLASS_NAMES = {0: "Butterflyfish", 1: "Parrotfish", 2: "Angelfish"}


# ── Augmentation primitives ───────────────────────────────────────────────

def augment_hsv(image, h_gain=0.02, s_gain=0.7, v_gain=0.4):
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1.0
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    dtype = image.dtype
    x = np.arange(256, dtype=np.int16)
    lut_h = ((x * r[0]) % 180).astype(dtype)
    lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
    merged = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


def augment_brightness_contrast(image, brightness=0.2, contrast=0.2):
    alpha = 1.0 + random.uniform(-contrast, contrast)
    beta  = random.uniform(-brightness, brightness) * 255
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def augment_gaussian_noise(image, sigma=5.0):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment_flip_horizontal(image, boxes):
    flipped   = cv2.flip(image, 1)
    new_boxes = [[b[0], 1.0 - b[1], b[2], b[3], b[4]] for b in boxes]
    return flipped, new_boxes


def augment_rotate(image, boxes, angle):
    h, w   = image.shape[:2]
    cx, cy = w / 2, h / 2
    M      = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    new_boxes = []
    for box in boxes:
        cid, xc, yc, bw, bh = box
        x1, y1 = (xc - bw/2)*w, (yc - bh/2)*h
        x2, y2 = (xc + bw/2)*w, (yc + bh/2)*h
        corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
        corners_h = np.hstack([corners, np.ones((4,1), dtype=np.float32)])
        rc = corners_h @ M.T
        rx1 = max(0.0, min(rc[:,0].min(), w)) / w
        ry1 = max(0.0, min(rc[:,1].min(), h)) / h
        rx2 = max(0.0, min(rc[:,0].max(), w)) / w
        ry2 = max(0.0, min(rc[:,1].max(), h)) / h
        nw, nh = rx2-rx1, ry2-ry1
        if nw > 0 and nh > 0:
            new_boxes.append([cid, (rx1+rx2)/2, (ry1+ry2)/2, nw, nh])
    return rotated, new_boxes


def apply_random_augmentation(image, boxes):
    aug = augment_hsv(image)
    aug = augment_brightness_contrast(aug)
    if random.random() < 0.5:
        aug, boxes = augment_flip_horizontal(aug, boxes)
    if random.random() < 0.5:
        aug, boxes = augment_rotate(aug, boxes, random.uniform(-15.0, 15.0))
    aug = augment_gaussian_noise(aug, sigma=5.0)
    return aug, boxes


# ── I/O helpers ───────────────────────────────────────────────────────────

def read_yolo_label(label_path):
    boxes = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    boxes.append([float(p) for p in parts])
                except ValueError:
                    pass
    return boxes


def write_yolo_label(label_path, boxes):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n" for b in boxes]
    with open(label_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


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


# ── Augmentation per class ────────────────────────────────────────────────

def augment_class(source_images, needed, output_dir, class_name):
    if needed <= 0:
        return 0

    images_labels = []
    for img_path in source_images:
        lbl = Path(
            str(img_path)
            .replace("/images/", "/labels/")
            .replace("\\images\\", "\\labels\\")
        ).with_suffix(".txt")
        if lbl.exists():
            images_labels.append((img_path, lbl))

    if not images_labels:
        print(f"  WARNING: No labelled images for {class_name}")
        return 0

    out_img = output_dir / "images"
    out_lbl = output_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    generated = 0
    attempt   = 0
    pbar = tqdm(total=needed, desc=f"  Augmenting {class_name}", leave=False)

    while generated < needed and attempt < needed * 10:
        attempt += 1
        img_path, lbl_path = random.choice(images_labels)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes = read_yolo_label(lbl_path)
        if not boxes:
            continue
        aug_img, aug_boxes = apply_random_augmentation(img, boxes)
        if not aug_boxes:
            continue
        stem = f"{img_path.stem}_aug{generated:04d}"
        cv2.imwrite(str(out_img / f"{stem}.jpg"), aug_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        write_yolo_label(out_lbl / f"{stem}.txt", aug_boxes)
        generated += 1
        pbar.update(1)

    pbar.close()
    return generated


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  default="dataset/standardized")
    parser.add_argument("--output_dir", default="dataset/augmented")
    parser.add_argument("--plan",       default="configs/augmentation_plan.json")
    parser.add_argument("--target",     type=int, default=800)
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: {input_dir} not found. Run Phase 1C first.")
        return

    plan_path = Path(args.plan)
    if not plan_path.exists():
        print(f"ERROR: {plan_path} not found. Run phase1d_analyze_balance.py first.")
        return

    with open(plan_path, encoding="utf-8") as f:
        plan = json.load(f)

    # Step 1 — Copy originals
    print("\nCopying original images...")
    total_copied = 0
    out_img = output_dir / "images"
    out_lbl = output_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    for img in input_dir.rglob("*.jpg"):
        shutil.copy2(img, out_img / img.name)
        total_copied += 1
    for img in input_dir.rglob("*.png"):
        shutil.copy2(img, out_img / img.name)
        total_copied += 1
    for lbl in input_dir.rglob("*.txt"):
        shutil.copy2(lbl, out_lbl / lbl.name)

    print(f"  Copied {total_copied} original images")

    # Step 2 — Collect images per class
    images_by_class = defaultdict(list)
    for lbl_path in input_dir.rglob("*.txt"):
        classes_seen = set()
        try:
            with open(lbl_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            classes_seen.add(int(parts[0]))
                        except ValueError:
                            pass
        except Exception:
            continue
        img = find_image(lbl_path)
        if img:
            for cid in classes_seen:
                images_by_class[cid].append(img)

    # Step 3 — Augment under-represented classes
    print("\nAugmenting under-represented classes...")
    total_generated = 0

    for cid_str, info in plan.items():
        cid    = int(cid_str)
        needed = info.get("images_needed", 0)
        name   = info.get("class_name", CLASS_NAMES.get(cid, f"class_{cid}"))

        if needed <= 0:
            print(f"  OK  {name}: already at target ({info['current_images']} images)")
            continue

        print(f"\n  {name}: generating {needed} augmented images...")
        generated = augment_class(
            source_images=images_by_class.get(cid, []),
            needed=needed,
            output_dir=output_dir,
            class_name=name,
        )
        print(f"  Done: {generated}/{needed} generated")
        total_generated += generated

    total_out = len(list(out_img.rglob("*.jpg")))
    print("\n" + "=" * 55)
    print(f"  Original images  : {total_copied}")
    print(f"  Augmented images : {total_generated}")
    print(f"  Total in output  : {total_out}")
    print("=" * 55)
    print("\nNext: python src/utils/phase1d_split_dataset.py")


if __name__ == "__main__":
    main()