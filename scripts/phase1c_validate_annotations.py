#!/usr/bin/env python3
"""
PHASE 1C — Annotation Validation + Label Standardisation
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Step 1 — Validates all YOLO bounding box labels:
  • Coordinates in [0, 1] range
  • Width & height > 0
  • 5 fields per line (class x_c y_c w h)
  • Matching image file exists for every label

Step 2 — Re-maps raw class IDs to the 8 standard indicator classes
  using configs/reverse_mapping.json produced by phase1b_map_species.py.

Usage:
    python src/utils/phase1c_validate_annotations.py \
        --input_dir dataset/raw  \
        --output_dir dataset/standardized

Output:
    dataset/standardized/   — valid, re-mapped labels + copied images
    results/cleaning/annotation_validation.txt
"""

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Standard class definitions
# ─────────────────────────────────────────────────────────────────────────────

STANDARD_CLASSES: dict[int, str] = {
    0: "butterflyfish",
    1: "grouper",
    2: "parrotfish",
    3: "surgeonfish",
    4: "damselfish",
    5: "wrasse",
    6: "triggerfish",
    7: "angelfish",
}

NUM_CLASSES = len(STANDARD_CLASSES)


# ─────────────────────────────────────────────────────────────────────────────
# Single-box validation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoxError:
    line_no: int
    raw_line: str
    reason: str


def validate_box_line(line: str, line_no: int,
                      num_classes: int = NUM_CLASSES) -> Optional[BoxError]:
    """
    Validate a single YOLO bounding-box line.

    Returns a BoxError if invalid, else None.
    """
    parts = line.strip().split()

    if len(parts) != 5:
        return BoxError(line_no, line, f"Expected 5 fields, got {len(parts)}")

    try:
        class_id = int(parts[0])
        x_c, y_c, bw, bh = map(float, parts[1:])
    except ValueError:
        return BoxError(line_no, line, "Non-numeric field")

    if class_id < 0 or class_id >= num_classes:
        return BoxError(line_no, line,
                        f"class_id={class_id} out of range [0, {num_classes-1}]")

    for name, val in [("x_c", x_c), ("y_c", y_c), ("w", bw), ("h", bh)]:
        if not (0.0 <= val <= 1.0):
            return BoxError(line_no, line, f"{name}={val:.4f} outside [0, 1]")

    if bw <= 0 or bh <= 0:
        return BoxError(line_no, line, "Zero-area bounding box")

    return None  # Valid


# ─────────────────────────────────────────────────────────────────────────────
# Load class mapping from data.yaml
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_classes(yaml_path: Path) -> list[str]:
    """Read class names from a YOLOv8 data.yaml file."""
    import yaml  # type: ignore

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        names = data.get("names", {})
        if isinstance(names, dict):
            return [names[i] for i in sorted(names.keys())]
        if isinstance(names, list):
            return names
        return []
    except Exception:
        return []


def build_class_remap(
    original_classes: list[str],
    reverse_mapping: dict[str, int],
) -> dict[int, int]:
    """
    Map original dataset class IDs → standard indicator class IDs.

    Args:
        original_classes: Class names from dataset's data.yaml
        reverse_mapping:  {normalised_species_name: standard_class_id}

    Returns:
        {old_class_id: new_class_id}  — only includes mapped classes
    """
    remap: dict[int, int] = {}
    for old_id, class_name in enumerate(original_classes):
        norm = class_name.lower().strip()
        # Exact match
        if norm in reverse_mapping:
            remap[old_id] = reverse_mapping[norm]
            continue
        # Substring match
        for key, new_id in reverse_mapping.items():
            if key in norm or norm in key:
                remap[old_id] = new_id
                break

    return remap


# ─────────────────────────────────────────────────────────────────────────────
# Process a single dataset directory
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(
    dataset_dir: Path,
    output_dir: Path,
    reverse_mapping: dict[str, int],
    validation_errors: list[dict],
    stats: dict,
) -> None:
    """
    Validate annotations and copy re-mapped label + image pairs to output_dir.
    """
    # Try to find a data.yaml to get original class names
    yaml_files = list(dataset_dir.rglob("data.yaml"))
    original_classes: list[str] = []
    class_remap: dict[int, int] = {}

    if yaml_files:
        original_classes = load_dataset_classes(yaml_files[0])
        class_remap = build_class_remap(original_classes, reverse_mapping)
        print(f"  data.yaml: {len(original_classes)} classes → "
              f"{len(class_remap)} mapped to indicator families")

    label_paths = list(dataset_dir.rglob("*.txt"))
    # Exclude any stray data.yaml-like txt files
    label_paths = [p for p in label_paths if p.name != "classes.txt"]

    for label_path in tqdm(label_paths, desc=f"  {dataset_dir.name}", leave=False):
        valid_lines: list[str] = []
        errors: list[BoxError] = []

        with open(label_path) as f:
            raw_lines = f.readlines()

        for i, line in enumerate(raw_lines):
            if not line.strip():
                continue

            # If we have a remap, convert class ID first
            parts = line.strip().split()
            if len(parts) == 5 and class_remap:
                old_id = int(parts[0])
                if old_id not in class_remap:
                    # Class not in indicator families — skip
                    continue
                parts[0] = str(class_remap[old_id])
                line = " ".join(parts) + "\n"

            error = validate_box_line(line, i + 1)
            if error:
                errors.append(error)
                stats["invalid_boxes"] += 1
                validation_errors.append({
                    "file": str(label_path),
                    "line": error.line_no,
                    "raw": error.raw_line.strip(),
                    "reason": error.reason
                })
            else:
                valid_lines.append(line)
                stats["valid_boxes"] += 1

        if not valid_lines:
            stats["empty_labels"] += 1
            continue

        # Find matching image
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            img_path = label_path.with_suffix(ext)
            # Also look in sibling 'images' folder
            if not img_path.exists():
                img_candidate = Path(
                    str(label_path)
                    .replace("/labels/", "/images/")
                    .replace("\\labels\\", "\\images\\")
                ).with_suffix(ext)
                if img_candidate.exists():
                    img_path = img_candidate
                    break
            else:
                break

        if not img_path.exists():
            stats["missing_images"] += 1
            continue

        # Copy label and image to output
        rel = label_path.relative_to(dataset_dir)
        out_label = output_dir / "labels" / f"{label_path.stem}.txt"
        out_image = output_dir / "images" / img_path.name

        out_label.parent.mkdir(parents=True, exist_ok=True)
        out_image.parent.mkdir(parents=True, exist_ok=True)

        out_label.write_text("".join(valid_lines))
        shutil.copy2(img_path, out_image)
        stats["valid_images"] += 1


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def save_validation_report(
    stats: dict,
    errors: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("ANNOTATION VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        for k, v in stats.items():
            f.write(f"  {k:25s}: {v}\n")

        valid_rate = (stats["valid_boxes"] /
                      max(stats["valid_boxes"] + stats["invalid_boxes"], 1) * 100)
        f.write(f"\n  Annotation validity rate: {valid_rate:.1f}%\n")

        if errors:
            f.write("\n\nFIRST 100 ERRORS:\n")
            f.write("-" * 60 + "\n")
            for e in errors[:100]:
                f.write(f"  File: {e['file']}\n")
                f.write(f"  Line {e['line']}: {e['raw']}\n")
                f.write(f"  Error: {e['reason']}\n\n")

    print(f"✅ Validation report saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate YOLO annotations and standardise class IDs"
    )
    p.add_argument("--input_dir",  default="dataset/raw",
                   help="Source directory (default: dataset/raw)")
    p.add_argument("--output_dir", default="dataset/standardized",
                   help="Output directory (default: dataset/standardized)")
    p.add_argument("--mapping",    default="configs/reverse_mapping.json",
                   help="Species reverse mapping JSON (from phase1b_map_species.py)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║     PHASE 1C — ANNOTATION VALIDATION & STANDARDISATION          ║")
    print("╚" + "═" * 68 + "╝")

    # Load reverse mapping
    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        print(f"\n⚠️  Reverse mapping not found: {mapping_path}")
        print("   Run phase1b_map_species.py first, or create configs/reverse_mapping.json")
        reverse_mapping: dict[str, int] = {}
    else:
        reverse_mapping = json.loads(mapping_path.read_text())
        print(f"\n✅ Loaded {len(reverse_mapping)} species → class mappings")

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"\n❌ Input directory not found: {input_dir}")
        return

    stats: dict[str, int] = {
        "valid_images":   0,
        "valid_boxes":    0,
        "invalid_boxes":  0,
        "empty_labels":   0,
        "missing_images": 0,
    }
    errors: list[dict] = []

    # Process each source sub-directory
    source_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if not source_dirs:
        source_dirs = [input_dir]

    for source_dir in source_dirs:
        print(f"\nProcessing: {source_dir.name}")
        process_dataset(source_dir, output_dir, reverse_mapping, errors, stats)

    # Report
    save_validation_report(stats, errors, Path("results/cleaning/annotation_validation.txt"))

    valid_rate = (stats["valid_boxes"] /
                  max(stats["valid_boxes"] + stats["invalid_boxes"], 1) * 100)

    print("\n" + "=" * 70)
    print(f"  Valid images  : {stats['valid_images']}")
    print(f"  Valid boxes   : {stats['valid_boxes']}")
    print(f"  Invalid boxes : {stats['invalid_boxes']}")
    print(f"  Validity rate : {valid_rate:.1f}%")
    print("=" * 70)

    if valid_rate >= 95.0:
        print("✅ Annotation quality meets Phase 1C requirement (≥ 95%)")
    else:
        print("⚠️  Annotation validity below 95% — review errors in validation report")

    print("\nNext: python src/utils/phase1c_generate_statistics.py")


if __name__ == "__main__":
    main()
