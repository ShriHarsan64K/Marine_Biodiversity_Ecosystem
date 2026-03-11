"""
Phase 1C: Validate & Standardize Annotations
3-Class Marine Biodiversity Project
Classes: 0=Butterflyfish, 1=Parrotfish, 2=Angelfish

Author: Shri Harsan M | M.Tech Data Science | SRM Institute
"""

import json
import shutil
import argparse
from pathlib import Path

SKIP_FILES = {
    "readme.txt", "license.txt", "notes.txt", "classes.txt",
    "obj.data", "obj.names", "train.txt", "valid.txt", "test.txt",
    "_darknet.labels"
}

# Per-dataset hardcoded class remapping
# key = substring of dataset folder name
DATASET_CONFIGS = {
    "butterflyfish": {0: 0, 1: 0, 2: 0, 3: 0},  # all 4 species → butterflyfish
    "angelfish":     {0: -1, 1: 2},               # 'Images'=-1, 'angelfish'→2
    "seychelles":    {0: -1, 1: -1, 2: -1},        # all skip
    "parrotfish":    {0: 1, 1: 1},                  # labels may contain 0 or 1, both → class 1
}

CLASS_NAMES = {0: "Butterflyfish", 1: "Parrotfish", 2: "Angelfish"}


def load_reverse_mapping(config_path="configs/reverse_mapping.json"):
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {config_path}: {e}")
        return {}


def load_data_yaml(dataset_dir):
    yaml_path = Path(dataset_dir) / "data.yaml"
    if not yaml_path.exists():
        return []
    try:
        import yaml
        with open(yaml_path, encoding="utf-8", errors="ignore") as f:
            data = yaml.safe_load(f)
        return data.get("names", [])
    except Exception:
        return []


def get_class_mapping(dataset_name, class_names, reverse_mapping):
    for key, cfg in DATASET_CONFIGS.items():
        if key in dataset_name.lower():
            return cfg
    mapping = {}
    for idx, name in enumerate(class_names):
        name_orig = name.strip()
        name_lower = name_orig.lower()
        if name_orig in reverse_mapping:
            mapping[idx] = reverse_mapping[name_orig]
        elif name_lower in reverse_mapping:
            mapping[idx] = reverse_mapping[name_lower]
        else:
            mapping[idx] = -1
    return mapping


def validate_box(parts):
    if len(parts) != 5:
        return False
    try:
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            return False
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            return False
        return True
    except ValueError:
        return False


def process_split(dataset_dir, output_dir, reverse_mapping, split):
    """Process train/valid/test split."""
    dataset_dir = Path(dataset_dir)
    dataset_name = dataset_dir.name

    img_dir = dataset_dir / split / "images"
    lbl_dir = dataset_dir / split / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        return 0, 0, 0

    class_names = load_data_yaml(dataset_dir)
    class_map = get_class_mapping(dataset_name, class_names, reverse_mapping)

    out_img = Path(output_dir) / split / "images"
    out_lbl = Path(output_dir) / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    valid_imgs = skipped = remapped = 0

    for lbl_file in lbl_dir.rglob("*.txt"):
        if lbl_file.name.lower() in SKIP_FILES:
            continue
        try:
            with open(lbl_file, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            continue

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                orig_cls = int(parts[0])
            except ValueError:
                continue
            if not validate_box(parts):
                continue
            new_cls = class_map.get(orig_cls, -1)
            if new_cls == -1:
                skipped += 1
                continue
            parts[0] = str(new_cls)
            new_lines.append(" ".join(parts) + "\n")
            remapped += 1

        if not new_lines:
            continue

        stem = lbl_file.stem
        img_file = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
            candidate = img_dir / (stem + ext)
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            continue

        unique_stem = f"{dataset_name}_{split}_{stem}"
        shutil.copy2(img_file, out_img / (unique_stem + img_file.suffix))
        with open(out_lbl / (unique_stem + ".txt"), "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        valid_imgs += 1

    return valid_imgs, remapped, skipped


def process_flat(dataset_dir, output_dir, reverse_mapping, split="train"):
    """Process flat dataset (no train/valid/test subfolders)."""
    dataset_dir = Path(dataset_dir)
    dataset_name = dataset_dir.name

    img_dir = dataset_dir / "images"
    lbl_dir = dataset_dir / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        return 0, 0, 0

    class_names = load_data_yaml(dataset_dir)
    class_map = get_class_mapping(dataset_name, class_names, reverse_mapping)

    out_img = Path(output_dir) / split / "images"
    out_lbl = Path(output_dir) / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    valid_imgs = skipped = remapped = 0

    for lbl_file in lbl_dir.rglob("*.txt"):
        if lbl_file.name.lower() in SKIP_FILES:
            continue
        try:
            with open(lbl_file, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            continue

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                orig_cls = int(parts[0])
            except ValueError:
                continue
            if not validate_box(parts):
                continue
            new_cls = class_map.get(orig_cls, -1)
            if new_cls == -1:
                skipped += 1
                continue
            parts[0] = str(new_cls)
            new_lines.append(" ".join(parts) + "\n")
            remapped += 1

        if not new_lines:
            continue

        stem = lbl_file.stem
        img_file = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
            candidate = img_dir / (stem + ext)
            if candidate.exists():
                img_file = candidate
                break
        if img_file is None:
            continue

        unique_stem = f"{dataset_name}_{stem}"
        shutil.copy2(img_file, out_img / (unique_stem + img_file.suffix))
        with open(out_lbl / (unique_stem + ".txt"), "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        valid_imgs += 1

    return valid_imgs, remapped, skipped


def count_classes(output_dir):
    counts = {0: 0, 1: 0, 2: 0}
    for split in ["train", "valid", "test"]:
        lbl_dir = Path(output_dir) / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.rglob("*.txt"):
            try:
                with open(lbl_file, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls = int(parts[0])
                            if cls in counts:
                                counts[cls] += 1
            except Exception:
                continue
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  default="dataset/raw/roboflow")
    parser.add_argument("--parrot_dir", default="dataset/raw/kaggle/parrotfish_yolo")
    parser.add_argument("--output_dir", default="dataset/standardized")
    parser.add_argument("--config",     default="configs/reverse_mapping.json")
    args = parser.parse_args()

    reverse_mapping = load_reverse_mapping(args.config)
    print(f"Loaded {len([k for k in reverse_mapping if not k.startswith('_')])} species mappings\n")

    total_imgs = total_boxes = total_skipped = 0

    # Process roboflow datasets (train/valid/test splits)
    roboflow_root = Path(args.input_dir)
    if roboflow_root.exists():
        for dataset_dir in sorted(roboflow_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            print(f"Dataset: {dataset_dir.name}")
            for split in ["train", "valid", "test"]:
                imgs, boxes, skipped = process_split(
                    dataset_dir, args.output_dir, reverse_mapping, split
                )
                if imgs > 0:
                    print(f"  {split:5s}: {imgs:4d} images  {boxes:5d} boxes  {skipped:4d} skipped")
                    total_imgs += imgs
                    total_boxes += boxes
                    total_skipped += skipped
            print()

    # Process parrotfish (flat structure → goes to train)
    parrot_dir = Path(args.parrot_dir)
    if parrot_dir.exists():
        print(f"Dataset: {parrot_dir.name} (flat → train)")
        imgs, boxes, skipped = process_flat(
            parrot_dir, args.output_dir, reverse_mapping, split="train"
        )
        print(f"  train: {imgs:4d} images  {boxes:5d} boxes  {skipped:4d} skipped")
        total_imgs += imgs
        total_boxes += boxes
        total_skipped += skipped
        print()
    else:
        print(f"WARNING: parrotfish_yolo not found at {parrot_dir}")
        print("Run auto_annotate_parrotfish.py first!\n")

    # Write data.yaml
    out_yaml = Path(args.output_dir) / "data.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        f.write("train: ../train/images\n")
        f.write("val: ../valid/images\n")
        f.write("test: ../test/images\n\n")
        f.write("nc: 3\n")
        f.write("names: ['Butterflyfish', 'Parrotfish', 'Angelfish']\n")

    class_counts = count_classes(args.output_dir)

    print("=" * 55)
    print("PHASE 1C COMPLETE — 3-CLASS PROJECT")
    print("=" * 55)
    print(f"Total images : {total_imgs}")
    print(f"Total boxes  : {total_boxes}")
    print(f"Skipped      : {total_skipped}")
    print(f"\nClass Distribution:")
    for cid, cnt in class_counts.items():
        bar = "#" * min(cnt // 10, 40)
        print(f"  {cid} {CLASS_NAMES[cid]:15s}: {cnt:5d}  {bar}")

    min_c = min(class_counts.values())
    if min_c == 0:
        print("\nWARNING: Some classes have 0 annotations!")
    elif min_c < 200:
        print(f"\nWARNING: Low data for some classes (min={min_c})")
    else:
        print(f"\nAll classes OK (min={min_c})")

    print(f"\ndata.yaml → {out_yaml}")
    print("Ready for Phase 1D!")


if __name__ == "__main__":
    main()