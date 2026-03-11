#!/usr/bin/env python3
"""
PHASE 3A — YOLOv8s Baseline Training
3-Class Marine Biodiversity Project
Classes: 0=Butterflyfish, 1=Parrotfish, 2=Angelfish

Model   : YOLOv8s (pretrained on COCO, 11M parameters)
Target  : mAP@0.5 > 85%
Hardware: RTX 3050 6GB

Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Usage:
    python src/training/phase3a_train_baseline.py \
        --data    dataset/processed/data.yaml \
        --weights yolov8s.pt \
        --epochs  100 \
        --batch   16
"""

import argparse
import sys
import time
from pathlib import Path

# ── Verify environment before importing ultralytics ───────────────────────────
def check_environment():
    import torch
    print("╔" + "═" * 62 + "╗")
    print("║      PHASE 3A — YOLOv8s BASELINE TRAINING                   ║")
    print("║      Marine Biodiversity — 3 Indicator Species               ║")
    print("╚" + "═" * 62 + "╝\n")

    print("── Environment Check ──────────────────────────────────────────")
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM     : {vram:.1f} GB")
        if vram < 5.5:
            print("  WARNING  : Low VRAM — switching to batch_size=8")
            return False   # signal to reduce batch
    else:
        print("  WARNING  : CUDA not available — training will be slow!")
    return True             # batch_size ok


# ── Custom on-the-fly Ancuti enhancement callback ─────────────────────────────
def make_enhance_callback():
    """
    Returns an ultralytics on_train_batch_start callback that applies
    Ancuti enhancement to each training batch on-the-fly.
    Only activates with 20% probability per batch to act as augmentation.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.enhancement.ancuti_fusion import enhance
        import numpy as np
        import torch

        def enhance_batch(trainer):
            """Apply Ancuti enhancement to ~20% of training batch images."""
            import random
            if random.random() > 0.20:
                return
            try:
                imgs = trainer.batch["img"]         # (B, C, H, W) float tensor
                B = imgs.shape[0]
                for i in range(B):
                    if random.random() < 0.5:
                        # Convert tensor → numpy BGR → enhance → back
                        img_np = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        img_bgr = img_np[:, :, ::-1].copy()
                        enh = enhance(img_bgr)
                        enh_rgb = enh[:, :, ::-1].astype(np.float32) / 255.0
                        imgs[i] = torch.from_numpy(enh_rgb).permute(2, 0, 1).to(imgs.device)
                trainer.batch["img"] = imgs
            except Exception:
                pass  # Never crash training due to enhancement error

        print("  ✅ Ancuti on-the-fly enhancement callback loaded")
        return enhance_batch

    except ImportError as e:
        print(f"  ⚠️  Enhancement callback not loaded: {e}")
        print("     Training continues without on-the-fly enhancement")
        return None


# ── Training configuration ────────────────────────────────────────────────────

def build_train_args(args, reduce_batch: bool) -> dict:
    """Build the full YOLOv8 training argument dictionary."""
    batch = args.batch
    if reduce_batch:
        batch = max(8, batch // 2)
        print(f"  Batch size reduced to {batch} due to VRAM constraint")

    return {
        # ── Core ──────────────────────────────────────────────────────────
        "data":          args.data,
        "epochs":        args.epochs,
        "imgsz":         args.imgsz,
        "batch":         batch,
        "device":        0,                 # GPU 0 (RTX 3050)

        # ── Optimizer ─────────────────────────────────────────────────────
        "optimizer":     "AdamW",
        "lr0":           0.001,             # initial LR
        "lrf":           0.1,               # final LR = lr0 * lrf = 0.0001
        "momentum":      0.937,
        "weight_decay":  0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "cos_lr":        True,              # cosine LR decay

        # ── Augmentation (YOLOv8 built-in) ────────────────────────────────
        "mosaic":        1.0,               # mosaic augmentation
        "mixup":         0.1,               # mixup augmentation
        "hsv_h":         0.015,             # HSV hue shift
        "hsv_s":         0.7,               # HSV saturation
        "hsv_v":         0.4,               # HSV value
        "fliplr":        0.5,               # horizontal flip
        "flipud":        0.0,               # no vertical flip (fish don't swim upside down)
        "degrees":       15.0,              # rotation ±15°
        "translate":     0.1,               # translation
        "scale":         0.5,               # scale
        "shear":         0.0,               # no shear

        # ── Loss weights ──────────────────────────────────────────────────
        "box":           7.5,
        "cls":           0.5,
        "dfl":           1.5,

        # ── Validation & saving ───────────────────────────────────────────
        "val":           True,
        "save":          True,
        "save_period":   10,                # save checkpoint every 10 epochs
        "patience":      20,                # early stopping
        "exist_ok":      True,

        # ── Output ────────────────────────────────────────────────────────
        "project":       "results/detection",
        "name":          "baseline_yolov8s",
        "plots":         True,              # auto-generate training curves
        "verbose":       True,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3A — YOLOv8s baseline training")
    p.add_argument("--data",    default="dataset/processed/data.yaml",
                   help="Path to data.yaml")
    p.add_argument("--weights", default="yolov8s.pt",
                   help="Pretrained weights (default: yolov8s.pt)")
    p.add_argument("--epochs",  type=int, default=100)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--no-enhance", action="store_true",
                   help="Disable on-the-fly Ancuti enhancement")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Environment check
    reduce_batch = not check_environment()

    # Verify data.yaml exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"\nERROR: data.yaml not found at {data_path}")
        print("Run Phase 1D first: python src/utils/phase1d_split_dataset.py")
        return

    # Verify weights exist
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"\nWARNING: {args.weights} not found locally")
        print("Ultralytics will download it automatically...")

    print(f"\n── Training Configuration ─────────────────────────────────────")
    print(f"  Data     : {args.data}")
    print(f"  Weights  : {args.weights}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Img size : {args.imgsz}×{args.imgsz}")
    print(f"  Classes  : 3 (Butterflyfish, Parrotfish, Angelfish)")
    print(f"  Target   : mAP@0.5 > 85%")
    print(f"  Output   : results/detection/baseline_yolov8s/")

    # Load enhancement callback
    enhance_cb = None
    if not args.no_enhance:
        enhance_cb = make_enhance_callback()

    # Import YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nERROR: ultralytics not installed")
        print("Run: pip install ultralytics")
        return

    # Load model
    print(f"\nLoading {args.weights}...")
    model = YOLO(args.weights)

    # Register enhancement callback
    if enhance_cb is not None:
        model.add_callback("on_train_batch_start", enhance_cb)
        print("  On-the-fly Ancuti enhancement: ENABLED (20% of batches)")
    else:
        print("  On-the-fly enhancement: DISABLED")

    # Build training config
    train_cfg = build_train_args(args, reduce_batch)

    print(f"\n{'='*62}")
    print(f"  TRAINING START — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*62}\n")

    start = time.time()

    # Train
    results = model.train(**train_cfg)

    elapsed = time.time() - start
    hours   = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print(f"\n{'='*62}")
    print(f"  TRAINING COMPLETE in {hours}h {minutes}m")
    print(f"{'='*62}")

    # Print key metrics
    try:
        metrics = results.results_dict
        map50   = metrics.get("metrics/mAP50(B)",   0)
        map5095 = metrics.get("metrics/mAP50-95(B)", 0)
        prec    = metrics.get("metrics/precision(B)", 0)
        rec     = metrics.get("metrics/recall(B)",    0)

        print(f"\n── Final Metrics ──────────────────────────────────────────────")
        print(f"  mAP@0.5      : {map50:.4f}  (target > 0.85)")
        print(f"  mAP@0.5:0.95 : {map5095:.4f}")
        print(f"  Precision    : {prec:.4f}")
        print(f"  Recall       : {rec:.4f}")

        if map50 >= 0.85:
            print(f"\n  TARGET MET ✅  mAP@0.5 = {map50:.4f} > 0.85")
            print("  Proceed to Phase 3C validation!")
        elif map50 >= 0.80:
            print(f"\n  CLOSE ⚠️  mAP@0.5 = {map50:.4f}")
            print("  Run Phase 3B hyperparameter tuning")
        else:
            print(f"\n  BELOW TARGET ❌  mAP@0.5 = {map50:.4f}")
            print("  Run Phase 3B — hyperparameter tuning required")

    except Exception as e:
        print(f"  Could not read final metrics: {e}")
        print("  Check results/detection/baseline_yolov8s/results.csv")

    best_weights = Path("results/detection/baseline_yolov8s/weights/best.pt")
    print(f"\n  Best weights : {best_weights}")
    print(f"  Training log : results/detection/baseline_yolov8s/results.csv")
    print(f"\nNext: python src/training/phase3c_evaluate.py")


if __name__ == "__main__":
    main()
