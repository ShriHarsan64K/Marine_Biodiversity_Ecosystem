#!/usr/bin/env python3
"""
PHASE 3B — Hyperparameter Tuning
3-Class Marine Biodiversity Project
Classes: 0=Butterflyfish, 1=Parrotfish, 2=Angelfish

Only run this if Phase 3A baseline mAP@0.5 < 0.80.

Tunes:
  - Image size     : [512, 640, 800]
  - Batch size     : [8, 16]
  - Learning rate  : [0.0005, 0.001, 0.002]
  - Mosaic         : [0.5, 1.0]

Each trial runs for 30 epochs (fast scan), then best config
is trained for 100 epochs.

Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Usage:
    python src/training/phase3b_tune.py \
        --data    dataset/processed/data.yaml \
        --weights yolov8s.pt \
        --trials  6
"""

import argparse
import csv
import itertools
import json
import time
from pathlib import Path


# ── Search space ──────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    "imgsz":  [640, 512],           # image size
    "lr0":    [0.001, 0.0005],      # learning rate
    "mosaic": [1.0, 0.5],           # mosaic augmentation
}

FAST_EPOCHS  = 30    # epochs per trial
FINAL_EPOCHS = 100   # epochs for best config


# ── Trial runner ──────────────────────────────────────────────────────────────

def run_trial(model_weights: str, data: str, config: dict,
              trial_id: int, batch: int) -> dict:
    """Run a single training trial and return metrics."""
    from ultralytics import YOLO

    name = (f"tune_t{trial_id:02d}"
            f"_imgsz{config['imgsz']}"
            f"_lr{config['lr0']}"
            f"_mosaic{config['mosaic']}")

    print(f"\n── Trial {trial_id:02d}/{name} ────────────────────────────────────")
    print(f"   imgsz={config['imgsz']}  lr0={config['lr0']}  mosaic={config['mosaic']}")

    model = YOLO(model_weights)

    start = time.time()
    try:
        results = model.train(
            data         = data,
            epochs       = FAST_EPOCHS,
            imgsz        = config["imgsz"],
            batch        = batch,
            lr0          = config["lr0"],
            lrf          = 0.1,
            optimizer    = "AdamW",
            weight_decay = 0.0005,
            cos_lr       = True,
            mosaic       = config["mosaic"],
            mixup        = 0.1,
            hsv_h        = 0.015,
            hsv_s        = 0.7,
            hsv_v        = 0.4,
            fliplr       = 0.5,
            flipud       = 0.0,
            degrees      = 15.0,
            box          = 7.5,
            cls          = 0.5,
            dfl          = 1.5,
            patience     = 10,
            project      = "results/detection/tuning",
            name         = name,
            exist_ok     = True,
            verbose      = False,
            plots        = False,
        )
        elapsed = time.time() - start
        metrics = results.results_dict
        map50   = float(metrics.get("metrics/mAP50(B)",    0))
        map5095 = float(metrics.get("metrics/mAP50-95(B)", 0))
        prec    = float(metrics.get("metrics/precision(B)", 0))
        rec     = float(metrics.get("metrics/recall(B)",    0))

    except Exception as e:
        print(f"   ERROR: {e}")
        map50 = map5095 = prec = rec = 0.0
        elapsed = time.time() - start

    result = {
        "trial_id":  trial_id,
        "name":      name,
        "imgsz":     config["imgsz"],
        "lr0":       config["lr0"],
        "mosaic":    config["mosaic"],
        "batch":     batch,
        "map50":     round(map50,   4),
        "map5095":   round(map5095, 4),
        "precision": round(prec,    4),
        "recall":    round(rec,     4),
        "time_min":  round(elapsed / 60, 1),
    }

    print(f"   mAP@0.5={map50:.4f}  mAP@0.5:0.95={map5095:.4f}"
          f"  P={prec:.3f}  R={rec:.3f}  ({elapsed/60:.1f} min)")
    return result


# ── Best config re-train ───────────────────────────────────────────────────────

def train_best_config(model_weights: str, data: str,
                      best: dict, batch: int) -> None:
    """Train the best config found from tuning for FINAL_EPOCHS."""
    from ultralytics import YOLO

    print(f"\n{'='*62}")
    print(f"  BEST CONFIG — Full {FINAL_EPOCHS}-epoch training")
    print(f"  imgsz={best['imgsz']}  lr0={best['lr0']}  mosaic={best['mosaic']}")
    print(f"  mAP@0.5 in 30-epoch trial: {best['map50']:.4f}")
    print(f"{'='*62}\n")

    model = YOLO(model_weights)
    model.train(
        data         = data,
        epochs       = FINAL_EPOCHS,
        imgsz        = best["imgsz"],
        batch        = batch,
        lr0          = best["lr0"],
        lrf          = 0.1,
        optimizer    = "AdamW",
        weight_decay = 0.0005,
        cos_lr       = True,
        mosaic       = best["mosaic"],
        mixup        = 0.1,
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        fliplr       = 0.5,
        flipud       = 0.0,
        degrees      = 15.0,
        box          = 7.5,
        cls          = 0.5,
        dfl          = 1.5,
        patience     = 20,
        project      = "results/detection",
        name         = "tuned_yolov8s",
        exist_ok     = True,
        verbose      = True,
        plots        = True,
    )
    print("\nTuned model saved: results/detection/tuned_yolov8s/weights/best.pt")
    print("Next: python src/training/phase3c_evaluate.py --weights results/detection/tuned_yolov8s/weights/best.pt")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3B — Hyperparameter tuning")
    p.add_argument("--data",    default="dataset/processed/data.yaml")
    p.add_argument("--weights", default="yolov8s.pt")
    p.add_argument("--trials",  type=int, default=6,
                   help="Max trials to run (default: 6)")
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--skip-retrain", action="store_true",
                   help="Only run trials, skip best-config full retrain")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("╔" + "═" * 62 + "╗")
    print("║      PHASE 3B — HYPERPARAMETER TUNING                       ║")
    print("║      3-Class Marine Biodiversity Project                     ║")
    print("╚" + "═" * 62 + "╝\n")

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: {args.data} not found")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed — pip install ultralytics")
        return

    # Generate trial configs
    keys   = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    all_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    configs = all_configs[:args.trials]

    print(f"Search space: {len(all_configs)} combinations")
    print(f"Running     : {len(configs)} trials × {FAST_EPOCHS} epochs each")
    print(f"Batch size  : {args.batch}\n")

    trial_results = []
    for i, cfg in enumerate(configs, 1):
        result = run_trial(args.weights, args.data, cfg, i, args.batch)
        trial_results.append(result)

    # Sort by mAP50
    trial_results.sort(key=lambda x: x["map50"], reverse=True)

    # Print summary table
    print(f"\n{'='*72}")
    print(f"  TUNING SUMMARY")
    print(f"{'='*72}")
    print(f"  {'#':>3}  {'imgsz':>6}  {'lr0':>7}  {'mosaic':>7}  "
          f"{'mAP50':>7}  {'mAP50-95':>9}  {'time':>6}")
    print(f"  {'-'*68}")
    for i, r in enumerate(trial_results, 1):
        marker = " ← BEST" if i == 1 else ""
        print(f"  {i:>3}  {r['imgsz']:>6}  {r['lr0']:>7}  {r['mosaic']:>7}  "
              f"{r['map50']:>7.4f}  {r['map5095']:>9.4f}  {r['time_min']:>5.1f}m{marker}")

    # Save results
    out_dir = Path("results/detection/tuning")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(trial_results, f, indent=2)

    with open(out_dir / "tuning_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=trial_results[0].keys())
        writer.writeheader()
        writer.writerows(trial_results)

    print(f"\n  Results saved: {out_dir}/tuning_results.json")

    best = trial_results[0]
    print(f"\n  Best config  : imgsz={best['imgsz']}  lr0={best['lr0']}  mosaic={best['mosaic']}")
    print(f"  Best mAP@0.5 : {best['map50']:.4f}")

    if best["map50"] >= 0.85:
        print("\n  Already above target (>0.85) from 30-epoch trial!")
        print("  Full retrain will likely exceed target comfortably.")

    if not args.skip_retrain:
        train_best_config(args.weights, args.data, best, args.batch)


if __name__ == "__main__":
    main()
