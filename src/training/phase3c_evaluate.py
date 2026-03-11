#!/usr/bin/env python3
"""
PHASE 3C — Model Validation & Analysis
3-Class Marine Biodiversity Project
Classes: 0=Butterflyfish, 1=Parrotfish, 2=Angelfish

Runs full test set evaluation and generates:
  - Per-class mAP, Precision, Recall
  - Confusion matrix
  - Training curves plot
  - Inference speed benchmark (FPS)
  - Per-class metrics CSV
  - Phase 3 summary report

Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Usage:
    python src/training/phase3c_evaluate.py \
        --weights results/detection/baseline_yolov8s/weights/best.pt \
        --data    dataset/processed/data.yaml \
        --output  results/detection
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ["Butterflyfish", "Parrotfish", "Angelfish"]
TARGET_MAP50 = 0.85


# ── Validation ────────────────────────────────────────────────────────────────

def run_validation(weights: str, data: str, imgsz: int, output_dir: Path) -> dict:
    """Run YOLOv8 validation on the test set."""
    from ultralytics import YOLO

    print("\n── Running validation on test set ─────────────────────────────")
    model = YOLO(weights)

    results = model.val(
        data    = data,
        imgsz   = imgsz,
        split   = "test",
        project = str(output_dir),
        name    = "evaluation",
        exist_ok= True,
        verbose = True,
        plots   = True,
        save_json=True,
    )
    return results


# ── FPS Benchmark ─────────────────────────────────────────────────────────────

def benchmark_fps(weights: str, imgsz: int, n_runs: int = 100) -> dict:
    """Benchmark inference speed on RTX 3050."""
    import torch
    from ultralytics import YOLO

    print("\n── Inference Speed Benchmark ──────────────────────────────────")
    model = YOLO(weights)

    # Warm up
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    for _ in range(10):
        model.predict(source=dummy, verbose=False)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(source=dummy, verbose=False)
        times.append(time.perf_counter() - t0)

    avg_ms  = np.mean(times) * 1000
    std_ms  = np.std(times) * 1000
    fps     = 1.0 / np.mean(times)
    min_fps = 1.0 / np.max(times)
    max_fps = 1.0 / np.min(times)

    result = {
        "avg_latency_ms": round(avg_ms, 2),
        "std_latency_ms": round(std_ms, 2),
        "avg_fps":        round(fps, 1),
        "min_fps":        round(min_fps, 1),
        "max_fps":        round(max_fps, 1),
        "target_fps":     25,
        "fps_pass":       fps >= 25.0,
    }

    status = "PASS ✅" if result["fps_pass"] else "FAIL ❌"
    print(f"  Avg latency : {avg_ms:.2f} ms ± {std_ms:.2f} ms")
    print(f"  FPS         : {fps:.1f}  (target ≥ 25)  {status}")
    return result


# ── Per-class metrics ─────────────────────────────────────────────────────────

def extract_per_class_metrics(val_results) -> list:
    """Extract per-class AP, precision, recall from validation results."""
    metrics = []
    try:
        # ultralytics stores per-class AP in results.box.maps
        maps   = val_results.box.maps      # per-class mAP50
        ap50   = val_results.box.ap50      # per-class AP@0.5
        prec   = val_results.box.p         # per-class precision
        rec    = val_results.box.r         # per-class recall

        for i, name in enumerate(CLASS_NAMES):
            metrics.append({
                "class_id":   i,
                "class_name": name,
                "ap50":       round(float(ap50[i])  if i < len(ap50)  else 0, 4),
                "precision":  round(float(prec[i])  if i < len(prec)  else 0, 4),
                "recall":     round(float(rec[i])   if i < len(rec)   else 0, 4),
                "map50":      round(float(maps[i])  if i < len(maps)  else 0, 4),
            })
    except Exception as e:
        print(f"  Warning: Could not extract per-class metrics: {e}")
        for i, name in enumerate(CLASS_NAMES):
            metrics.append({
                "class_id": i, "class_name": name,
                "ap50": 0, "precision": 0, "recall": 0, "map50": 0,
            })
    return metrics


# ── Training curves plot ───────────────────────────────────────────────────────

def plot_training_curves(results_csv: Path, output_path: Path) -> None:
    """Parse results.csv and plot training curves."""
    if not results_csv.exists():
        print(f"  Warning: {results_csv} not found — skipping training curves")
        return

    epochs, train_loss, val_loss, map50_vals = [], [], [], []

    with open(results_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row.get("epoch", 0)))
                # Loss columns vary by ultralytics version
                tl = (float(row.get("train/box_loss", 0)) +
                      float(row.get("train/cls_loss", 0)) +
                      float(row.get("train/dfl_loss", 0)))
                vl = (float(row.get("val/box_loss", 0)) +
                      float(row.get("val/cls_loss", 0)) +
                      float(row.get("val/dfl_loss", 0)))
                m50 = float(row.get("metrics/mAP50(B)", 0))
                train_loss.append(tl)
                val_loss.append(vl)
                map50_vals.append(m50)
            except (ValueError, KeyError):
                continue

    if not epochs:
        print("  Warning: Could not parse results.csv")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, train_loss, label="Train Loss", color="#FF7043", linewidth=1.5)
    ax1.plot(epochs, val_loss,   label="Val Loss",   color="#42A5F5", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # mAP curve
    ax2.plot(epochs, map50_vals, color="#66BB6A", linewidth=1.5, label="mAP@0.5")
    ax2.axhline(TARGET_MAP50, color="red", linestyle="--",
                linewidth=1.2, label=f"Target ({TARGET_MAP50})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP@0.5")
    ax2.set_title("Validation mAP@0.5")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Mark best epoch
    if map50_vals:
        best_ep  = epochs[np.argmax(map50_vals)]
        best_map = max(map50_vals)
        ax2.scatter([best_ep], [best_map], color="red", zorder=5, s=60)
        ax2.annotate(f"Best: {best_map:.3f}\n(ep {best_ep})",
                     xy=(best_ep, best_map),
                     xytext=(best_ep + max(1, len(epochs)//10), best_map - 0.05),
                     fontsize=8, color="red")

    plt.suptitle("Phase 3 — YOLOv8s Training Curves (3-Class)", fontsize=13)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved: {output_path}")


# ── Per-class bar chart ───────────────────────────────────────────────────────

def plot_per_class_metrics(per_class: list, output_path: Path) -> None:
    """Bar chart of AP50, Precision, Recall per class."""
    names = [m["class_name"] for m in per_class]
    ap50  = [m["ap50"]       for m in per_class]
    prec  = [m["precision"]  for m in per_class]
    rec   = [m["recall"]     for m in per_class]

    x, w = np.arange(len(names)), 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w,   ap50, w, label="AP@0.5",    color="#42A5F5", alpha=0.88)
    ax.bar(x,       prec, w, label="Precision",  color="#66BB6A", alpha=0.88)
    ax.bar(x + w,   rec,  w, label="Recall",     color="#FF7043", alpha=0.88)
    ax.axhline(TARGET_MAP50, color="red", linestyle="--",
               linewidth=1.2, label=f"Target ({TARGET_MAP50})")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Phase 3C — Per-Class Detection Metrics")
    ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-class chart saved: {output_path}")


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_per_class_csv(per_class: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_class[0].keys())
        writer.writeheader()
        writer.writerows(per_class)
    print(f"  Per-class CSV saved : {output_path}")


def save_summary_report(val_results, per_class: list,
                        fps_results: dict, output_path: Path) -> None:
    """Save full Phase 3 summary as JSON."""
    try:
        overall = {
            "map50":     round(float(val_results.box.map50),  4),
            "map5095":   round(float(val_results.box.map),    4),
            "precision": round(float(val_results.box.mp),     4),
            "recall":    round(float(val_results.box.mr),     4),
        }
    except Exception:
        overall = {"map50": 0, "map5095": 0, "precision": 0, "recall": 0}

    report = {
        "project":     "Marine Biodiversity Ecosystem Health Assessment",
        "phase":       "3C",
        "classes":     CLASS_NAMES,
        "target_map50": TARGET_MAP50,
        "target_fps":  25,
        "overall":     overall,
        "per_class":   per_class,
        "fps":         fps_results,
        "pass":        overall["map50"] >= TARGET_MAP50 and fps_results.get("fps_pass", False),
    }

    # Convert any numpy types to native Python types for JSON serialization
    def convert(obj):
        import numpy as np
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    report = convert(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Summary report saved: {output_path}")
    return report


def print_final_report(overall: dict, per_class: list, fps: dict) -> None:
    map50 = overall.get("map50", 0)
    print(f"\n{'='*62}")
    print(f"  PHASE 3C — FINAL EVALUATION REPORT")
    print(f"{'='*62}")
    print(f"\n  ── Overall Metrics ───────────────────────────────────────")
    map_status = "PASS ✅" if map50 >= TARGET_MAP50 else "FAIL ❌"
    print(f"  mAP@0.5      : {map50:.4f}  (target > {TARGET_MAP50})  {map_status}")
    print(f"  mAP@0.5:0.95 : {overall.get('map5095', 0):.4f}")
    print(f"  Precision    : {overall.get('precision', 0):.4f}")
    print(f"  Recall       : {overall.get('recall', 0):.4f}")

    print(f"\n  ── Per-Class Results ─────────────────────────────────────")
    print(f"  {'Class':18s}  {'AP@0.5':>7}  {'Prec':>7}  {'Recall':>7}")
    print(f"  {'-'*48}")
    for m in per_class:
        flag = " ✅" if m["ap50"] >= TARGET_MAP50 else " ❌"
        print(f"  {m['class_name']:18s}  {m['ap50']:>7.4f}  "
              f"{m['precision']:>7.4f}  {m['recall']:>7.4f}{flag}")

    print(f"\n  ── Inference Speed ───────────────────────────────────────")
    fps_status = "PASS ✅" if fps.get("fps_pass") else "FAIL ❌"
    print(f"  FPS          : {fps.get('avg_fps', 0):.1f}  (target ≥ 25)  {fps_status}")
    print(f"  Latency      : {fps.get('avg_latency_ms', 0):.2f} ms")

    print(f"\n{'='*62}")
    if map50 >= TARGET_MAP50:
        print(f"  ALL TARGETS MET! 🎉")
        print(f"  Proceed to Phase 4 — Object Tracking (BoT-SORT)")
    else:
        print(f"  Target not met. Run Phase 3B tuning.")
        print(f"  python src/training/phase3b_tune.py")
    print(f"{'='*62}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3C — Model evaluation")
    p.add_argument("--weights",  default="results/detection/baseline_yolov8s/weights/best.pt")
    p.add_argument("--data",     default="dataset/processed/data.yaml")
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--output",   default="results/detection")
    p.add_argument("--fps-runs", type=int, default=100,
                   help="Number of FPS benchmark runs (default: 100)")
    p.add_argument("--skip-fps", action="store_true",
                   help="Skip FPS benchmark (faster evaluation)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("╔" + "═" * 62 + "╗")
    print("║      PHASE 3C — MODEL VALIDATION & ANALYSIS                 ║")
    print("║      3-Class Marine Biodiversity Project                     ║")
    print("╚" + "═" * 62 + "╝")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"\nERROR: Weights not found: {weights_path}")
        print("Run Phase 3A first: python src/training/phase3a_train_baseline.py")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics")
        return

    # ── Validation ────────────────────────────────────────────────────
    val_results = run_validation(args.weights, args.data, args.imgsz, output_dir)
    per_class   = extract_per_class_metrics(val_results)

    # ── FPS Benchmark ─────────────────────────────────────────────────
    fps_results = {}
    if not args.skip_fps:
        fps_results = benchmark_fps(args.weights, args.imgsz, args.fps_runs)
    else:
        print("\n  FPS benchmark skipped (--skip-fps)")

    # ── Plots ─────────────────────────────────────────────────────────
    # Training curves from results.csv
    run_name     = weights_path.parents[1].name
    results_csv  = output_dir / run_name / "results.csv"
    plot_training_curves(results_csv, output_dir / "training_curves.png")
    plot_per_class_metrics(per_class, output_dir / "per_class_metrics.png")

    # ── Save outputs ──────────────────────────────────────────────────
    save_per_class_csv(per_class, output_dir / "per_class_metrics.csv")

    try:
        overall = {
            "map50":     round(float(val_results.box.map50), 4),
            "map5095":   round(float(val_results.box.map),   4),
            "precision": round(float(val_results.box.mp),    4),
            "recall":    round(float(val_results.box.mr),    4),
        }
    except Exception:
        overall = {"map50": 0, "map5095": 0, "precision": 0, "recall": 0}

    save_summary_report(val_results, per_class, fps_results,
                        output_dir / "phase3_report.json")

    # ── Final report ──────────────────────────────────────────────────
    print_final_report(overall, per_class, fps_results)


if __name__ == "__main__":
    main()