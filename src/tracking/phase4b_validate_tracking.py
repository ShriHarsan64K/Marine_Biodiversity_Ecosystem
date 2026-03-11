"""
PHASE 4B — Tracking Validation & MOTA Metrics
==============================================
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Computes:
  - MOTA (Multi-Object Tracking Accuracy)
  - ID-Switch rate
  - Track fragmentation rate
  - Precision / Recall on detections

Usage:
  # Validate tracker output CSV against ground truth CSV
  python src/tracking/phase4b_validate_tracking.py ^
    --pred   results/tracking/tracking_log.csv ^
    --gt     results/tracking/ground_truth.csv ^
    --frames 100

  # Run tracker on test images and auto-evaluate
  python src/tracking/phase4b_validate_tracking.py ^
    --source dataset/processed/test/images ^
    --gt     results/tracking/ground_truth.csv

Ground-truth CSV format (create manually or via annotation tool):
  frame, track_id, class_name, x1, y1, x2, y2
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUTPUT_DIR  = Path(r"results\tracking")
CLASS_NAMES = {0: "Butterflyfish", 1: "Parrotfish", 2: "Angelfish"}


# ── IoU helper ───────────────────────────────────────────────────────────────

def iou_bbox(a: list[float], b: list[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


# ── CSV loaders ──────────────────────────────────────────────────────────────

def load_tracking_csv(path: str) -> dict[int, list[dict]]:
    """
    Load a tracking CSV.
    Expected columns: frame, track_id, class_name, x1, y1, x2, y2, [conf]
    Returns: {frame_id: [{"tid":..., "cls":..., "box":[x1,y1,x2,y2]}, ...]}
    """
    data: dict[int, list[dict]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            data[frame].append({
                "tid":  int(row["track_id"]),
                "cls":  row.get("class_name", "?"),
                "box":  [float(row["x1"]), float(row["y1"]),
                         float(row["x2"]), float(row["y2"])],
                "conf": float(row.get("conf", 1.0)),
            })
    return data


def load_gt_csv(path: str) -> dict[int, list[dict]]:
    """
    Load ground-truth CSV.
    Columns: frame, track_id, class_name, x1, y1, x2, y2
    Returns same structure as load_tracking_csv
    """
    data: dict[int, list[dict]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            data[frame].append({
                "tid":  int(row["track_id"]),
                "cls":  row.get("class_name", "?"),
                "box":  [float(row["x1"]), float(row["y1"]),
                         float(row["x2"]), float(row["y2"])],
            })
    return data


# ── MOTA computation ─────────────────────────────────────────────────────────

def compute_mota(gt_data: dict[int, list],
                 pred_data: dict[int, list],
                 iou_thresh: float = 0.45) -> dict:
    """
    Compute MOTA and supporting metrics over all frames.

    MOTA = 1 - (FN + FP + IDSW) / GT_total

    Returns dict with:
      mota, motp, precision, recall,
      tp, fp, fn, id_switches,
      track_fragmentations, mostly_tracked, mostly_lost
    """
    frames = sorted(set(gt_data) | set(pred_data))

    tp_total = fp_total = fn_total = 0
    id_switches   = 0
    iou_sum       = 0.0
    matched_count = 0

    # Track ID mapping: gt_tid → last matched pred_tid
    gt_to_pred: dict[int, int] = {}

    # Track continuity
    active_gt_ids: set[int] = set()
    track_life: dict[int, int] = defaultdict(int)   # gt_tid → frames tracked
    track_gaps: dict[int, int] = defaultdict(int)   # gt_tid → gap frames
    fragmentations: dict[int, int] = defaultdict(int)

    for frame in frames:
        gts   = gt_data.get(frame, [])
        preds = pred_data.get(frame, [])

        # Build cost matrix (IoU)
        if gts and preds:
            cost = np.zeros((len(gts), len(preds)))
            for i, g in enumerate(gts):
                for j, p in enumerate(preds):
                    cost[i, j] = iou_bbox(g["box"], p["box"])

            matched_gt   = set()
            matched_pred = set()
            # Greedy matching (descending IoU)
            pairs = sorted(
                [(cost[i, j], i, j)
                 for i in range(len(gts))
                 for j in range(len(preds))],
                reverse=True
            )
            frame_matches: list[tuple[int, int]] = []
            for iou_val, gi, pi in pairs:
                if iou_val < iou_thresh:
                    break
                if gi in matched_gt or pi in matched_pred:
                    continue
                matched_gt.add(gi)
                matched_pred.add(pi)
                frame_matches.append((gi, pi))
                iou_sum    += iou_val
                matched_count += 1

            tp_total += len(frame_matches)
            fn_total += len(gts)   - len(frame_matches)
            fp_total += len(preds) - len(frame_matches)

            # ID switch detection
            for gi, pi in frame_matches:
                gt_tid   = gts[gi]["tid"]
                pred_tid = preds[pi]["tid"]
                if gt_tid in gt_to_pred and gt_to_pred[gt_tid] != pred_tid:
                    id_switches += 1
                gt_to_pred[gt_tid] = pred_tid
                track_life[gt_tid] += 1

        else:
            fn_total += len(gts)
            fp_total += len(preds)

        # Track fragmentation: GT object reappears after absence
        current_gt_ids = {g["tid"] for g in gts}
        for tid in current_gt_ids:
            if tid in active_gt_ids:
                track_gaps[tid] = 0
            else:
                if track_gaps[tid] > 0:
                    fragmentations[tid] += 1
                active_gt_ids.add(tid)
        for tid in active_gt_ids - current_gt_ids:
            track_gaps[tid] += 1

    gt_total = tp_total + fn_total

    mota = 1.0 - (fn_total + fp_total + id_switches) / max(gt_total, 1)
    motp = iou_sum / max(matched_count, 1)
    precision = tp_total / max(tp_total + fp_total, 1)
    recall    = tp_total / max(gt_total, 1)

    # Mostly tracked / mostly lost
    mostly_tracked = mostly_lost = partial = 0
    for tid, life in track_life.items():
        gt_len = sum(1 for f_data in gt_data.values()
                     for g in f_data if g["tid"] == tid)
        ratio  = life / max(gt_len, 1)
        if ratio >= 0.80:
            mostly_tracked += 1
        elif ratio <= 0.20:
            mostly_lost += 1
        else:
            partial += 1

    return {
        "mota":                round(float(mota), 4),
        "motp":                round(float(motp), 4),
        "precision":           round(float(precision), 4),
        "recall":              round(float(recall), 4),
        "tp":                  int(tp_total),
        "fp":                  int(fp_total),
        "fn":                  int(fn_total),
        "id_switches":         int(id_switches),
        "track_fragmentations":int(sum(fragmentations.values())),
        "mostly_tracked":      int(mostly_tracked),
        "mostly_lost":         int(mostly_lost),
        "partial_tracked":     int(partial),
        "gt_total":            int(gt_total),
        "frames_evaluated":    len(frames),
    }


# ── Report printing ───────────────────────────────────────────────────────────

def print_report(metrics: dict) -> None:
    target_mota = 0.78   # roadmap target

    print(f"\n{'='*62}")
    print(f"  PHASE 4B — TRACKING VALIDATION REPORT")
    print(f"{'='*62}")
    print(f"  Frames evaluated      : {metrics['frames_evaluated']}")
    print(f"  Ground-truth objects  : {metrics['gt_total']}")
    print()
    print(f"  MOTA                  : {metrics['mota']:.4f}  "
          f"{'✅ PASS' if metrics['mota'] >= target_mota else '❌ FAIL'}  "
          f"(target ≥ {target_mota})")
    print(f"  MOTP (avg IoU)        : {metrics['motp']:.4f}")
    print(f"  Precision             : {metrics['precision']:.4f}")
    print(f"  Recall                : {metrics['recall']:.4f}")
    print()
    print(f"  True Positives        : {metrics['tp']}")
    print(f"  False Positives       : {metrics['fp']}")
    print(f"  False Negatives       : {metrics['fn']}")
    print(f"  ID Switches           : {metrics['id_switches']}")
    print(f"  Track Fragmentations  : {metrics['track_fragmentations']}")
    print()
    print(f"  Mostly Tracked (≥80%) : {metrics['mostly_tracked']}")
    print(f"  Partially Tracked     : {metrics['partial_tracked']}")
    print(f"  Mostly Lost   (≤20%)  : {metrics['mostly_lost']}")
    print(f"{'='*62}\n")


# ── Tracker runner (for --source mode) ───────────────────────────────────────

def run_tracker_on_source(source: str, weights: str,
                          conf: float, iou: float) -> Path:
    """Run phase4a tracker on source and return path to tracking_log.csv."""
    from ultralytics import YOLO
    import cv2
    import time

    model = YOLO(weights)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "tracking_log_val.csv"

    try:
        src = int(source)
    except ValueError:
        src = source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "class_name", "x1", "y1", "x2", "y2", "conf"])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame, persist=True, tracker="botsort.yaml",
                conf=conf, iou=iou, verbose=False
            )
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                for box in boxes:
                    tid  = int(box.id[0])
                    cls  = int(box.cls[0])
                    name = CLASS_NAMES.get(cls, "?")
                    c    = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    writer.writerow([frame_idx, tid, name, x1, y1, x2, y2, round(c, 4)])

            frame_idx += 1

    cap.release()
    print(f"  Tracker output: {csv_path}")
    return csv_path


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 4B — Tracking Validation")
    p.add_argument("--pred",    help="Predicted tracking CSV (from phase4a)")
    p.add_argument("--gt",      required=True, help="Ground-truth CSV")
    p.add_argument("--source",  help="Run tracker first (video/images)")
    p.add_argument("--weights", default=r"results\detection\baseline_yolov8s\weights\best.pt")
    p.add_argument("--conf",    type=float, default=0.45)
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return p.parse_args()


def main():
    args   = parse_args()
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pred_csv = args.pred
    if args.source:
        print(f"  Running tracker on: {args.source}")
        pred_csv = str(run_tracker_on_source(
            args.source, args.weights, args.conf, args.iou
        ))

    if not pred_csv:
        raise ValueError("Provide --pred or --source")

    print(f"\n  Loading predictions : {pred_csv}")
    print(f"  Loading ground truth: {args.gt}")

    pred_data = load_tracking_csv(pred_csv)
    gt_data   = load_gt_csv(args.gt)

    metrics = compute_mota(gt_data, pred_data)
    print_report(metrics)

    report_path = out / "validation_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {report_path}")


if __name__ == "__main__":
    main()
