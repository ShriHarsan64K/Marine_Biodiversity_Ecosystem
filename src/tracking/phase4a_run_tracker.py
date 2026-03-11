"""
PHASE 4A — Marine Fish Tracker (BoT-SORT + ByteTrack)
======================================================
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

AUTO-ROUTING — just pass --source, the script picks the right tracker:
  - Image directory / single image  →  ByteTrack  (no optical flow, per-image safe)
  - Video file / webcam index       →  BoT-SORT   (temporal GMC optical flow)

Usage:
  # Image directory (test set)
  python src/tracking/phase4a_run_tracker.py ^
    --source dataset\processed\test\images

  # Video file
  python src/tracking/phase4a_run_tracker.py ^
    --source path\to\reef_video.mp4 --save-video

  # Webcam
  python src/tracking/phase4a_run_tracker.py --source 0
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES  = {0: "Butterflyfish", 1: "Parrotfish", 2: "Angelfish"}
CLASS_COLORS = {0: (0, 200, 255), 1: (0, 165, 255), 2: (255, 128, 0)}  # BGR
WEIGHTS_PATH = r"results\detection\baseline_yolov8s\weights\best.pt"
OUTPUT_DIR   = Path(r"results\tracking")
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VID_EXTS     = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv"}


# ── Argument parser ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Phase 4A — Marine Fish Tracker")
    p.add_argument("--weights",    default=WEIGHTS_PATH)
    p.add_argument("--source",     required=True,
                   help="Image dir / video file / webcam index  (auto-detected)")
    p.add_argument("--conf",       type=float, default=0.45)
    p.add_argument("--iou",        type=float, default=0.45)
    p.add_argument("--imgsz",      type=int,   default=640)
    p.add_argument("--save-video", action="store_true",
                   help="Save annotated output video (video mode only)")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--max-frames", type=int,   default=None)
    return p.parse_args()


# ── Auto source-type detection ────────────────────────────────────────────────
def detect_mode(source: str) -> str:
    """
    Returns 'images' or 'video'.

    Decision logic:
      webcam int  (\"0\", \"1\", ...)    → video   (BoT-SORT with GMC)
      directory with image files     → images  (ByteTrack per-image)
      single image file              → images
      video file extension           → video
      anything else                  → video   (safe fallback)
    """
    try:
        int(source)
        return "video"          # webcam index
    except ValueError:
        pass

    p = Path(source)
    if p.is_dir():
        return "images" if any(f.suffix.lower() in IMG_EXTS for f in p.iterdir()) \
               else "video"
    if p.suffix.lower() in IMG_EXTS:
        return "images"
    return "video"


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_trail(frame, history, color, max_len=30):
    pts = history[-max_len:]
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = tuple(int(v * alpha) for v in color)
        cv2.line(frame, pts[i - 1], pts[i], c, 2)


def draw_dashboard(frame, live, unique_total, fps, frame_idx):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (290, 170), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    lines = [
        f"Frame: {frame_idx:5d}   FPS: {fps:5.1f}",
        "-----------------------------",
        "  Species        Live  Unique",
        "-----------------------------",
    ]
    for name in CLASS_NAMES.values():
        lines.append(
            f"  {name:<14s}  {live.get(name, 0):3d}   {unique_total.get(name, 0):4d}"
        )
    lines.append("-----------------------------")
    lines.append(f"  TOTAL UNIQUE         {sum(unique_total.values()):4d}")
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (18, 30 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


# ── Shared session state ──────────────────────────────────────────────────────
class SessionState:
    def __init__(self):
        self.unique_ids    = {}
        self.id_counts     = defaultdict(set)
        self.frame_counts  = []
        self.track_history = defaultdict(list)

    def process_boxes(self, boxes, frame, frame_idx):
        live = defaultdict(int)
        if boxes is None or len(boxes) == 0:
            return live
        for box in boxes:
            if box.id is None:
                continue
            xyxy  = box.xyxy[0].cpu().numpy().astype(int)
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            tid   = int(box.id[0])
            name  = CLASS_NAMES.get(cls, f"cls{cls}")
            color = CLASS_COLORS.get(cls, (200, 200, 200))
            x1, y1, x2, y2 = xyxy
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self.track_history[tid].append((cx, cy))
            draw_trail(frame, self.track_history[tid], color)
            self.unique_ids[tid] = name
            self.id_counts[name].add(tid)
            live[name] += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{tid} {name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return live

    def unique_totals(self):
        return {name: len(ids) for name, ids in self.id_counts.items()}

    def build_summary(self, frames_processed, output_dir):
        unique_counts = self.unique_totals()
        total = sum(unique_counts.values())
        summary = {
            "phase": "4A",
            "frames_processed": frames_processed,
            "unique_fish_total": total,
            "unique_per_class": unique_counts,
            "total_track_ids": len(self.unique_ids),
        }
        out = output_dir / "tracking_summary.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'='*62}")
        print(f"  PHASE 4A — TRACKING SUMMARY")
        print(f"{'='*62}")
        print(f"  Frames processed : {frames_processed}")
        print(f"  Total unique fish: {total}")
        for name, count in unique_counts.items():
            print(f"    {name:<16s}: {count}")
        print(f"\n  Summary saved : {out}")
        print(f"{'='*62}\n")
        return summary


# ── CSV helpers ───────────────────────────────────────────────────────────────
def open_csv(output_dir):
    csv_path = output_dir / "tracking_log.csv"
    f = open(csv_path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["frame", "track_id", "class_id", "class_name",
                "x1", "y1", "x2", "y2", "conf"])
    print(f"  Saving CSV    : {csv_path}")
    return f, w


def write_csv_row(writer, frame_idx, boxes):
    if boxes is None or boxes.id is None:
        return
    for box in boxes:
        tid  = int(box.id[0])
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        writer.writerow([frame_idx, tid, cls, CLASS_NAMES.get(cls, "?"),
                         x1, y1, x2, y2, round(conf, 4)])


# ══════════════════════════════════════════════════════════════════════════════
# MODE A — IMAGE DIRECTORY
# Tracker  : ByteTrack  (no GMC / optical flow → safe for mixed-resolution images)
# persist  : False      (each image is independent, no cross-frame track state)
# Resize   : all frames → imgsz×imgsz  (prevents pyramid size mismatch)
# ══════════════════════════════════════════════════════════════════════════════
def track_images(model, source, conf, iou, imgsz, output_dir, max_frames):
    imgs = sorted(p for p in Path(source).iterdir()
                  if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise RuntimeError(f"No images found in: {source}")
    if max_frames:
        imgs = imgs[:max_frames]

    print(f"  Source        : {source}  ({len(imgs)} images)")
    print(f"  Conf / IOU    : {conf} / {iou}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    state = SessionState()
    csv_f, csv_w = open_csv(output_dir)
    print()

    try:
        for frame_idx, img_path in enumerate(imgs):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            # Uniform size → no optical-flow pyramid crash
            frame = cv2.resize(frame, (imgsz, imgsz))

            results = model.track(
                frame,
                persist = False,            # independent images — no state carry-over
                tracker = "bytetrack.yaml", # no GMC → safe for image sets
                conf    = conf,
                iou     = iou,
                imgsz   = imgsz,
                verbose = False,
            )

            boxes = results[0].boxes
            live  = state.process_boxes(boxes, frame, frame_idx)
            write_csv_row(csv_w, frame_idx, boxes)
            state.frame_counts.append({
                "frame": frame_idx,
                "live":  dict(live),
                "unique": state.unique_totals(),
            })

            if frame_idx % 50 == 0 or frame_idx == len(imgs) - 1:
                print(f"  Frame {frame_idx+1:5d} / {len(imgs)} | "
                      f"Unique: {state.unique_totals()}")
    finally:
        csv_f.close()

    return state.build_summary(len(imgs), output_dir)


# ══════════════════════════════════════════════════════════════════════════════
# MODE B — VIDEO FILE / WEBCAM
# Tracker  : BoT-SORT  (CVPR 2023, GMC optical flow for occlusion handling)
# persist  : True       (maintains track IDs across consecutive frames)
# ══════════════════════════════════════════════════════════════════════════════
def track_video(model, source, conf, iou, imgsz, output_dir, save_video, max_frames):
    try:
        src = int(source)
    except (ValueError, TypeError):
        src = source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Source        : {source}")
    print(f"  Resolution    : {w}x{h}  @  {fps_in:.1f} FPS")
    print(f"  Total frames  : {total if total > 0 else 'live'}")
    print(f"  Conf / IOU    : {conf} / {iou}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    state = SessionState()
    csv_f, csv_w = open_csv(output_dir)

    writer = None
    if save_video:
        out_path = output_dir / "tracked_output.mp4"
        writer   = cv2.VideoWriter(str(out_path),
                                   cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
        print(f"  Saving video  : {out_path}")

    print("  Press [Q] to stop early\n")
    frame_idx  = 0
    t_prev     = time.time()
    fps_smooth = fps_in

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            results = model.track(
                frame,
                persist = True,             # carry IDs across video frames
                tracker = "botsort.yaml",   # GMC optical flow for occlusion handling
                conf    = conf,
                iou     = iou,
                imgsz   = imgsz,
                verbose = False,
            )

            boxes = results[0].boxes
            live  = state.process_boxes(boxes, frame, frame_idx)
            write_csv_row(csv_w, frame_idx, boxes)

            t_now      = time.time()
            fps_smooth = 0.9 * fps_smooth + 0.1 / max(t_now - t_prev, 1e-6)
            t_prev     = t_now

            draw_dashboard(frame, live, state.unique_totals(), fps_smooth, frame_idx)
            if writer:
                writer.write(frame)
            cv2.imshow("Marine Tracker  (Q=quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  Stopped by user.")
                break

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  Frame {frame_idx:5d} | FPS {fps_smooth:5.1f} | "
                      f"Unique: {state.unique_totals()}")
    finally:
        cap.release()
        if writer:
            writer.release()
        csv_f.close()
        cv2.destroyAllWindows()

    return state.build_summary(frame_idx, output_dir)


# ── Main — auto-routes based on source type ───────────────────────────────────
def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)

    print(f"\n{'='*62}")
    print(f"  PHASE 4A — Marine Fish Tracker")
    print(f"{'='*62}")
    print(f"  Loading model : {args.weights}")

    model = YOLO(args.weights)
    mode  = detect_mode(args.source)

    if mode == "images":
        print(f"  Mode          : IMAGE DIRECTORY")
        print(f"  Tracker       : ByteTrack  (no optical flow — image-safe)")
        track_images(model, args.source, args.conf, args.iou,
                     args.imgsz, output_dir, args.max_frames)
    else:
        print(f"  Mode          : VIDEO / WEBCAM")
        print(f"  Tracker       : BoT-SORT   (GMC optical flow — video-optimised)")
        track_video(model, args.source, args.conf, args.iou,
                    args.imgsz, output_dir, args.save_video, args.max_frames)


if __name__ == "__main__":
    main()