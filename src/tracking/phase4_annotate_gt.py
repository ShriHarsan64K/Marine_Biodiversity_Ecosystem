"""
PHASE 4 — Ground Truth Annotation Tool
=======================================
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Use this to manually annotate 100 frames for MOTA validation.
Outputs: ground_truth.csv  (frame, track_id, class_name, x1, y1, x2, y2)

Controls:
  SPACE / N  → next frame
  B          → back one frame
  1 / 2 / 3  → select class (1=Butterflyfish, 2=Parrotfish, 3=Angelfish)
  Click+Drag → draw bounding box
  R          → redo last box on this frame
  S          → save CSV and quit
  Q          → quit without saving

Usage:
  python src/tracking/phase4_annotate_gt.py ^
    --source  path/to/video.mp4 ^
    --output  results/tracking/ground_truth.csv ^
    --frames  100
"""

import argparse
import csv
import cv2
from pathlib import Path

CLASS_NAMES  = {1: "Butterflyfish", 2: "Parrotfish", 3: "Angelfish"}
CLASS_COLORS = {1: (0, 200, 255),   2: (0, 165, 255), 3: (255, 128, 0)}

# Per-frame annotations: {frame_idx: [{"tid", "cls", "box":[x1,y1,x2,y2]}]}
annotations: dict[int, list[dict]] = {}
current_class = 1
next_track_id = 1


def draw_state(frame, frame_annotations, frame_idx, current_cls):
    canvas = frame.copy()
    h, w   = canvas.shape[:2]

    # Existing boxes
    for ann in frame_annotations:
        color = CLASS_COLORS.get(ann["cls_key"], (200, 200, 200))
        x1, y1, x2, y2 = ann["box"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{ann['tid']}  {CLASS_NAMES[ann['cls_key']]}"
        cv2.putText(canvas, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # HUD
    cls_name = CLASS_NAMES.get(current_cls, "?")
    color    = CLASS_COLORS.get(current_cls, (200, 200, 200))
    cv2.putText(canvas, f"Frame {frame_idx}  |  Class: [{current_cls}] {cls_name}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(canvas,
                "SPACE=next  B=back  1/2/3=class  R=undo  S=save+quit  Q=quit",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    return canvas


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", default=r"results\tracking\ground_truth.csv")
    p.add_argument("--frames", type=int, default=100)
    return p.parse_args()


def main():
    global current_class, next_track_id

    args = parse_args()
    cap  = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.source}")

    # Pre-load up to --frames frames
    frames = []
    while len(frames) < args.frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"\n  Loaded {len(frames)} frames from {args.source}")
    print(f"  Output: {args.output}\n")

    # Annotation state
    for i in range(len(frames)):
        annotations[i] = []

    idx        = 0
    drawing    = False
    ix, iy     = 0, 0
    temp_frame = None

    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing, ix, iy, temp_frame
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy  = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            temp_frame = draw_state(frames[idx], annotations[idx],
                                    idx, current_class).copy()
            color = CLASS_COLORS.get(current_class, (200, 200, 200))
            cv2.rectangle(temp_frame, (ix, iy), (x, y), color, 2)
            cv2.imshow("GT Annotator", temp_frame)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing    = False
            global next_track_id
            x1, y1 = min(ix, x), min(iy, y)
            x2, y2 = max(ix, x), max(iy, y)
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                annotations[idx].append({
                    "tid":     next_track_id,
                    "cls_key": current_class,
                    "cls":     CLASS_NAMES[current_class],
                    "box":     [x1, y1, x2, y2],
                })
                next_track_id += 1

    cv2.namedWindow("GT Annotator")
    cv2.setMouseCallback("GT Annotator", mouse_cb)

    while True:
        display = draw_state(frames[idx], annotations[idx], idx, current_class)
        cv2.imshow("GT Annotator", display)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord(" "), ord("n")):
            if idx < len(frames) - 1:
                idx += 1
        elif key == ord("b"):
            if idx > 0:
                idx -= 1
        elif key == ord("1"):
            current_class = 1
        elif key == ord("2"):
            current_class = 2
        elif key == ord("3"):
            current_class = 3
        elif key == ord("r"):
            if annotations[idx]:
                annotations[idx].pop()
        elif key == ord("s"):
            break
        elif key == ord("q"):
            print("  Quit without saving.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    # Write CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "class_name", "x1", "y1", "x2", "y2"])
        for frame_idx, anns in annotations.items():
            for a in anns:
                x1, y1, x2, y2 = a["box"]
                writer.writerow([
                    frame_idx, a["tid"], a["cls"], x1, y1, x2, y2
                ])

    total = sum(len(v) for v in annotations.values())
    print(f"\n  Saved {total} annotations to {args.output}")


if __name__ == "__main__":
    main()
