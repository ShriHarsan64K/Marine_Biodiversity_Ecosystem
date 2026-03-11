"""
PHASE 6 — Live Tracker & MHI Engine
=====================================
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

HOW IT WORKS:
  1. Opens video file OR webcam
  2. Runs YOLOv8s detection + ByteTrack tracking on every frame
  3. Maintains a CUMULATIVE count of all detections since session start
  4. Every UPDATE_INTERVAL seconds:
       - Recomputes species counts from the cumulative total
       - Recalculates all Phase 5 biodiversity indices + MHI
       - Writes results to  results/dashboard/live_state.json
  5. The dashboard HTML polls live_state.json every second and updates live

USAGE:
  # Video file:
  python src/dashboard/phase6_tracker.py --source path/to/video.mp4

  # Webcam:
  python src/dashboard/phase6_tracker.py --source 0

  # Custom update interval:
  python src/dashboard/phase6_tracker.py --source video.mp4 ^
      --interval 2 --conf 0.4

  # Save annotated output video:
  python src/dashboard/phase6_tracker.py --source video.mp4 --save

KEYBOARD:
  q  →  quit
  p  →  pause / resume
  r  →  reset cumulative counts (start fresh mid-session)
"""

import argparse
import json
import sys
import time
import cv2
import numpy as np
from collections import Counter
from datetime import datetime
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from src.biodiversity.phase5a_indices import compute_all_indices
from src.biodiversity.phase5b_weighted import weighted_shannon
from src.biodiversity.phase5c_trophic  import trophic_analysis
from src.biodiversity.phase5d_mhi      import compute_mhi

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = ROOT / "results" / "detection" / "baseline_yolov8s" / "weights" / "best.pt"
OUTPUT_DIR      = ROOT / "results" / "dashboard"
LIVE_STATE_FILE = OUTPUT_DIR / "live_state.json"
LOG_CSV_FILE    = OUTPUT_DIR / "session_log.csv"

CLASS_NAMES = {0: "Butterflyfish", 1: "Parrotfish", 2: "Angelfish"}
COLORS      = {
    "Butterflyfish": (0, 212, 255),   # cyan
    "Parrotfish":    (0, 255, 157),   # green
    "Angelfish":     (160, 100, 255), # purple
}

# ── Argument parser ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Phase 6 Live Tracker")
    p.add_argument("--source",   default="0",
                   help="Video file path or webcam index (default: 0)")
    p.add_argument("--model",    default=str(MODEL_PATH),
                   help="Path to best.pt")
    p.add_argument("--conf",     type=float, default=0.35,
                   help="Detection confidence threshold")
    p.add_argument("--window",   type=int,   default=0,
                   help="Unused (cumulative mode). Kept for CLI compatibility.")
    p.add_argument("--interval", type=float, default=1.0,
                   help="MHI update interval in seconds (default: 1.0)")
    p.add_argument("--site",     default="Live Reef Survey",
                   help="Site name shown in dashboard")
    p.add_argument("--save",     action="store_true",
                   help="Save annotated output video")
    p.add_argument("--no-display", action="store_true",
                   help="Run headless (no cv2 window)")
    p.add_argument("--imgsz",    type=int, default=640)
    return p.parse_args()


# ── State writer ──────────────────────────────────────────────────────────────
def write_live_state(state: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LIVE_STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(LIVE_STATE_FILE)   # atomic replace


# ── MHI builder ──────────────────────────────────────────────────────────────
def build_state(counts: dict, site: str, frame_no: int,
                fps: float, session_start: float) -> dict:
    """
    Given {species: count} cumulative totals, compute full MHI state
    and return the dict that will be written to live_state.json.
    """
    if sum(counts.values()) == 0:
        counts = {"Butterflyfish": 0, "Parrotfish": 0, "Angelfish": 0}

    mhi_result = compute_mhi(counts, site_name=site)

    elapsed = time.time() - session_start
    mm, ss  = divmod(int(elapsed), 60)

    state = {
        "site":             site,
        "timestamp":        datetime.now().isoformat(),
        "session_elapsed":  f"{mm:02d}:{ss:02d}",
        "frame":            frame_no,
        "fps_actual":       round(fps, 1),
        "counts":           counts,
        "total_fish":       sum(counts.values()),
        "mhi":              mhi_result["mhi"],
        "grade":            mhi_result["grade"],
        "alert":            mhi_result["alert"],
        "components":       mhi_result["components"],
        "indices":          mhi_result["indices"],
        "trophic": {
            "group_counts":           mhi_result["trophic"]["group_counts"],
            "group_pcts":             mhi_result["trophic"]["group_pcts"],
            "overall_trophic_status": mhi_result["trophic"]["overall_trophic_status"],
        },
        "ecological_signal":   mhi_result["ecological_signal"],
        "degradation_signals": mhi_result["degradation_signals"],
    }
    return state


# ── Frame annotation ──────────────────────────────────────────────────────────
def annotate_frame(frame, results, state, fps_display):
    """Draw bounding boxes, IDs, and a HUD overlay onto the frame."""
    H, W = frame.shape[:2]
    overlay = frame.copy()

    # ── Detections ──────────────────────────────────────────────────────────
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            cls_id   = int(boxes.cls[i].item())
            conf     = float(boxes.conf[i].item())
            track_id = int(boxes.id[i].item()) if boxes.id is not None else -1
            name     = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
            color    = COLORS.get(name, (200, 200, 200))

            # Box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            # Glow effect
            cv2.rectangle(overlay, (x1-2, y1-2), (x2+2, y2+2), color, 1)

            # Label background
            label = f"{name}  #{track_id}  {conf:.2f}" if track_id >= 0 else f"{name}  {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(overlay, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(overlay, label, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

    # Blend overlay (semi-transparent boxes)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # ── HUD panel (top-left) ─────────────────────────────────────────────────
    hud_w, hud_h = 320, 220
    hud = np.zeros((hud_h, hud_w, 3), dtype=np.uint8)
    hud[:] = (4, 15, 25)

    def put(text, x, y, scale=0.45, color=(180,220,240), thick=1):
        cv2.putText(hud, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    mhi  = state.get("mhi", 0.0)
    alrt = state.get("alert", "")
    alrt_col = (0,255,157) if alrt=="HEALTHY" else (255,194,0) if alrt=="WARNING" else (255,61,90)

    # Title
    put("REEF HEALTH MONITOR", 8, 18, 0.5, (0,229,204), 1)
    # Divider
    cv2.line(hud, (8,24), (hud_w-8,24), (0,100,120), 1)
    # MHI
    cv2.putText(hud, f"{mhi:.1f}", (8, 62), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, alrt_col, 2, cv2.LINE_AA)
    put("/100", 100, 62, 0.5, (100,160,180))
    put(f"Grade: {state.get('grade','')}  |  {alrt}", 8, 78, 0.42, alrt_col)
    cv2.line(hud, (8,86), (hud_w-8,86), (0,60,80), 1)

    # Species counts
    put("SPECIES COUNTS:", 8, 100, 0.4, (0,200,200))
    y = 116
    for sp, cnt in state.get("counts", {}).items():
        col = COLORS.get(sp, (180,180,180))
        put(f"  {sp:<16s} {cnt:>4}", 8, y, 0.4, col)
        y += 14

    # Indices
    cv2.line(hud, (8, y+2), (hud_w-8, y+2), (0,60,80), 1)
    y += 14
    idx = state.get("indices", {})
    put(f"H'={idx.get('shannon_H',0):.4f}  D={idx.get('simpsons_D',0):.4f}  J'={idx.get('pielou_J',0):.4f}",
        8, y, 0.38, (120,180,200))
    y += 14
    put(f"FPS:{fps_display:.0f}  Frame:{state.get('frame',0)}  T:{state.get('session_elapsed','--')}",
        8, y, 0.38, (80,120,140))

    # Blend HUD onto frame
    roi = frame[10:10+hud_h, 10:10+hud_w]
    mask = np.ones((hud_h, hud_w), dtype=np.float32) * 0.80
    frame[10:10+hud_h, 10:10+hud_w] = (
        hud * mask[:,:,None] + roi * (1 - mask[:,:,None])
    ).astype(np.uint8)

    # HUD border
    cv2.rectangle(frame, (10, 10), (10+hud_w, 10+hud_h), (0,180,180), 1)

    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Detect source type
    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    print(f"\n{'='*60}")
    print(f"  PHASE 6 — LIVE REEF HEALTH MONITOR")
    print(f"{'='*60}")
    print(f"  Source   : {src}")
    print(f"  Model    : {args.model}")
    print(f"  Conf     : {args.conf}")
    print(f"  Mode     : cumulative (all detections counted, none forgotten)")
    print(f"  Interval : {args.interval}s")
    print(f"  Output   : {LIVE_STATE_FILE}")
    print(f"{'='*60}\n")

    # Load model
    print("  Loading YOLOv8s model...")
    model = YOLO(args.model)
    print("  Model loaded ✅\n")

    # Open video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open source '{src}'")
        sys.exit(1)

    vid_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {vid_w}×{vid_h}  {vid_fps:.1f}fps  {total_fr} frames")

    # Output video writer
    writer = None
    if args.save:
        out_path = OUTPUT_DIR / "annotated_output.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(str(out_path), fourcc, vid_fps, (vid_w, vid_h))
        print(f"  Saving annotated video to: {out_path}")

    # Session log CSV
    log_file = open(LOG_CSV_FILE, "w", encoding="utf-8")
    log_file.write("timestamp,frame,mhi,grade,alert,butterflyfish,parrotfish,angelfish,"
                   "shannon_H,simpsons_D,pielou_J\n")

    # Cumulative detection counter — grows throughout session, never forgets
    cumulative = Counter()
    frame_no   = 0
    paused     = False
    session_start = time.time()
    last_update   = time.time()
    last_fps_time = time.time()
    fps_actual    = vid_fps
    frame_count_fps = 0

    # Write initial zeroed state immediately so dashboard shows something
    init_state = build_state(
        {"Butterflyfish":0,"Parrotfish":0,"Angelfish":0},
        args.site, 0, 0.0, session_start
    )
    init_state["status"] = "initialising"
    write_live_state(init_state)
    print(f"  Dashboard state file: {LIVE_STATE_FILE}")
    print(f"\n  Press  Q = quit   P = pause/resume   R = reset window\n")

    current_state = init_state

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # End of video — loop back or exit
                if isinstance(src, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            frame_no += 1
            frame_count_fps += 1

            # ── Run YOLO tracking ──────────────────────────────────────────
            results = model.track(
                frame,
                persist=True,
                conf=args.conf,
                iou=0.5,
                tracker="bytetrack.yaml",
                imgsz=args.imgsz,
                verbose=False,
            )

            # ── Accumulate detections (cumulative — never forgotten) ───────
            if results and results[0].boxes is not None and results[0].boxes.cls is not None:
                for cls_id in results[0].boxes.cls.tolist():
                    name = CLASS_NAMES.get(int(cls_id))
                    if name:
                        cumulative[name] += 1

            # ── Update MHI every interval ──────────────────────────────────
            now = time.time()
            if now - last_update >= args.interval:
                # Use cumulative totals — all detections since session start
                counts_dict = {
                    "Butterflyfish": cumulative.get("Butterflyfish", 0),
                    "Parrotfish":    cumulative.get("Parrotfish",    0),
                    "Angelfish":     cumulative.get("Angelfish",     0),
                }

                # FPS
                elapsed_fps = now - last_fps_time
                if elapsed_fps > 0:
                    fps_actual = frame_count_fps / elapsed_fps
                last_fps_time   = now
                frame_count_fps = 0

                current_state = build_state(
                    counts_dict, args.site, frame_no,
                    fps_actual, session_start
                )
                current_state["status"]       = "running"
                current_state["total_detections"] = sum(cumulative.values())

                write_live_state(current_state)
                last_update = now

                # Log CSV
                idx = current_state["indices"]
                c   = current_state["counts"]
                log_file.write(
                    f"{current_state['timestamp']},{frame_no},"
                    f"{current_state['mhi']},{current_state['grade']},"
                    f"{current_state['alert']},"
                    f"{c.get('Butterflyfish',0)},{c.get('Parrotfish',0)},{c.get('Angelfish',0)},"
                    f"{idx.get('shannon_H',0):.6f},{idx.get('simpsons_D',0):.6f},{idx.get('pielou_J',0):.6f}\n"
                )
                log_file.flush()

                # Console summary
                print(f"\r  Frame {frame_no:>6}  "
                      f"MHI={current_state['mhi']:5.1f}  "
                      f"{current_state['grade']:<10s} "
                      f"BF={c.get('Butterflyfish',0):>4}  "
                      f"PF={c.get('Parrotfish',0):>4}  "
                      f"AF={c.get('Angelfish',0):>4}  "
                      f"FPS={fps_actual:.0f}",
                      end="", flush=True)

            # ── Annotate & display ─────────────────────────────────────────
            if not args.no_display or writer:
                ann = annotate_frame(frame.copy(), results, current_state, fps_actual)
                if writer:
                    writer.write(ann)
                if not args.no_display:
                    cv2.imshow("Reef Health Monitor — Phase 6 (Q=quit, P=pause, R=reset)", ann)

        # ── Key handling ───────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n\n  Quit requested.")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"\n  {'PAUSED' if paused else 'RESUMED'}")
        elif key == ord('r'):
            cumulative.clear()
            print("\n  Cumulative counts reset.")

    # ── Cleanup ────────────────────────────────────────────────────────────
    # Write final state
    final_state = dict(current_state)
    final_state["status"] = "finished"
    write_live_state(final_state)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    log_file.close()

    print(f"\n\n{'='*60}")
    print(f"  Session complete.")
    print(f"  Live state  : {LIVE_STATE_FILE}")
    print(f"  Session log : {LOG_CSV_FILE}")
    if args.save:
        print(f"  Output video: {OUTPUT_DIR / 'annotated_output.mp4'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()