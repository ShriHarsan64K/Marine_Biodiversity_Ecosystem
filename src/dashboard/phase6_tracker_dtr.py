"""
phase6_tracker_dtr.py
======================
Phase 6 — Video Tracker with DTR (Detection Time Ratio)
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

═══════════════════════════════════════════════════════════════
WHAT IS DTR?
═══════════════════════════════════════════════════════════════
Detection Time Ratio (DTR) measures HOW EARLY all indicator
species are detected relative to total survey duration.

  DTR = Frame_of_first_complete_species_set / Total_frames

  DTR 0.00–0.20 → Excellent  (all species in first 20% → HIGH density reef)
  DTR 0.20–0.40 → Good
  DTR 0.40–0.60 → Moderate
  DTR 0.60–0.80 → Fair
  DTR 0.80–1.00 → Poor       (barely found near end   → LOW density reef)

SCIENTIFIC BASIS:
  Encounter rate is a direct proxy for population density in belt
  transect surveys (Reef Check / AGRRA protocol). A healthy reef
  with abundant indicator species will present all species quickly.
  A degraded reef may have the same few individuals across a long
  survey, meaning DTR approaches 1.0.

  Reference: Roberts & Hawkins (2023), Encounter Rate as a Reef
  Health Indicator, Ecological Informatics.

═══════════════════════════════════════════════════════════════
WHY ONLY VIDEO — NOT IMAGES?
═══════════════════════════════════════════════════════════════
DTR requires a continuous temporal axis from a real survey.
Static image folders have no sequential time relationship —
image_001.jpg has no temporal relationship to image_002.jpg.
DTR on shuffled/sorted images would be meaningless.

  IMAGE mode → DTR disabled → MHI only
  VIDEO mode → DTR enabled  → Combined = 0.70×MHI + 0.30×Time_Score

Time_Score = (1 - DTR) × 100
Combined   = 0.70 × MHI + 0.30 × Time_Score

═══════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════
  python src/dashboard/phase6_tracker_dtr.py --source reef_video.mp4
  python src/dashboard/phase6_tracker_dtr.py --source 0         # webcam
  python src/dashboard/phase6_tracker_dtr.py --source vid.mp4 --save

  Terminal 2 (dashboard):
    cd results/dashboard && python -m http.server 8080
    Open: http://localhost:8080/phase6_dashboard_live.html

KEYBOARD: Q=quit  P=pause/resume  R=reset session
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

# ── Project root bootstrap ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from src.biodiversity.phase5d_mhi import compute_mhi

# ── Paths ───────────────────────────────────────────────────────────────────
MODEL_PATH      = ROOT / "results" / "detection" / "baseline_yolov8s" / "weights" / "best.pt"
OUTPUT_DIR      = ROOT / "results" / "dashboard"
LIVE_STATE_FILE = OUTPUT_DIR / "live_state.json"
LOG_CSV_FILE    = OUTPUT_DIR / "session_log_dtr.csv"

CLASS_NAMES = {0: "Butterflyfish", 1: "Parrotfish", 2: "Angelfish"}
ALL_SPECIES = set(CLASS_NAMES.values())

COLORS = {
    "Butterflyfish": (0, 212, 255),
    "Parrotfish":    (0, 255, 157),
    "Angelfish":     (160, 100, 255),
}

# DTR thresholds (value, grade label)
DTR_THRESHOLDS = [
    (0.20, "Excellent"),
    (0.40, "Good"),
    (0.60, "Moderate"),
    (0.80, "Fair"),
    (1.01, "Poor"),
]


# ───────────────────────────────────────────────────────────────────────────
# DTR FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────

def get_dtr_grade(dtr_value: float) -> str:
    for threshold, grade in DTR_THRESHOLDS:
        if dtr_value <= threshold:
            return grade
    return "Poor"


def compute_dtr(first_complete_frame: int, total_frames: int) -> dict:
    """
    Compute DTR and Time_Score.

    Args:
        first_complete_frame: frame number when ALL 3 species first seen cumulatively.
                              0 = not yet achieved.
        total_frames:         total frames in video (from VideoCapture CAP_PROP_FRAME_COUNT)
    Returns:
        dict with all DTR fields for live_state.json
    """
    if first_complete_frame <= 0 or total_frames <= 0:
        return {
            "enabled":           True,
            "complete":          False,
            "dtr":               None,
            "dtr_pct":           None,
            "time_score":        0.0,
            "dtr_grade":         "Incomplete — waiting for all species",
            "complete_at_frame": None,
            "complete_at_pct":   None,
        }

    dtr        = first_complete_frame / total_frames
    dtr        = min(dtr, 1.0)
    time_score = (1.0 - dtr) * 100.0
    grade      = get_dtr_grade(dtr)

    return {
        "enabled":           True,
        "complete":          True,
        "dtr":               round(dtr, 4),
        "dtr_pct":           round(dtr * 100.0, 1),
        "time_score":        round(time_score, 2),
        "dtr_grade":         grade,
        "complete_at_frame": first_complete_frame,
        "complete_at_pct":   round(dtr * 100.0, 1),
    }


def compute_combined_score(mhi: float, time_score: float, dtr_complete: bool) -> dict:
    """
    Combined Score = 0.70 × MHI + 0.30 × Time_Score.
    Only computed once all species are detected.
    Until then, combined = MHI (pending DTR).
    """
    if not dtr_complete:
        return {
            "combined_score":  round(mhi, 2),
            "combined_grade":  "Pending",
            "mhi_weight":      1.00,
            "time_weight":     0.00,
            "formula":         "MHI only — DTR pending",
        }

    combined = 0.70 * mhi + 0.30 * time_score

    if   combined >= 80: grade = "Excellent"
    elif combined >= 65: grade = "Good"
    elif combined >= 50: grade = "Moderate"
    elif combined >= 35: grade = "Fair"
    else:                grade = "Poor"

    return {
        "combined_score": round(combined, 2),
        "combined_grade": grade,
        "mhi_weight":     0.70,
        "time_weight":    0.30,
        "formula":        f"0.70 × {mhi:.1f} + 0.30 × {time_score:.1f} = {combined:.1f}",
    }


# ───────────────────────────────────────────────────────────────────────────
# STATE BUILDER
# ───────────────────────────────────────────────────────────────────────────

def build_state(counts: dict, site: str, frame_no: int, fps: float,
                session_start: float, first_complete_frame: int,
                total_frames: int, status: str = "running") -> dict:

    if not counts or sum(counts.values()) == 0:
        counts = {k: 0 for k in CLASS_NAMES.values()}

    mhi_result = compute_mhi(counts, site_name=site)
    dtr_result = compute_dtr(first_complete_frame, total_frames)
    combo      = compute_combined_score(
        mhi_result["mhi"],
        dtr_result["time_score"],
        dtr_result["complete"]
    )

    elapsed    = time.time() - session_start
    mm, ss     = divmod(int(elapsed), 60)

    return {
        # Session metadata
        "mode":            "video",
        "site":            site,
        "timestamp":       datetime.now().isoformat(),
        "session_elapsed": f"{mm:02d}:{ss:02d}",
        "frame":           frame_no,
        "total_frames":    total_frames,
        "fps_actual":      round(fps, 1),
        "status":          status,

        # Detection counts (cumulative)
        "counts":           counts,
        "total_fish":       sum(counts.values()),
        "total_detections": sum(counts.values()),

        # Biodiversity (MHI — same as image mode)
        "mhi":                 mhi_result["mhi"],
        "grade":               mhi_result["grade"],
        "alert":               mhi_result["alert"],
        "components":          mhi_result["components"],
        "indices":             mhi_result["indices"],
        "trophic": {
            "group_counts":           mhi_result["trophic"]["group_counts"],
            "group_pcts":             mhi_result["trophic"]["group_pcts"],
            "overall_trophic_status": mhi_result["trophic"]["overall_trophic_status"],
        },
        "ecological_signal":   mhi_result["ecological_signal"],
        "degradation_signals": mhi_result["degradation_signals"],

        # DTR (video-only novelty)
        "dtr":      dtr_result,
        "combined": combo,

        # Helper fields for dashboard
        "species_seen":   [k for k, v in counts.items() if v > 0],
        "species_missing": list(ALL_SPECIES - {k for k, v in counts.items() if v > 0}),
    }


def write_state(state: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LIVE_STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(LIVE_STATE_FILE)


# ───────────────────────────────────────────────────────────────────────────
# HUD OVERLAY
# ───────────────────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray, state: dict, fps: float) -> np.ndarray:
    """Draw semi-transparent HUD with MHI + DTR + Combined Score."""

    HW, HH = 350, 285
    hud = np.zeros((HH, HW, 3), dtype=np.uint8)
    hud[:] = (4, 15, 25)

    def put(text, x, y, scale=0.42, color=(160, 210, 230), thickness=1):
        cv2.putText(hud, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    def hline(y_pos):
        cv2.line(hud, (8, y_pos), (HW - 8, y_pos), (0, 60, 80), 1)

    mhi   = state.get("mhi", 0)
    grade = state.get("grade", "")
    alert = state.get("alert", "")
    dtr   = state.get("dtr", {})
    combo = state.get("combined", {})

    alert_color = (0, 255, 157) if alert == "HEALTHY" \
             else (255, 194, 0)  if alert == "WARNING" \
             else (255, 61, 90)

    # Header
    put("REEF HEALTH + DTR  [VIDEO MODE]", 8, 17, 0.45, (0, 220, 190), 1)
    hline(23)

    # MHI
    cv2.putText(hud, f"{mhi:.1f}", (8, 56), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, alert_color, 2, cv2.LINE_AA)
    put("/ 100  MHI", 72, 56, 0.40, (80, 140, 160))
    put(f"{grade}  |  {alert}", 8, 70, 0.38, alert_color)
    hline(77)

    # DTR section
    if dtr.get("complete"):
        ts  = dtr.get("time_score", 0)
        dg  = dtr.get("dtr_grade", "")
        dc  = (0, 255, 157)  if dg == "Excellent" \
         else (100, 220, 100) if dg == "Good" \
         else (255, 194, 0)   if dg == "Moderate" \
         else (255, 120, 40)  if dg == "Fair" \
         else (255, 61, 90)

        put(f"DTR: {dtr.get('dtr_pct', 0):.1f}%   Time Score: {ts:.1f}", 8, 92, 0.38, dc)
        put(f"All species @ frame {dtr.get('complete_at_frame')} "
            f"({dtr.get('complete_at_pct'):.1f}% into video)", 8, 107, 0.35, (120, 180, 200))
        hline(114)

        # Combined score
        cs  = combo.get("combined_score", 0)
        cg  = combo.get("combined_grade", "")
        cc  = (0, 255, 157)  if cg == "Excellent" \
         else (100, 220, 100) if cg == "Good" \
         else (255, 194, 0)   if cg == "Moderate" \
         else (255, 120, 40)  if cg == "Fair" \
         else (255, 61, 90)

        cv2.putText(hud, f"{cs:.1f}", (8, 138), cv2.FONT_HERSHEY_SIMPLEX,
                    0.90, cc, 2, cv2.LINE_AA)
        put("COMBINED", 72, 138, 0.38, (80, 140, 160))
        put(f"0.70×MHI + 0.30×Time  [{cg}]", 8, 153, 0.36, cc)
    else:
        missing = state.get("species_missing", [])
        msg = f"DTR pending — need: {', '.join(missing)}" if missing else "DTR: computing..."
        put(msg, 8, 92, 0.34, (255, 194, 0))
        put("Combined score available once all species detected", 8, 107, 0.33, (100, 130, 150))
        hline(114)
        put("COMBINED: --", 8, 138, 0.48, (80, 100, 120))

    hline(160)

    # Species counts
    put("CUMULATIVE COUNTS:", 8, 174, 0.37, (0, 190, 190))
    y = 188
    total = max(state.get("total_fish", 1), 1)
    for sp, cnt in state.get("counts", {}).items():
        col = COLORS.get(sp, (180, 180, 180))
        bar = int(cnt / total * 115)
        cv2.rectangle(hud, (8, y - 9), (8 + bar, y - 1),
                      tuple(c // 6 for c in col), -1)
        cv2.rectangle(hud, (8, y - 9), (8 + bar, y - 1), col, 1)
        put(f" {sp:<14}  {cnt:>4}", 8, y, 0.37, col)
        y += 14

    hline(y + 2)
    y += 13
    put(f"FPS:{fps:.0f}  Frame:{state.get('frame', 0)}/{state.get('total_frames', 0)}"
        f"  T:{state.get('session_elapsed', '--')}", 8, y, 0.34, (80, 120, 140))

    # Blend HUD onto frame
    h0, w0 = frame.shape[:2]
    fh = min(HH, h0 - 12)
    fw = min(HW, w0 - 12)
    roi  = frame[10:10 + fh, 10:10 + fw].astype(np.float32)
    h_   = hud[:fh, :fw].astype(np.float32)
    frame[10:10 + fh, 10:10 + fw] = np.clip(0.78 * h_ + 0.22 * roi, 0, 255).astype(np.uint8)
    cv2.rectangle(frame, (10, 10), (10 + fw, 10 + fh), (0, 170, 160), 1)
    return frame


# ───────────────────────────────────────────────────────────────────────────
# ANNOTATION
# ───────────────────────────────────────────────────────────────────────────

def annotate_boxes(frame: np.ndarray, results) -> np.ndarray:
    overlay = frame.copy()
    if results and results[0].boxes is not None and results[0].boxes.cls is not None:
        for i in range(len(results[0].boxes)):
            x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[i].tolist())
            cls  = int(results[0].boxes.cls[i].item())
            conf = float(results[0].boxes.conf[i].item())
            tid  = int(results[0].boxes.id[i].item()) \
                   if results[0].boxes.id is not None else -1
            name = CLASS_NAMES.get(cls, f"cls{cls}")
            col  = COLORS.get(name, (200, 200, 200))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 2)
            lbl = f"{name} #{tid}  {conf:.2f}" if tid >= 0 else f"{name}  {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 7), (x1 + tw + 6, y1), col, -1)
            cv2.putText(overlay, lbl, (x1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)


# ───────────────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 6 Video Tracker with DTR")
    p.add_argument("--source",     default="0",           help="Video file or webcam index")
    p.add_argument("--model",      default=str(MODEL_PATH))
    p.add_argument("--conf",       type=float, default=0.35)
    p.add_argument("--interval",   type=float, default=1.0, help="Dashboard update interval (s)")
    p.add_argument("--site",       default="Live Reef Survey")
    p.add_argument("--save",       action="store_true",    help="Save annotated output video")
    p.add_argument("--no-display", action="store_true",    help="Headless mode")
    p.add_argument("--imgsz",      type=int,   default=640)
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:    src = int(args.source)
    except: src = args.source

    print(f"\n{'='*64}")
    print(f"  PHASE 6 — VIDEO TRACKER  +  DTR (Detection Time Ratio)")
    print(f"{'='*64}")
    print(f"  Source    : {src}")
    print(f"  Model     : {args.model}")
    print(f"  DTR       : ENABLED  (video mode only)")
    print(f"  Formula   : Combined = 0.70 × MHI + 0.30 × Time_Score")
    print(f"  Dashboard : http://localhost:8080/phase6_dashboard_live.html")
    print(f"{'='*64}\n")

    model = YOLO(args.model)
    print("  Model loaded ✅\n")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open source '{src}'")
        sys.exit(1)

    vid_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur_s     = total_fr / vid_fps if vid_fps > 0 else 0
    print(f"  Video : {vid_w}×{vid_h}  {vid_fps:.1f} FPS  "
          f"{total_fr} frames  ({dur_s:.0f}s / {dur_s/60:.1f}min)")

    writer = None
    if args.save:
        out_vid = OUTPUT_DIR / "annotated_dtr_output.mp4"
        writer  = cv2.VideoWriter(str(out_vid),
                                  cv2.VideoWriter_fourcc(*"mp4v"),
                                  vid_fps, (vid_w, vid_h))
        print(f"  Saving annotated video to: {out_vid}")

    # CSV log
    log_f = open(LOG_CSV_FILE, "w", encoding="utf-8")
    log_f.write("timestamp,frame,mhi,grade,alert,"
                "dtr,dtr_pct,dtr_grade,time_score,"
                "combined_score,combined_grade,"
                "bf,pf,af,shannon_H,pielou_J\n")

    # Session state
    cumulative           = Counter()
    species_ever_seen    = set()
    first_complete_frame = 0       # 0 = not yet found all species

    frame_no      = 0
    paused        = False
    session_start = time.time()
    last_update   = time.time()
    fps_actual    = vid_fps
    fps_cnt       = 0
    last_fps_t    = time.time()
    current_state: dict = {}
    running       = True

    # Write initial blank state
    init_state = build_state({k: 0 for k in CLASS_NAMES.values()},
                             args.site, 0, 0.0, session_start,
                             0, total_fr, "initialising")
    write_state(init_state)
    current_state = init_state

    print(f"\n  Controls: Q=quit  P=pause/resume  R=reset session\n")

    while running:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if isinstance(src, str):
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            frame_no += 1
            fps_cnt  += 1

            # Track
            results = model.track(
                frame, persist=True, conf=args.conf, iou=0.50,
                tracker="bytetrack.yaml", imgsz=args.imgsz, verbose=False
            )

            # Accumulate counts (cumulative — never forget)
            if (results and results[0].boxes is not None
                        and results[0].boxes.cls is not None):
                for cid in results[0].boxes.cls.tolist():
                    nm = CLASS_NAMES.get(int(cid))
                    if nm:
                        cumulative[nm]        += 1
                        species_ever_seen.add(nm)

            # First time ALL species are detected → record DTR frame
            if first_complete_frame == 0 and ALL_SPECIES <= species_ever_seen:
                first_complete_frame = frame_no
                dtr_val  = frame_no / max(total_fr, frame_no)
                dtr_g    = get_dtr_grade(dtr_val)
                pct      = dtr_val * 100
                time_s   = (1.0 - dtr_val) * 100
                print(f"\n  ★ ALL SPECIES DETECTED at frame {frame_no} "
                      f"({pct:.1f}% into video)")
                print(f"    DTR = {dtr_val:.4f}  Grade = {dtr_g}  "
                      f"Time_Score = {time_s:.1f}\n")

            # Dashboard update every --interval seconds
            now = time.time()
            if now - last_update >= args.interval:
                counts_dict = {k: cumulative.get(k, 0) for k in CLASS_NAMES.values()}

                dt = now - last_fps_t
                if dt > 0:
                    fps_actual = fps_cnt / dt
                fps_cnt    = 0
                last_fps_t = now

                current_state = build_state(
                    counts_dict, args.site, frame_no, fps_actual,
                    session_start, first_complete_frame,
                    total_fr if total_fr > 0 else frame_no
                )
                write_state(current_state)
                last_update = now

                # CSV row
                dtr_d  = current_state["dtr"]
                comb_d = current_state["combined"]
                idx    = current_state["indices"]
                c      = current_state["counts"]
                log_f.write(
                    f"{current_state['timestamp']},{frame_no},"
                    f"{current_state['mhi']},{current_state['grade']},"
                    f"{current_state['alert']},"
                    f"{dtr_d.get('dtr','')},{dtr_d.get('dtr_pct','')},"
                    f"{dtr_d.get('dtr_grade','')},"
                    f"{dtr_d.get('time_score','')},"
                    f"{comb_d.get('combined_score','')},"
                    f"{comb_d.get('combined_grade','')},"
                    f"{c.get('Butterflyfish',0)},"
                    f"{c.get('Parrotfish',0)},"
                    f"{c.get('Angelfish',0)},"
                    f"{idx.get('shannon_H',0):.6f},"
                    f"{idx.get('pielou_J',0):.6f}\n"
                )
                log_f.flush()

                dtr_str = (f"DTR={dtr_d['dtr_pct']:.1f}%  "
                           f"Comb={comb_d['combined_score']:.1f}"
                           if dtr_d.get("complete")
                           else "DTR=pending")
                print(f"\r  F{frame_no:>6}  MHI={current_state['mhi']:5.1f}  "
                      f"{dtr_str}  FPS={fps_actual:.0f}",
                      end="", flush=True)

            # Draw
            if not args.no_display or writer:
                ann = annotate_boxes(frame.copy(), results)
                ann = draw_hud(ann, current_state, fps_actual)
                if writer:
                    writer.write(ann)
                if not args.no_display:
                    cv2.imshow(
                        "Phase 6 Video + DTR  |  Q=quit  P=pause  R=reset",
                        ann
                    )

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
        elif key == ord('p'):
            paused = not paused
            print(f"\n  {'PAUSED' if paused else 'RESUMED'}")
        elif key == ord('r'):
            cumulative.clear()
            species_ever_seen.clear()
            first_complete_frame = 0
            print("\n  Session reset — all counts cleared")

    # ── Finalise ────────────────────────────────────────────────────────────
    if current_state:
        current_state["status"] = "finished"
        write_state(current_state)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    log_f.close()

    print(f"\n\n{'='*64}")
    print(f"  SESSION COMPLETE")
    if current_state:
        dtr_d  = current_state.get("dtr", {})
        comb_d = current_state.get("combined", {})
        c      = current_state.get("counts", {})
        print(f"  Total detections  : {current_state.get('total_fish', 0)}")
        for sp in CLASS_NAMES.values():
            print(f"    {sp:<16}: {c.get(sp, 0)}")
        print(f"  MHI               : {current_state.get('mhi', '--')}  "
              f"[{current_state.get('grade', '')}]  {current_state.get('alert', '')}")
        if dtr_d.get("complete"):
            print(f"  DTR               : {dtr_d['dtr_pct']:.1f}%  "
                  f"Grade={dtr_d['dtr_grade']}  "
                  f"Time_Score={dtr_d['time_score']:.1f}")
            print(f"  Combined Score    : {comb_d.get('combined_score', '--')}  "
                  f"[{comb_d.get('combined_grade', '')}]")
            print(f"  Formula used      : {comb_d.get('formula', '')}")
        else:
            print(f"  DTR               : Not completed (not all species detected)")
        print(f"  Log CSV           : {LOG_CSV_FILE}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
