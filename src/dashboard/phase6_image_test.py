"""
PHASE 6 — Image Folder Live Test Runner  (v2 — Shuffled / Stratified)
=======================================================================
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

WHY SHUFFLE?
  Your test set is sorted by class (all Butterflyfish first, then Parrotfish,
  then Angelfish). A rolling window on sorted images sees only 1 species at a
  time → artificially low diversity → misleading MHI.

  A real reef video has all species mixed together in every clip.
  Shuffling simulates that — it is the scientifically correct approach.

MODES (--mode):
  stratified  [DEFAULT]
              Interleaves classes evenly: BF, PF, AF, BF, PF, AF …
              Best for thesis demo — guaranteed all species visible
              throughout the entire run, MHI is stable and meaningful.

  shuffle     Random shuffle of all images.
              Most realistic simulation of real reef video.

  sorted      Original sorted order (grouped by class).
              Only use this to demonstrate WHY shuffle is needed.

USAGE:
  python src/dashboard/phase6_image_test.py
  python src/dashboard/phase6_image_test.py --mode shuffle
  python src/dashboard/phase6_image_test.py --mode sorted   # shows the bug
  python src/dashboard/phase6_image_test.py --delay 0.05 --loop

KEYBOARD:
  q  quit    p  pause/resume    r  reset window    s  save screenshot

DASHBOARD:
  1. python src/dashboard/phase6_image_test.py
  2. cd results/dashboard && python -m http.server 8080
  3. http://localhost:8080/phase6_dashboard_live.html
"""

import argparse
import json
import random
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
from src.biodiversity.phase5d_mhi import compute_mhi

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH      = ROOT / "results" / "detection" / "baseline_yolov8s" / "weights" / "best.pt"
DEFAULT_IMG_DIR = ROOT / "dataset" / "processed" / "test" / "images"
OUTPUT_DIR      = ROOT / "results" / "dashboard"
LIVE_STATE_FILE = OUTPUT_DIR / "live_state.json"
LOG_CSV_FILE    = OUTPUT_DIR / "image_test_log.csv"

IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "Butterflyfish", 1: "Parrotfish", 2: "Angelfish"}
COLORS = {
    "Butterflyfish": (0,  212, 255),
    "Parrotfish":    (0,  255, 157),
    "Angelfish":     (160, 100, 255),
}


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Phase 6 Image Test Runner v2")
    p.add_argument("--folder",     default=str(DEFAULT_IMG_DIR))
    p.add_argument("--model",      default=str(MODEL_PATH))
    p.add_argument("--conf",       type=float, default=0.35)
    p.add_argument("--window",     type=int,   default=0,
                   help="Unused in image mode (cumulative counting). Kept for CLI compatibility.")
    p.add_argument("--interval",   type=float, default=1.0,
                   help="Dashboard update interval in seconds")
    p.add_argument("--delay",      type=float, default=0.08,
                   help="Delay per image in seconds (controls speed)")
    p.add_argument("--mode",       default="stratified",
                   choices=["stratified", "shuffle", "sorted"],
                   help="Image ordering mode (default: stratified)")
    p.add_argument("--site",       default="SRM Marine Survey - Test Set")
    p.add_argument("--loop",       action="store_true",
                   help="Loop through images repeatedly")
    p.add_argument("--no-display", action="store_true",
                   help="Headless mode — no cv2 window")
    p.add_argument("--imgsz",      type=int, default=640)
    return p.parse_args()


# ── Image ordering ────────────────────────────────────────────────────────────
def order_images(images: list, mode: str) -> list:
    """
    stratified : round-robin interleave by class — BF, PF, AF, BF, PF, AF ...
    shuffle    : random order
    sorted     : original alphabetical (grouped by class — shows rolling-window bug)
    """
    if mode == "sorted":
        return sorted(images)

    if mode == "shuffle":
        imgs = list(images)
        random.shuffle(imgs)
        return imgs

    # ── stratified ────────────────────────────────────────────────────────
    buckets = {"Butterflyfish": [], "Parrotfish": [], "Angelfish": [], "Other": []}
    for img in images:
        stem   = img.stem.lower()
        parent = img.parent.name.lower()
        if   "butterfly" in stem or "bf" in stem or "butterfly" in parent:
            buckets["Butterflyfish"].append(img)
        elif "parrot"    in stem or "pf" in stem or "parrot"    in parent:
            buckets["Parrotfish"].append(img)
        elif "angel"     in stem or "af" in stem or "angel"     in parent:
            buckets["Angelfish"].append(img)
        else:
            buckets["Other"].append(img)

    for b in buckets.values():
        random.shuffle(b)

    # Round-robin
    keys = ["Butterflyfish", "Parrotfish", "Angelfish", "Other"]
    iters = {k: iter(v) for k, v in buckets.items()}
    done  = set()
    result = []
    while len(done) < len(keys):
        for k in keys:
            if k in done:
                continue
            try:
                result.append(next(iters[k]))
            except StopIteration:
                done.add(k)
    return result


# ── State print ───────────────────────────────────────────────────────────────
def print_bucket_info(images: list, mode: str):
    buckets = {"Butterflyfish": 0, "Parrotfish": 0, "Angelfish": 0, "Other": 0}
    for img in images:
        stem = img.stem.lower()
        if   "butterfly" in stem or "bf" in stem: buckets["Butterflyfish"] += 1
        elif "parrot"    in stem or "pf" in stem: buckets["Parrotfish"]    += 1
        elif "angel"     in stem or "af" in stem: buckets["Angelfish"]     += 1
        else:                                     buckets["Other"]         += 1
    print(f"\n  Image ordering ({mode}):")
    for k, v in buckets.items():
        if v:
            print(f"    {k:<16s}: {v}")
    print(f"    {'Total':<16s}: {len(images)}")


# ── Atomic JSON write ─────────────────────────────────────────────────────────
def write_state(state: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LIVE_STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(LIVE_STATE_FILE)


# ── Build MHI state dict ──────────────────────────────────────────────────────
def build_state(counts: dict, site: str, frame_no: int, fps: float,
                session_start: float, total_images: int,
                status: str = "running", window_size: int = 0,
                mode: str = "stratified", pass_no: int = 1) -> dict:

    if sum(counts.values()) == 0:
        counts = {"Butterflyfish": 0, "Parrotfish": 0, "Angelfish": 0}

    mhi_result = compute_mhi(counts, site_name=site)
    elapsed    = time.time() - session_start
    mm, ss     = divmod(int(elapsed), 60)
    progress   = round(frame_no / total_images * 100, 1) if total_images > 0 else 0

    return {
        "site":             site,
        "timestamp":        datetime.now().isoformat(),
        "session_elapsed":  f"{mm:02d}:{ss:02d}",
        "frame":            frame_no,
        "total_images":     total_images,
        "progress_pct":     progress,
        "fps_actual":       round(fps, 1),
        "status":           status,
        "mode":             mode,
        "pass":             pass_no,
        "total_detections": window_size,  # cumulative total detections
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


# ── Annotate frame ────────────────────────────────────────────────────────────
def annotate(frame, result, state, fps, frame_no, total, mode):
    overlay = frame.copy()

    if result and result[0].boxes is not None:
        for i in range(len(result[0].boxes)):
            x1,y1,x2,y2 = map(int, result[0].boxes.xyxy[i].tolist())
            cls   = int(result[0].boxes.cls[i].item())
            conf  = float(result[0].boxes.conf[i].item())
            name  = CLASS_NAMES.get(cls, f"cls{cls}")
            color = COLORS.get(name, (200,200,200))
            cv2.rectangle(overlay,(x1,y1),(x2,y2),color,2)
            cv2.rectangle(overlay,(x1-2,y1-2),(x2+2,y2+2),color,1)
            lbl = f"{name}  {conf:.2f}"
            (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
            cv2.rectangle(overlay,(x1,y1-th-7),(x1+tw+5,y1),color,-1)
            cv2.putText(overlay,lbl,(x1+3,y1-3),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,0,0),1,cv2.LINE_AA)

    frame = cv2.addWeighted(overlay,0.85,frame,0.15,0)

    # HUD panel
    hw, hh = 316, 224
    hud = np.zeros((hh,hw,3),np.uint8)
    hud[:] = (4,15,25)

    def put(t,x,y,s=0.42,c=(160,210,230),th=1):
        cv2.putText(hud,t,(x,y),cv2.FONT_HERSHEY_SIMPLEX,s,c,th,cv2.LINE_AA)

    mhi   = state.get("mhi",0)
    alert = state.get("alert","")
    ac    = (0,255,157) if alert=="HEALTHY" else (255,194,0) if alert=="WARNING" else (255,61,90)

    put("REEF HEALTH MONITOR",8,17,0.48,(0,220,190),1)
    cv2.line(hud,(8,23),(hw-8,23),(0,100,110),1)
    cv2.putText(hud,f"{mhi:.1f}",(8,60),cv2.FONT_HERSHEY_SIMPLEX,1.3,ac,2,cv2.LINE_AA)
    put("/100",96,60,0.44,(80,140,160))
    put(f"{state.get('grade','')}  |  {alert}",8,74,0.4,ac)

    mc = (0,200,180) if mode=="stratified" else (140,200,100) if mode=="shuffle" else (180,80,80)
    put(f"mode: {mode}",8,88,0.36,mc)
    cv2.line(hud,(8,95),(hw-8,95),(0,60,70),1)

    put("CUMULATIVE COUNTS:",8,109,0.37,(0,190,190))
    y = 123
    total_fish = max(state.get("total_fish",1), 1)
    for sp, cnt in state.get("counts",{}).items():
        col = COLORS.get(sp,(180,180,180))
        bar = int(cnt / total_fish * 130)
        cv2.rectangle(hud,(8,y-9),(8+bar,y-1),
                      tuple(c//5 for c in col),-1)
        cv2.rectangle(hud,(8,y-9),(8+bar,y-1),col,1)
        put(f" {sp:<15s} {cnt:>4}",8,y,0.37,col)
        y += 14

    cv2.line(hud,(8,y+1),(hw-8,y+1),(0,60,70),1)
    y += 13
    idx = state.get("indices",{})
    put(f"H'={idx.get('shannon_H',0):.4f}  D={idx.get('simpsons_D',0):.4f}",8,y,0.36,(100,170,190))
    y += 12
    put(f"J'={idx.get('pielou_J',0):.4f}  S={idx.get('species_richness',0)}",8,y,0.36,(100,170,190))
    y += 12
    prog = state.get("progress_pct",0)
    put(f"[{frame_no}/{total}] {prog:.0f}%  FPS:{fps:.0f}  {state.get('session_elapsed','--')}",
        8,y,0.35,(80,120,140))

    # Progress bar
    by = hh-6
    bmax = hw-16
    bfill = int(bmax*prog/100)
    cv2.rectangle(hud,(8,by-5),(hw-8,by),(20,50,60),-1)
    if bfill>0:
        cv2.rectangle(hud,(8,by-5),(8+bfill,by),(0,200,180),-1)

    roi  = frame[10:10+hh,10:10+hw]
    mask = np.full((hh,hw),0.82,np.float32)
    frame[10:10+hh,10:10+hw] = (
        hud*mask[:,:,np.newaxis]+roi*(1-mask[:,:,np.newaxis])
    ).astype(np.uint8)
    cv2.rectangle(frame,(10,10),(10+hw,10+hh),(0,170,160),1)
    return frame


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.folder)
    if not img_dir.exists():
        print(f"  ERROR: folder not found: {img_dir}")
        sys.exit(1)

    raw = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not raw:
        print(f"  ERROR: no images in {img_dir}")
        sys.exit(1)

    print(f"\n{'='*64}")
    print(f"  PHASE 6 — IMAGE TEST RUNNER  v2")
    print(f"{'='*64}")
    print(f"  Folder   : {img_dir}")
    print(f"  Images   : {len(raw)}")
    print(f"  Mode     : {args.mode}")
    print(f"  Window   : {args.window} frames")
    print(f"  Delay    : {args.delay}s/image")
    print(f"  Loop     : {args.loop}")

    if args.mode == "stratified":
        print(f"\n  Stratified: interleaves BF/PF/AF evenly → all species always")
        print(f"  Cumulative counts used — all species always remembered. Best for demo.")
    elif args.mode == "shuffle":
        print(f"\n  Shuffle: random order → realistic reef video simulation.")
    else:
        print(f"\n  WARNING: sorted mode shows the class-grouping bug intentionally.")

    images = order_images(raw, args.mode)
    print_bucket_info(images, args.mode)

    print(f"\n  DASHBOARD SETUP:")
    print(f"    Terminal 2: cd results\\dashboard && python -m http.server 8080")
    print(f"    Browser:    http://localhost:8080/phase6_dashboard_live.html")
    print(f"\n  Q=quit  P=pause  R=reset window  S=screenshot\n")

    print("  Loading YOLOv8s...")
    model = YOLO(args.model)
    print("  Model loaded ✅\n")

    # Cumulative counter — never resets, only grows
    cumulative    = Counter()   # total detections across all frames so far
    frame_no      = 0
    pass_no       = 0
    paused        = False
    session_start = time.time()
    last_update   = time.time()
    fps_actual    = 0.0
    fps_cnt       = 0
    last_fps_t    = time.time()
    current_state: dict = {}
    shot_n        = 0
    running       = True

    # Initial state
    init = build_state({"Butterflyfish":0,"Parrotfish":0,"Angelfish":0},
                       args.site, 0, 0, session_start, len(images),
                       "initialising", 0, args.mode)
    write_state(init)
    current_state = init

    log = open(LOG_CSV_FILE,"w",encoding="utf-8")
    log.write("timestamp,pass,frame,image,mode,mhi,grade,alert,bf,pf,af,shannon,simpson,pielou\n")

    while running:
        pass_no += 1
        if pass_no > 1:
            if not args.loop:
                break
            images = order_images(raw, args.mode)  # re-order each loop
            cumulative.clear()                      # fresh count each pass

        for img_path in images:
            if not running:
                break

            while paused:
                key = cv2.waitKey(100) & 0xFF
                if   key == ord('q'): running=False; break
                elif key == ord('p'): paused=False;  print("\n  RESUMED")
            if not running:
                break

            # Detect
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            dh = 540
            dw = int(frame.shape[1]*dh/frame.shape[0])
            fd = cv2.resize(frame,(dw,dh))

            res = model.predict(fd, conf=args.conf, iou=0.5,
                                imgsz=args.imgsz, verbose=False)

            frame_no += 1
            fps_cnt  += 1

            # Accumulate into session-level cumulative counter
            if res and res[0].boxes is not None:
                for cid in res[0].boxes.cls.tolist():
                    nm = CLASS_NAMES.get(int(cid))
                    if nm:
                        cumulative[nm] += 1

            # Update dashboard
            now = time.time()
            if now - last_update >= args.interval:
                # Use cumulative counts — all detections since session start
                cd = {
                    "Butterflyfish": cumulative.get("Butterflyfish",0),
                    "Parrotfish":    cumulative.get("Parrotfish",   0),
                    "Angelfish":     cumulative.get("Angelfish",    0),
                }
                dt = now - last_fps_t
                fps_actual = fps_cnt/dt if dt>0 else 0
                fps_cnt    = 0
                last_fps_t = now

                current_state = build_state(
                    cd, args.site, frame_no, fps_actual,
                    session_start, len(images),
                    "running", sum(cumulative.values()), args.mode, pass_no
                )
                write_state(current_state)
                last_update = now

                idx = current_state["indices"]
                c   = current_state["counts"]
                log.write(f"{current_state['timestamp']},{pass_no},{frame_no},"
                          f"{img_path.name},{args.mode},"
                          f"{current_state['mhi']},{current_state['grade']},"
                          f"{current_state['alert']},"
                          f"{c.get('Butterflyfish',0)},{c.get('Parrotfish',0)},{c.get('Angelfish',0)},"
                          f"{idx.get('shannon_H',0):.6f},{idx.get('simpsons_D',0):.6f},{idx.get('pielou_J',0):.6f}\n")
                log.flush()

                print(f"\r  Pass{pass_no} [{frame_no:>4}/{len(images)}] "
                      f"MHI={current_state['mhi']:5.1f} "
                      f"{current_state['grade']:<10s}"
                      f"BF={c.get('Butterflyfish',0):>3} "
                      f"PF={c.get('Parrotfish',0):>3} "
                      f"AF={c.get('Angelfish',0):>3}  "
                      f"FPS={fps_actual:.0f}",
                      end="", flush=True)

            if not args.no_display:
                ann = annotate(fd.copy(), res, current_state,
                               fps_actual, frame_no, len(images), args.mode)
                cv2.imshow("Phase 6  [Q=quit  P=pause  R=reset  S=screenshot]", ann)

            key = cv2.waitKey(max(1,int(args.delay*1000))) & 0xFF
            if   key == ord('q'): running=False; break
            elif key == ord('p'): paused=True;   print("\n  PAUSED")
            elif key == ord('r'): cumulative.clear(); print("\n  Cumulative counts reset")
            elif key == ord('s'):
                shot_n += 1
                sp = OUTPUT_DIR/f"screenshot_{shot_n:03d}.png"
                cv2.imwrite(str(sp), ann)
                print(f"\n  Saved: {sp}")

            time.sleep(max(0, args.delay-0.01))

        if args.loop:
            print(f"\n  Pass {pass_no} done — re-shuffling...")

    # Final state
    if current_state:
        current_state["status"] = "finished"
        write_state(current_state)

    cv2.destroyAllWindows()
    log.close()

    print(f"\n\n{'='*64}")
    print(f"  Session complete  |  Mode: {args.mode}")
    print(f"  Frames processed : {frame_no}")
    if current_state:
        c = current_state.get("counts",{})
        print(f"  Final MHI  : {current_state.get('mhi','--')}")
        print(f"  Final Grade: {current_state.get('grade','--')}")
        print(f"  Counts     : BF={c.get('Butterflyfish',0)}  "
              f"PF={c.get('Parrotfish',0)}  AF={c.get('Angelfish',0)}")
    print(f"  State file : {LIVE_STATE_FILE}")
    print(f"  Log CSV    : {LOG_CSV_FILE}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()