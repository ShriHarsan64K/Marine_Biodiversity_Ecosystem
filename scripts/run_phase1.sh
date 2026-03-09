#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — COMPLETE PIPELINE RUNNER
#  Marine Biodiversity Ecosystem Health Assessment
#  Author: Shri Harsan M | M.Tech Data Science | SRM Institute
#
#  Usage (from project root, with venv activated):
#      chmod +x scripts/run_phase1.sh
#      bash scripts/run_phase1.sh
#
#  Each step prints a header. If a step fails, the script stops.
#  Re-run individual steps manually using the python commands shown below.
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

PYTHON="python3.11"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

step() {
    echo ""
    echo "══════════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "══════════════════════════════════════════════════════════════════════"
}

# ─── PHASE 1A: Environment & Structure ────────────────────────────────────────
step "PHASE 1A — Scaffold project workspace"
$PYTHON scripts/phase1a_scaffold_project.py 2>&1 | tee "$LOG_DIR/phase1a_scaffold.log"

step "PHASE 1A — Verify installation (must pass 7/7)"
$PYTHON scripts/phase1a_verify_installation.py 2>&1 | tee "$LOG_DIR/phase1a_verify.log"

# ─── PHASE 1B: Dataset Collection ────────────────────────────────────────────
step "PHASE 1B — Build species taxonomy mapping"
$PYTHON src/utils/phase1b_map_species.py 2>&1 | tee "$LOG_DIR/phase1b_mapping.log"

step "PHASE 1B — Download Roboflow datasets"
$PYTHON src/utils/phase1b_download_roboflow.py 2>&1 | tee "$LOG_DIR/phase1b_roboflow.log"

step "PHASE 1B — Download Kaggle datasets"
$PYTHON src/utils/phase1b_download_kaggle.py 2>&1 | tee "$LOG_DIR/phase1b_kaggle.log"

# ─── PHASE 1C: Data Cleaning ──────────────────────────────────────────────────
step "PHASE 1C — Remove duplicate images (dry run preview)"
$PYTHON src/utils/phase1c_remove_duplicates.py \
    --input_dir dataset/raw --dry_run \
    2>&1 | tee "$LOG_DIR/phase1c_dedup_dry.log"

echo ""
echo "  ► Review results/cleaning/duplicate_report.txt"
echo "  ► Then run (REMOVES files):  python src/utils/phase1c_remove_duplicates.py --input_dir dataset/raw"
read -rp "  ► Press ENTER to continue to quality filtering (or Ctrl+C to pause here)… "

step "PHASE 1C — Filter low-quality images (dry run preview)"
$PYTHON src/utils/phase1c_filter_quality.py \
    --input_dir dataset/raw --dry_run \
    2>&1 | tee "$LOG_DIR/phase1c_quality_dry.log"

echo ""
echo "  ► Review results/cleaning/quality_report.txt"
read -rp "  ► Press ENTER to actually filter (moves rejects to dataset/quarantine)… "

$PYTHON src/utils/phase1c_filter_quality.py \
    --input_dir dataset/raw \
    2>&1 | tee "$LOG_DIR/phase1c_quality.log"

step "PHASE 1C — Validate annotations & standardise labels"
$PYTHON src/utils/phase1c_validate_annotations.py \
    --input_dir  dataset/raw \
    --output_dir dataset/standardized \
    2>&1 | tee "$LOG_DIR/phase1c_annotations.log"

step "PHASE 1C — Generate dataset statistics"
$PYTHON src/utils/phase1c_generate_statistics.py \
    --dataset_dir dataset/standardized \
    2>&1 | tee "$LOG_DIR/phase1c_stats.log"

# ─── PHASE 1D: Augmentation & Split ──────────────────────────────────────────
step "PHASE 1D — Analyse class balance"
$PYTHON src/utils/phase1d_analyze_balance.py \
    --dataset_dir dataset/standardized \
    --target 800 \
    2>&1 | tee "$LOG_DIR/phase1d_balance.log"

step "PHASE 1D — Apply offline augmentation"
$PYTHON src/utils/phase1d_augment_dataset.py \
    --input_dir  dataset/standardized \
    --output_dir dataset/augmented \
    --target 800 \
    2>&1 | tee "$LOG_DIR/phase1d_augment.log"

step "PHASE 1D — Stratified train/val/test split"
$PYTHON src/utils/phase1d_split_dataset.py \
    --input_dir  dataset/augmented \
    --output_dir dataset/processed \
    2>&1 | tee "$LOG_DIR/phase1d_split.log"

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  🎉 PHASE 1 PIPELINE COMPLETE"
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Deliverables:"
echo "    ✅  dataset/processed/         — Final balanced dataset"
echo "    ✅  dataset/processed/data.yaml — YOLOv8 training config"
echo "    ✅  configs/species_mapping.yaml"
echo "    ✅  configs/augmentation_plan.json"
echo "    ✅  results/cleaning/           — Statistics, plots, reports"
echo "    ✅  logs/                       — Full execution logs"
echo ""
echo "  Next Phase:"
echo "    Phase 2 — Image Enhancement Pipeline"
echo "    src/enhancement/phase2_ancuti_fusion.py"
echo ""
