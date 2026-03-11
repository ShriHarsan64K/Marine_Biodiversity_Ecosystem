"""
PHASE 5 — Master Runner
========================
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Reads tracking_log.csv from Phase 4, computes all biodiversity indices,
and produces a full MHI report + visualisation.

Usage:
  python src/biodiversity/phase5_run.py
  python src/biodiversity/phase5_run.py --csv results/tracking/tracking_log.csv
  python src/biodiversity/phase5_run.py --csv results/tracking/tracking_log.csv ^
    --site "Reef Site A"
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

# ── Allow running from project root without install ───────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.biodiversity.phase5a_indices import compute_all_indices, print_indices
from src.biodiversity.phase5b_weighted import weighted_shannon, print_weighted
from src.biodiversity.phase5c_trophic  import trophic_analysis, print_trophic
from src.biodiversity.phase5d_mhi      import compute_mhi, print_mhi, save_report

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = Path(r"results\biodiversity")


# ── CSV loader ────────────────────────────────────────────────────────────────

def load_counts_from_csv(csv_path: str) -> dict[str, int]:
    """Read tracking_log.csv and return {class_name: total_detections}."""
    counts: Counter = Counter()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("class_name", "").strip()
            if name:
                counts[name] += 1
    return dict(counts)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_report(mhi_result: dict, output_dir: Path) -> None:
    """Generate a 2×2 summary figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = mhi_result["counts"]
    idx    = mhi_result["indices"]
    comp   = mhi_result["components"]
    trop   = mhi_result["trophic"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Marine Health Index — {mhi_result['site']}\n"
        f"MHI = {mhi_result['mhi']:.1f}/100  {mhi_result['emoji']} {mhi_result['grade']}",
        fontsize=14, fontweight="bold"
    )

    # ── Panel 1: Species counts bar chart ────────────────────────────────────
    ax = axes[0, 0]
    species = list(counts.keys())
    values  = list(counts.values())
    colors  = ["#00C8FF", "#00A5FF", "#FF8000"][:len(species)]
    bars = ax.bar(species, values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Species Detections (Phase 4)", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(values) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel 2: Biodiversity indices radar / bar ────────────────────────────
    ax = axes[0, 1]
    index_names  = ["H' Shannon", "H'_w Weighted", "D Simpson", "J' Evenness"]
    index_values = [
        idx["shannon_H"],
        idx["weighted_H"],
        idx["simpsons_D"],
        idx["pielou_J"],
    ]
    bar_colors = ["#2ECC71", "#27AE60", "#3498DB", "#9B59B6"]
    bars2 = ax.barh(index_names, index_values, color=bar_colors, edgecolor="white")
    for bar, v in zip(bars2, index_values):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, max(index_values) * 1.3)
    ax.set_title("Biodiversity Indices", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel 3: MHI component breakdown ────────────────────────────────────
    ax = axes[1, 0]
    comp_names = [
        "Wtd Shannon\n(×0.30)",
        "Trophic\nBalance\n(×0.25)",
        "Apex\nPredator\n(×0.20)",
        "Indicator\nPresence\n(×0.15)",
        "Evenness\n(×0.10)",
    ]
    comp_scores = [
        comp["weighted_shannon_score"],
        comp["trophic_balance_score"],
        comp["apex_predator_score"],
        comp["indicator_presence_score"],
        comp["evenness_score"],
    ]
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    contribs = [s * w for s, w in zip(comp_scores, weights)]
    cmap  = plt.cm.RdYlGn
    norms = [s / 100 for s in comp_scores]
    colors3 = [cmap(n) for n in norms]
    bars3 = ax.bar(comp_names, comp_scores, color=colors3, edgecolor="white")
    ax.axhline(y=mhi_result["mhi"], color="navy", linestyle="--",
               linewidth=1.5, label=f"MHI={mhi_result['mhi']:.1f}")
    for bar, v, c in zip(bars3, comp_scores, contribs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_title("MHI Component Scores (0–100)", fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel 4: Trophic pie ─────────────────────────────────────────────────
    ax = axes[1, 1]
    grp_counts = trop["group_counts"]
    if grp_counts:
        grp_labels = list(grp_counts.keys())
        grp_sizes  = list(grp_counts.values())
        grp_colors = plt.cm.Set3(np.linspace(0, 1, len(grp_labels)))
        wedges, texts, autotexts = ax.pie(
            grp_sizes, labels=grp_labels, autopct="%1.1f%%",
            colors=grp_colors, startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
        )
        for at in autotexts:
            at.set_fontsize(9)
    ax.set_title("Trophic Group Distribution", fontweight="bold")

    # ── MHI gauge annotation ─────────────────────────────────────────────────
    grade_color = {"Excellent": "#27AE60", "Good": "#2ECC71",
                   "Fair": "#F39C12", "Poor": "#E67E22", "Critical": "#E74C3C"}
    gc = grade_color.get(mhi_result["grade"], "#7F8C8D")
    fig.text(0.5, 0.01,
             f"MHI = {mhi_result['mhi']:.2f}/100  |  {mhi_result['emoji']} {mhi_result['grade']}  |  Alert: {mhi_result['alert']}",
             ha="center", fontsize=12, fontweight="bold", color=gc,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=gc + "22", edgecolor=gc))

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = output_dir / "mhi_report.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved : {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 — MHI Runner")
    p.add_argument("--csv",  default=r"results\tracking\tracking_log.csv",
                   help="Path to tracking_log.csv from Phase 4")
    p.add_argument("--site", default="Marine Survey Site",
                   help="Site name for the report")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)

    print(f"\n{'='*62}")
    print(f"  PHASE 5 — BIODIVERSITY & HEALTH ASSESSMENT")
    print(f"{'='*62}")
    print(f"  Reading CSV : {args.csv}")

    counts = load_counts_from_csv(args.csv)
    print(f"  Detections  : {counts}")
    print(f"  Total fish  : {sum(counts.values())}\n")

    # ── 5A: Core indices ──────────────────────────────────────────────────────
    print("  ── PHASE 5A: Core Biodiversity Indices ──")
    indices = compute_all_indices(counts)
    print_indices(indices)

    # ── 5B: Weighted Shannon ─────────────────────────────────────────────────
    print("  ── PHASE 5B: Weighted Shannon ──")
    ws = weighted_shannon(counts)
    print_weighted(ws)

    # ── 5C: Trophic Analysis ─────────────────────────────────────────────────
    print("  ── PHASE 5C: Trophic Analysis ──")
    tr = trophic_analysis(counts)
    print_trophic(tr)

    # ── 5D: Composite MHI ────────────────────────────────────────────────────
    print("  ── PHASE 5D: Marine Health Index ──")
    mhi_result = compute_mhi(counts, site_name=args.site)
    print_mhi(mhi_result)

    # ── Save outputs ─────────────────────────────────────────────────────────
    save_report(mhi_result, output_dir)
    plot_report(mhi_result, output_dir)

    print(f"\n  All outputs saved to: {output_dir}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
