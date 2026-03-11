"""
PHASE 5D — Composite Marine Health Index (MHI)
===============================================
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

MHI = 0.30 × H'_weighted_normalised
    + 0.25 × Trophic_Balance
    + 0.20 × Apex_Predator_Score
    + 0.15 × Indicator_Presence
    + 0.10 × Evenness

Output: 0–100 scale
  80–100 : Excellent  (pristine reef)      🟢
  60–79  : Good       (healthy reef)       🟢
  40–59  : Fair       (stressed reef)      🟡
  20–39  : Poor       (degraded reef)      🟠
   0–19  : Critical   (ecosystem collapse) 🔴

Alert thresholds:
  🔴 CRITICAL : MHI < 30
  🟡 WARNING  : MHI 30–50 or declining >10 pts/hr
  🟢 HEALTHY  : MHI > 60 and stable
"""

import json
import math
from pathlib import Path
from datetime import datetime

from src.biodiversity.phase5a_indices import compute_all_indices
from src.biodiversity.phase5b_weighted import weighted_shannon, SPECIES_WEIGHTS
from src.biodiversity.phase5c_trophic  import trophic_analysis


# ── MHI weights (from roadmap) ────────────────────────────────────────────────
MHI_WEIGHTS = {
    "weighted_shannon":  0.30,
    "trophic_balance":   0.25,
    "apex_predator":     0.20,
    "indicator_presence":0.15,
    "evenness":          0.10,
}

# Indicator species for our 3-class project
INDICATOR_SPECIES = ["Butterflyfish", "Parrotfish", "Angelfish"]

# Alert thresholds
ALERT_CRITICAL = 30.0
ALERT_WARNING  = 50.0
ALERT_HEALTHY  = 60.0


# ── Component calculators ─────────────────────────────────────────────────────

def _normalise_shannon(weighted_H: float,
                       counts: dict | None = None) -> float:
    """
    Scale H'_w to 0–100 using the CORRECT theoretical maximum for the
    actual number of species and their weights.

    max H'_w occurs when all species are equally abundant (pi = 1/S).
    max H'_w = -Σ(wi × 1/S × ln(1/S)) / Σ(wi)
               = ln(S) × Σ(wi)/S / Σ(wi)  = ln(S)   <-- only if all wi equal
    With unequal weights the full formula is needed.
    """
    import math
    if counts is None or len(counts) == 0:
        return 0.0
    active = {sp: n for sp, n in counts.items() if n > 0}
    S = len(active)
    if S <= 1:
        return 0.0 if weighted_H == 0 else 100.0
    pi_eq = 1.0 / S
    w_sum = sum(SPECIES_WEIGHTS.get(sp, 1.0) for sp in active)
    max_hw = sum(
        SPECIES_WEIGHTS.get(sp, 1.0) * (-pi_eq * math.log(pi_eq))
        for sp in active
    ) / w_sum if w_sum > 0 else math.log(S)
    if max_hw == 0:
        return 0.0
    return round(min(100.0, (weighted_H / max_hw) * 100.0), 4)


def _apex_predator_score(counts: dict[str, int]) -> float:
    """
    Score based on apex predator presence.
    Our 3-class project has no apex predators (Grouper/Sharks not in dataset).
    Score = 50 (neutral) when absent — not penalised for dataset scope,
    but would be 0 in a full ecosystem survey with no apex predators.
    """
    apex_species = {"Sharks", "Grouper"}
    total = sum(counts.values())
    if total == 0:
        return 0.0
    apex_count = sum(n for sp, n in counts.items() if sp in apex_species)
    if apex_count == 0:
        # No apex predators in our 3-class scope → neutral 50
        # Note in report: apex predator monitoring requires expanded dataset
        return 50.0
    pct = apex_count / total * 100
    if 5 <= pct <= 10:
        return 100.0
    elif pct < 5:
        return max(0.0, pct / 5.0 * 100)
    else:
        return max(0.0, 100.0 - (pct - 10) * 5)


def _indicator_presence_score(counts: dict[str, int],
                               indicators: list[str]) -> float:
    """
    Score = (number of indicator species present / total indicators) × 100.
    Weighted by ecological importance: present + abundant = bonus.
    """
    if not indicators:
        return 0.0
    total = sum(counts.values())
    if total == 0:
        return 0.0

    present = sum(1 for sp in indicators if counts.get(sp, 0) > 0)
    base_score = present / len(indicators) * 100

    # Abundance bonus: each indicator >10% of community adds up to 5 pts
    bonus = 0.0
    for sp in indicators:
        if counts.get(sp, 0) > 0:
            pct = counts[sp] / total * 100
            if pct >= 10:
                bonus += 5.0
    return min(100.0, base_score + bonus)


def _evenness_score(j: float) -> float:
    """Scale Pielou's J' (0–1) to 0–100."""
    return round(j * 100.0, 2)


# ── Main MHI computation ──────────────────────────────────────────────────────

def compute_mhi(counts: dict[str, int],
                site_name: str = "Unknown Site",
                timestamp: str | None = None) -> dict:
    """
    Compute the full Marine Health Index.

    Args:
        counts    : {species_name: count}  from tracking_log.csv
        site_name : descriptive name for the monitoring site
        timestamp : ISO timestamp string (defaults to now)

    Returns:
        Full MHI report dict
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    # ── Component 1: Weighted Shannon (normalised to 0-100) ──────────────────
    ws_result   = weighted_shannon(counts)
    c1_raw      = _normalise_shannon(ws_result["weighted_H"], counts)
    c1          = round(c1_raw, 2)

    # ── Component 2: Trophic Balance ─────────────────────────────────────────
    tr_result   = trophic_analysis(counts)
    c2          = round(tr_result["trophic_balance_score"], 2)

    # ── Component 3: Apex Predator Score ─────────────────────────────────────
    c3          = round(_apex_predator_score(counts), 2)

    # ── Component 4: Indicator Presence ──────────────────────────────────────
    c4          = round(_indicator_presence_score(counts, INDICATOR_SPECIES), 2)

    # ── Component 5: Evenness ────────────────────────────────────────────────
    idx_result  = compute_all_indices(counts)
    c5          = round(_evenness_score(idx_result["pielou_J"]), 2)

    # ── Composite MHI ────────────────────────────────────────────────────────
    mhi = (
        MHI_WEIGHTS["weighted_shannon"]   * c1 +
        MHI_WEIGHTS["trophic_balance"]    * c2 +
        MHI_WEIGHTS["apex_predator"]      * c3 +
        MHI_WEIGHTS["indicator_presence"] * c4 +
        MHI_WEIGHTS["evenness"]           * c5
    )
    mhi = round(mhi, 2)

    # ── Grade ────────────────────────────────────────────────────────────────
    grade, emoji, alert = _mhi_grade(mhi)

    return {
        "site":            site_name,
        "timestamp":       timestamp,
        "counts":          counts,
        "total_fish":      sum(counts.values()),
        "mhi":             mhi,
        "grade":           grade,
        "emoji":           emoji,
        "alert":           alert,
        "components": {
            "weighted_shannon_score":   c1,
            "trophic_balance_score":    c2,
            "apex_predator_score":      c3,
            "indicator_presence_score": c4,
            "evenness_score":           c5,
        },
        "weights":         MHI_WEIGHTS,
        "indices": {
            "shannon_H":        idx_result["shannon_H"],
            "weighted_H":       ws_result["weighted_H"],
            "simpsons_D":       idx_result["simpsons_D"],
            "pielou_J":         idx_result["pielou_J"],
            "species_richness": idx_result["species_richness"],
        },
        "trophic":         tr_result,
        "ecological_signal": ws_result["ecological_signal"],
        "degradation_signals": tr_result["degradation_signals"],
    }


def _mhi_grade(mhi: float) -> tuple[str, str, str]:
    """Return (grade, emoji, alert_level)."""
    if mhi >= 80:
        return "Excellent", "🟢", "HEALTHY"
    elif mhi >= 60:
        return "Good",      "🟢", "HEALTHY"
    elif mhi >= 40:
        return "Fair",      "🟡", "WARNING"
    elif mhi >= 20:
        return "Poor",      "🟠", "WARNING"
    else:
        return "Critical",  "🔴", "CRITICAL"


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_mhi(result: dict) -> None:
    w = MHI_WEIGHTS
    c = result["components"]
    idx = result["indices"]

    print(f"\n{'='*62}")
    print(f"  MARINE HEALTH INDEX (MHI) — PHASE 5D")
    print(f"{'='*62}")
    print(f"  Site      : {result['site']}")
    print(f"  Timestamp : {result['timestamp']}")
    print(f"  Fish seen : {result['total_fish']}")
    print()
    print(f"  ┌─────────────────────────────────────────────────┐")
    print(f"  │   MHI = {result['mhi']:6.2f} / 100   "
          f"{result['emoji']} {result['grade']:<10s}           │")
    print(f"  │   Alert  : {result['alert']:<40s}│")
    print(f"  └─────────────────────────────────────────────────┘")
    print()
    print(f"  Component breakdown:")
    print(f"  {'Component':<30s} {'Weight':>6}  {'Score':>6}  {'Contribution':>12}")
    print(f"  {'─'*58}")
    rows = [
        ("Weighted Shannon H'_w", w["weighted_shannon"],   c["weighted_shannon_score"]),
        ("Trophic Balance",       w["trophic_balance"],    c["trophic_balance_score"]),
        ("Apex Predator Presence",w["apex_predator"],      c["apex_predator_score"]),
        ("Indicator Presence",    w["indicator_presence"], c["indicator_presence_score"]),
        ("Evenness (Pielou J')",  w["evenness"],           c["evenness_score"]),
    ]
    for name, weight, score in rows:
        contrib = weight * score
        print(f"  {name:<30s} {weight:>6.2f}  {score:>6.1f}  {contrib:>12.2f}")
    print(f"  {'─'*58}")
    print(f"  {'TOTAL MHI':>50s} {result['mhi']:>8.2f}")
    print()
    print(f"  Biodiversity Indices:")
    print(f"    Shannon H'       = {idx['shannon_H']:.4f}")
    print(f"    Weighted H'_w    = {idx['weighted_H']:.4f}")
    print(f"    Simpson's D      = {idx['simpsons_D']:.4f}")
    print(f"    Pielou's J'      = {idx['pielou_J']:.4f}")
    print(f"    Species Richness = {idx['species_richness']}")
    print()
    print(f"  Ecological Signal:")
    for part in result["ecological_signal"].split(" | "):
        print(f"    • {part}")
    print()
    print(f"  Degradation Signals:")
    for sig in result["degradation_signals"]:
        print(f"    {sig}")
    print(f"{'='*62}\n")


def save_report(result: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Strip non-serialisable keys for JSON
    import copy
    clean = copy.deepcopy(result)
    # trophic sub-dict is already JSON-safe
    out = output_dir / "mhi_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)
    print(f"  MHI report saved: {out}")
    return out


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_counts = {
        "Butterflyfish": 56,
        "Parrotfish":    163,
        "Angelfish":     271,
    }
    result = compute_mhi(test_counts, site_name="Test Site (Phase 4 detections)")
    print_mhi(result)