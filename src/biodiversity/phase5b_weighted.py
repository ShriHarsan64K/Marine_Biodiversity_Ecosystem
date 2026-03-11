"""
PHASE 5B — Weighted Shannon Health Index
=========================================
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Applies ecological importance weights to Shannon index.
H'_w = -Σ(wi × pi × ln(pi)) / Σ(wi_active)

Weights from roadmap (Wilson et al. 2010, Mumby et al. 2006):
  Butterflyfish : 2.0  (coral dependency — early warning indicator)
  Parrotfish    : 1.6  (algae control — bioerosion balance)
  Angelfish     : 1.1  (reef structure indicator)
"""

import math
from src.biodiversity.phase5a_indices import compute_all_indices, print_indices


# ── Ecological weights (roadmap Table 5B) ─────────────────────────────────────
SPECIES_WEIGHTS: dict[str, float] = {
    "Butterflyfish":  2.0,   # obligate corallivore — coral health proxy
    "Grouper":        1.8,   # apex predator — fishing pressure
    "Parrotfish":     1.6,   # herbivore — algae control critical
    "Sharks":         2.0,   # top predator — pristine reef indicator
    "Surgeonfish":    1.4,   # herbivore guild
    "Cleaner Wrasse": 1.5,   # mutualism — ecosystem services
    "Damselfish":     1.0,   # baseline habitat indicator
    "Wrasse":         1.2,   # functional diversity
    "Triggerfish":    1.3,   # degradation sensitivity
    "Angelfish":      1.1,   # reef structure
}
DEFAULT_WEIGHT = 1.0         # for any unlisted species


def get_weight(species: str) -> float:
    return SPECIES_WEIGHTS.get(species, DEFAULT_WEIGHT)


# ── Weighted Shannon ──────────────────────────────────────────────────────────

def weighted_shannon(counts: dict[str, int]) -> dict:
    """
    Compute weighted Shannon index H'_w.

    H'_w = -Σ(wi × pi × ln(pi)) / Σ(wi_active)

    where wi_active = weights of species that are actually present (n > 0).

    Returns dict with:
      weighted_H, unweighted_H, weight_map,
      weighted_grade, ecological_signal
    """
    from src.biodiversity.phase5a_indices import shannon_index, shannon_grade

    total = sum(counts.values())
    if total == 0:
        return {"weighted_H": 0.0, "unweighted_H": 0.0,
                "weight_map": {}, "weighted_grade": "Poor",
                "ecological_signal": "No fish detected"}

    # Only include species actually detected
    active = {sp: n for sp, n in counts.items() if n > 0}

    numerator   = 0.0
    w_sum       = 0.0
    weight_map  = {}

    for sp, n in active.items():
        pi  = n / total
        wi  = get_weight(sp)
        contribution = -wi * pi * math.log(pi)
        numerator   += contribution
        w_sum       += wi
        weight_map[sp] = {
            "count":        n,
            "proportion":   round(pi, 4),
            "weight":       wi,
            "contribution": round(contribution, 6),
        }

    weighted_H   = round(numerator / w_sum if w_sum > 0 else 0.0, 6)
    unweighted_H = shannon_index(counts)
    grade, emoji = shannon_grade(weighted_H, len(active))

    # Ecological signal — what the weighted score means for reef health
    signal = _ecological_signal(active, weighted_H)

    return {
        "weighted_H":       weighted_H,
        "unweighted_H":     unweighted_H,
        "weight_map":       weight_map,
        "weighted_grade":   grade,
        "weighted_emoji":   emoji,
        "ecological_signal": signal,
        "w_sum":            round(w_sum, 4),
    }


def _ecological_signal(active: dict[str, int], weighted_H: float) -> str:
    """Generate an ecological interpretation string."""
    signals = []

    if "Butterflyfish" in active:
        signals.append("Coral habitat present (Butterflyfish detected)")
    else:
        signals.append("⚠ No Butterflyfish — possible coral degradation")

    if "Parrotfish" in active:
        signals.append("Algae control functioning (Parrotfish detected)")
    else:
        signals.append("⚠ No Parrotfish — algae bloom risk")

    if "Angelfish" in active:
        signals.append("Reef structure intact (Angelfish detected)")

    # Grade against H'_w_max (equal-proportions maximum for this species set)
    _s = len(active)
    _pi_eq = 1.0 / _s if _s > 0 else 1.0
    _w_sum = sum(SPECIES_WEIGHTS.get(sp, 1.0) for sp in active)
    _hw_max = (
        sum(SPECIES_WEIGHTS.get(sp, 1.0) * (-_pi_eq * math.log(_pi_eq)) for sp in active)
        / _w_sum
    ) if _w_sum > 0 else 1.0
    _ratio = weighted_H / _hw_max if _hw_max > 0 else 0.0

    if _ratio >= 0.85:
        signals.append("✅ Excellent weighted diversity — pristine reef condition")
    elif _ratio >= 0.65:
        signals.append("✅ Good weighted diversity — healthy reef")
    elif _ratio >= 0.45:
        signals.append("🟡 Fair diversity — reef under moderate stress")
    else:
        signals.append("🔴 Low diversity — reef degradation alert")

    return " | ".join(signals)


def print_weighted(result: dict) -> None:
    print(f"\n  {'─'*54}")
    print(f"  WEIGHTED SHANNON HEALTH INDEX")
    print(f"  {'─'*54}")
    print(f"  Unweighted H'   = {result['unweighted_H']:.4f}")
    print(f"  Weighted   H'_w = {result['weighted_H']:.4f}"
          f"  →  {result['weighted_emoji']} {result['weighted_grade']}")
    print(f"\n  Per-species contributions:")
    print(f"  {'Species':<18s} {'Count':>6}  {'Pi':>6}  {'Wi':>5}  {'Contribution':>12}")
    print(f"  {'─'*54}")
    for sp, info in result['weight_map'].items():
        print(f"  {sp:<18s} {info['count']:>6}  {info['proportion']:>6.3f}"
              f"  {info['weight']:>5.1f}  {info['contribution']:>12.6f}")
    print(f"\n  Ecological Signal:")
    for part in result['ecological_signal'].split(" | "):
        print(f"    • {part}")
    print(f"  {'─'*54}\n")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_counts = {
        "Butterflyfish": 56,
        "Parrotfish":    163,
        "Angelfish":     271,
    }
    print("\n  Input counts:", test_counts)
    result = weighted_shannon(test_counts)
    print_weighted(result)