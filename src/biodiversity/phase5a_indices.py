"""
PHASE 5A — Core Biodiversity Indices
=====================================
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Computes:
  1. Shannon Diversity Index  H'  = -Σ(pi × ln(pi))
  2. Simpson's Index          D   = 1 - Σ(pi²)
  3. Pielou's Evenness        J'  = H' / ln(S)
  4. Species Richness         S   = count of species with n > 0

All functions accept a dict  {species_name: count}  and return floats.
"""

import math
from typing import Optional


# ── Thresholds (from roadmap) ─────────────────────────────────────────────────
# Thresholds on H'/H'_max RATIO (0–1) — fair for any species richness S.
# With S=3: H'_max=ln(3)=1.099, so H'=0.94 → ratio=0.856 → Good ✅
SHANNON_THRESHOLDS = [
    (0.85, "Excellent", "🟢"),
    (0.65, "Good",      "🟢"),
    (0.45, "Fair",      "🟡"),
    (0.0,  "Poor",      "🔴"),
]


# ── Core index functions ──────────────────────────────────────────────────────

def species_richness(counts: dict[str, int]) -> int:
    """S — number of species with at least 1 individual detected."""
    return sum(1 for n in counts.values() if n > 0)


def shannon_index(counts: dict[str, int]) -> float:
    """
    Shannon Diversity Index H'.
    H' = -Σ(pi × ln(pi))   where  pi = ni / N

    Returns 0.0 if total count is 0 or only 1 species present.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for n in counts.values():
        if n > 0:
            pi = n / total
            h -= pi * math.log(pi)
    return round(h, 6)


def simpsons_index(counts: dict[str, int]) -> float:
    """
    Simpson's Diversity Index D = 1 - Σ(pi²).
    Range [0, 1].  0 = no diversity, 1 = maximum diversity.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    d = sum((n / total) ** 2 for n in counts.values() if n > 0)
    return round(1.0 - d, 6)


def pielou_evenness(counts: dict[str, int]) -> float:
    """
    Pielou's Evenness J' = H' / ln(S).
    Range [0, 1].  1 = perfectly even distribution.
    Returns 0.0 if S <= 1 (undefined / trivial).
    """
    s = species_richness(counts)
    if s <= 1:
        return 0.0
    h = shannon_index(counts)
    return round(h / math.log(s), 6)


def shannon_grade(h: float, s: int = 1) -> tuple[str, str]:
    """
    Grade H' against H'_max=ln(S) so grading is fair for any richness S.
    ratio = H' / H'_max  (0–1)
    """
    h_max = math.log(s) if s > 1 else 1.0
    ratio = min(1.0, h / h_max) if h_max > 0 else 0.0
    for threshold, label, emoji in SHANNON_THRESHOLDS:
        if ratio >= threshold:
            return label, emoji
    return "Poor", "🔴"


# ── Summary function ──────────────────────────────────────────────────────────

def compute_all_indices(counts: dict[str, int]) -> dict:
    """
    Compute all core indices from a species count dict.

    Args:
        counts: e.g. {"Butterflyfish": 56, "Parrotfish": 163, "Angelfish": 271}

    Returns:
        dict with S, H, D, J, grade, emoji, proportions
    """
    total = sum(counts.values())
    s     = species_richness(counts)
    h     = shannon_index(counts)
    d     = simpsons_index(counts)
    j     = pielou_evenness(counts)
    grade, emoji = shannon_grade(h, s)

    proportions = {
        sp: round(n / total, 4) if total > 0 else 0.0
        for sp, n in counts.items()
    }

    return {
        "counts":           counts,
        "total_individuals":total,
        "species_richness": s,
        "shannon_H":        h,
        "simpsons_D":       d,
        "pielou_J":         j,
        "shannon_grade":    grade,
        "shannon_emoji":    emoji,
        "proportions":      proportions,
    }


def print_indices(result: dict) -> None:
    """Pretty-print a compute_all_indices result."""
    print(f"\n  {'─'*50}")
    print(f"  BIODIVERSITY INDICES")
    print(f"  {'─'*50}")
    print(f"  Species Richness   S  = {result['species_richness']}")
    _h    = result['shannon_H']
    _s    = result['species_richness']
    _hmax = math.log(_s) if _s > 1 else 1.0
    _ratio= _h / _hmax if _hmax > 0 else 0.0
    print(f"  Shannon Index      H' = {_h:.4f}  "
          f"(H'_max={_hmax:.4f}, ratio={_ratio:.3f})"
          f"  →  {result['shannon_emoji']} {result['shannon_grade']}")
    print(f"  Simpson's Index    D  = {result['simpsons_D']:.4f}")
    print(f"  Pielou's Evenness  J' = {result['pielou_J']:.4f}")
    print(f"\n  Species proportions:")
    for sp, prop in result['proportions'].items():
        bar = "█" * int(prop * 30)
        print(f"    {sp:<16s} {prop*100:5.1f}%  {bar}")
    print(f"  {'─'*50}\n")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Use actual Phase 4 detection counts from tracking_log.csv
    test_counts = {
        "Butterflyfish": 56,
        "Parrotfish":    163,
        "Angelfish":     271,
    }
    print("\n  Input counts:", test_counts)
    result = compute_all_indices(test_counts)
    print_indices(result)