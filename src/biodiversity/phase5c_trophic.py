"""
PHASE 5C — Trophic Pyramid Analysis
=====================================
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Assigns each species to a functional/trophic group and checks whether
the observed distribution matches a healthy reef pyramid.

Healthy Reef Pyramid (from roadmap / literature):
  Apex Predators   :  5-10%
  Mesopredators    : 15-20%
  Herbivores       : 35-45%
  Planktivores     : 25-35%
  Corallivores     :  5-10%
  Cleaners         :  1-5%
"""

# ── Trophic group assignments ─────────────────────────────────────────────────
TROPHIC_GROUPS: dict[str, str] = {
    # Apex predators
    "Sharks":         "Apex Predators",
    "Grouper":        "Apex Predators",
    # Mesopredators
    "Triggerfish":    "Mesopredators",
    # Herbivores
    "Parrotfish":     "Herbivores",
    "Surgeonfish":    "Herbivores",
    # Planktivores
    "Damselfish":     "Planktivores",
    # Corallivores (specialised)
    "Butterflyfish":  "Corallivores",
    "Angelfish":      "Corallivores",
    # Cleaners
    "Cleaner Wrasse": "Cleaners",
    "Wrasse":         "Cleaners",
}
DEFAULT_GROUP = "Other"

# Healthy reef target ranges  {group: (min_pct, max_pct)}
HEALTHY_RANGES: dict[str, tuple[float, float]] = {
    "Apex Predators": (5.0,  10.0),
    "Mesopredators":  (15.0, 20.0),
    "Herbivores":     (35.0, 45.0),
    "Planktivores":   (25.0, 35.0),
    "Corallivores":   (5.0,  10.0),
    "Cleaners":       (1.0,  5.0),
    "Other":          (0.0,  100.0),
}

# Degradation signals
DEGRADATION_RULES = [
    ("Apex Predators", "<",  2.0, "🔴 <2% apex predators — severe overfishing signal"),
    ("Herbivores",     ">", 60.0, "🟠 >60% herbivores — mesopredator collapse likely"),
    ("Corallivores",   "<",  3.0, "🟠 <3% corallivores — coral cover loss risk"),
    ("Apex Predators", "=",  0.0, "🔴 Zero apex predators detected"),
]


def get_trophic_group(species: str) -> str:
    return TROPHIC_GROUPS.get(species, DEFAULT_GROUP)


def trophic_analysis(counts: dict[str, int]) -> dict:
    """
    Analyse trophic structure of detected fish community.

    Args:
        counts: {species_name: count}

    Returns:
        dict with group_counts, group_pcts, trophic_balance_score,
        status_per_group, degradation_signals, overall_status
    """
    total = sum(counts.values())
    if total == 0:
        return {
            "trophic_balance_score": 0.0,
            "group_counts": {},
            "group_pcts": {},
            "status_per_group": {},
            "degradation_signals": ["No fish detected"],
            "overall_trophic_status": "Unknown",
        }

    # Aggregate into trophic groups
    group_counts: dict[str, int] = {}
    for sp, n in counts.items():
        grp = get_trophic_group(sp)
        group_counts[grp] = group_counts.get(grp, 0) + n

    group_pcts = {
        grp: round(n / total * 100, 2)
        for grp, n in group_counts.items()
    }

    # Determine which trophic groups are POSSIBLE given the species in counts.
    # Groups not represented by ANY species in counts are excluded from scoring
    # (we cannot penalise a 3-class dataset for lacking Sharks/Grouper).
    possible_groups: set[str] = set()
    for sp in counts:
        possible_groups.add(TROPHIC_GROUPS.get(sp, DEFAULT_GROUP))
    possible_groups.discard("Other")

    # Score each group: 100 if inside healthy range, penalise proportionally
    status_per_group: dict[str, dict] = {}
    total_score   = 0.0
    scored_groups = 0

    for grp, (lo, hi) in HEALTHY_RANGES.items():
        if grp == "Other":
            continue

        pct = group_pcts.get(grp, 0.0)

        # Groups not in our species scope → mark informational only, skip scoring
        if grp not in possible_groups:
            status_per_group[grp] = {
                "count":      0,
                "pct":        0.0,
                "target_lo":  lo,
                "target_hi":  hi,
                "score":      None,   # not scored
                "status":     "⬜ Not in dataset scope",
            }
            continue

        if lo <= pct <= hi:
            score  = 100.0
            status = "✅ Healthy"
        elif pct == 0.0 and lo == 0.0:
            score  = 100.0
            status = "✅ Not expected"
        elif pct < lo:
            score  = max(0.0, 100.0 - (lo - pct) / lo * 100)
            status = f"🟡 Low ({pct:.1f}% vs target {lo}-{hi}%)"
        else:
            score  = max(0.0, 100.0 - (pct - hi) / hi * 100)
            status = f"🟠 High ({pct:.1f}% vs target {lo}-{hi}%)"

        status_per_group[grp] = {
            "count":      group_counts.get(grp, 0),
            "pct":        pct,
            "target_lo":  lo,
            "target_hi":  hi,
            "score":      round(score, 2),
            "status":     status,
        }
        total_score   += score
        scored_groups += 1

    trophic_balance_score = round(total_score / scored_groups, 2) if scored_groups else 0.0

    # Degradation signals — only fire for groups within dataset scope
    signals = []
    for grp, op, threshold, msg in DEGRADATION_RULES:
        if grp not in possible_groups:
            continue   # can't signal degradation for species not in dataset
        pct = group_pcts.get(grp, 0.0)
        if op == "<" and pct < threshold:
            signals.append(msg)
        elif op == ">" and pct > threshold:
            signals.append(msg)
        elif op == "=" and pct == threshold:
            signals.append(msg)

    if not signals:
        signals.append("✅ No degradation signals detected within dataset scope")

    # Overall trophic status
    if trophic_balance_score >= 80:
        overall = "Healthy"
    elif trophic_balance_score >= 60:
        overall = "Moderate Stress"
    elif trophic_balance_score >= 40:
        overall = "Stressed"
    else:
        overall = "Degraded"

    return {
        "trophic_balance_score":  trophic_balance_score,
        "group_counts":           group_counts,
        "group_pcts":             group_pcts,
        "status_per_group":       status_per_group,
        "degradation_signals":    signals,
        "overall_trophic_status": overall,
    }


def print_trophic(result: dict) -> None:
    print(f"\n  {'─'*58}")
    print(f"  TROPHIC PYRAMID ANALYSIS")
    print(f"  {'─'*58}")
    print(f"  Trophic Balance Score : {result['trophic_balance_score']:.1f} / 100")
    print(f"  Overall Status        : {result['overall_trophic_status']}")
    print(f"\n  {'Group':<20s} {'Count':>6}  {'Pct':>6}  {'Target':>12}  Status")
    print(f"  {'─'*58}")
    for grp, info in result['status_per_group'].items():
        target = f"{info['target_lo']}-{info['target_hi']}%"
        score_str = f"{info['score']:.1f}" if info['score'] is not None else "  N/A"
        print(f"  {grp:<20s} {info['count']:>6}  {info['pct']:>5.1f}%"
              f"  {target:>12}  {info['status']}")
    print(f"\n  Degradation Signals:")
    for sig in result['degradation_signals']:
        print(f"    {sig}")
    print(f"  {'─'*58}\n")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_counts = {
        "Butterflyfish": 56,
        "Parrotfish":    163,
        "Angelfish":     271,
    }
    print("\n  Input counts:", test_counts)
    result = trophic_analysis(test_counts)
    print_trophic(result)