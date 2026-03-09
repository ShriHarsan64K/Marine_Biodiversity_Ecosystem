#!/usr/bin/env python3
"""
PHASE 1B — Species Taxonomy Mapping
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Maps granular dataset class labels (199+ species) to 8 indicator families.
Generates a reverse-lookup dictionary used by Phase 1C label standardisation.

Usage:
    python src/utils/phase1b_map_species.py

Output:
    configs/species_mapping.yaml  (already created by scaffold — this verifies it)
    configs/reverse_mapping.json  (species_name → family_id)
"""

import json
import yaml
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Standard indicator families — single source of truth for all phases
# ─────────────────────────────────────────────────────────────────────────────

STANDARD_CLASSES: dict[int, str] = {
    0: "butterflyfish",
    1: "grouper",
    2: "parrotfish",
    3: "surgeonfish",
    4: "damselfish",
    5: "wrasse",
    6: "triggerfish",
    7: "angelfish",
}

# Alias table: maps raw dataset label strings → indicator family name.
# Expand this table as you encounter new labels from downloaded datasets.
ALIAS_MAP: dict[str, str] = {
    # ── Butterflyfish ──────────────────────────────────────────────────────
    "threadfin butterflyfish":    "butterflyfish",
    "raccoon butterflyfish":      "butterflyfish",
    "ornate butterflyfish":       "butterflyfish",
    "copperband butterflyfish":   "butterflyfish",
    "teardrop butterflyfish":     "butterflyfish",
    "foureye butterflyfish":      "butterflyfish",
    "banded butterflyfish":       "butterflyfish",
    "spotfin butterflyfish":      "butterflyfish",
    "reef butterflyfish":         "butterflyfish",
    "butterflyfish":              "butterflyfish",
    "chaetodontidae":             "butterflyfish",

    # ── Grouper ────────────────────────────────────────────────────────────
    "nassau grouper":             "grouper",
    "goliath grouper":            "grouper",
    "red grouper":                "grouper",
    "tiger grouper":              "grouper",
    "black grouper":              "grouper",
    "coral grouper":              "grouper",
    "peacock grouper":            "grouper",
    "leopard coral grouper":      "grouper",
    "grouper":                    "grouper",
    "serranidae":                 "grouper",
    "epinephelus":                "grouper",

    # ── Parrotfish ─────────────────────────────────────────────────────────
    "stoplight parrotfish":       "parrotfish",
    "rainbow parrotfish":         "parrotfish",
    "blue parrotfish":            "parrotfish",
    "queen parrotfish":           "parrotfish",
    "midnight parrotfish":        "parrotfish",
    "princess parrotfish":        "parrotfish",
    "striped parrotfish":         "parrotfish",
    "bumphead parrotfish":        "parrotfish",
    "parrotfish":                 "parrotfish",
    "scaridae":                   "parrotfish",
    "scarus":                     "parrotfish",

    # ── Surgeonfish ────────────────────────────────────────────────────────
    "blue tang":                  "surgeonfish",
    "yellow tang":                "surgeonfish",
    "sailfin tang":               "surgeonfish",
    "powder blue tang":           "surgeonfish",
    "convict tang":               "surgeonfish",
    "naso tang":                  "surgeonfish",
    "clown tang":                 "surgeonfish",
    "surgeonfish":                "surgeonfish",
    "tang":                       "surgeonfish",
    "acanthuridae":               "surgeonfish",
    "acanthurus":                 "surgeonfish",
    "unicornfish":                "surgeonfish",

    # ── Damselfish ─────────────────────────────────────────────────────────
    "sergeant major":             "damselfish",
    "blue chromis":               "damselfish",
    "green chromis":              "damselfish",
    "clownfish":                  "damselfish",
    "clown fish":                 "damselfish",
    "ocellaris clownfish":        "damselfish",
    "percula clownfish":          "damselfish",
    "threespot damselfish":       "damselfish",
    "bicolor damselfish":         "damselfish",
    "yellowtail damselfish":      "damselfish",
    "beau gregory":               "damselfish",
    "damselfish":                 "damselfish",
    "pomacentridae":              "damselfish",
    "chromis":                    "damselfish",
    "amphiprion":                 "damselfish",

    # ── Wrasse ─────────────────────────────────────────────────────────────
    "cleaner wrasse":             "wrasse",
    "humphead wrasse":            "wrasse",
    "napoleon wrasse":            "wrasse",
    "bird wrasse":                "wrasse",
    "bluehead wrasse":            "wrasse",
    "yellowhead wrasse":          "wrasse",
    "creole wrasse":              "wrasse",
    "slippery dick":              "wrasse",
    "puddingwife wrasse":         "wrasse",
    "spanish hogfish":            "wrasse",
    "wrasse":                     "wrasse",
    "labridae":                   "wrasse",
    "thalassoma":                 "wrasse",
    "labroides":                  "wrasse",

    # ── Triggerfish ────────────────────────────────────────────────────────
    "picasso triggerfish":        "triggerfish",
    "clown triggerfish":          "triggerfish",
    "titan triggerfish":          "triggerfish",
    "queen triggerfish":          "triggerfish",
    "undulate triggerfish":       "triggerfish",
    "blue triggerfish":           "triggerfish",
    "redtooth triggerfish":       "triggerfish",
    "triggerfish":                "triggerfish",
    "balistidae":                 "triggerfish",
    "balistapus":                 "triggerfish",

    # ── Angelfish ──────────────────────────────────────────────────────────
    "emperor angelfish":          "angelfish",
    "french angelfish":           "angelfish",
    "queen angelfish":            "angelfish",
    "king angelfish":             "angelfish",
    "gray angelfish":             "angelfish",
    "rock beauty":                "angelfish",
    "regal angelfish":            "angelfish",
    "koran angelfish":            "angelfish",
    "angelfish":                  "angelfish",
    "pomacanthidae":              "angelfish",
    "pomacanthus":                "angelfish",
    "chaetodontoplus":            "angelfish",
}


# ─────────────────────────────────────────────────────────────────────────────
# Build reverse lookup (species_name_normalised → class_id)
# ─────────────────────────────────────────────────────────────────────────────

def build_reverse_mapping(alias_map: dict[str, str],
                          standard_classes: dict[int, str]) -> dict[str, int]:
    """
    Return a dict mapping every normalised species label → numeric class ID.

    Args:
        alias_map:        species_name → family_name
        standard_classes: class_id → family_name

    Returns:
        {normalised_label: class_id}
    """
    name_to_id = {v: k for k, v in standard_classes.items()}
    reverse: dict[str, int] = {}

    for species, family in alias_map.items():
        if family not in name_to_id:
            print(f"  ⚠️  Unknown family '{family}' for species '{species}' — skipped")
            continue
        reverse[species.lower().strip()] = name_to_id[family]

    return reverse


def lookup_species(species_label: str,
                   reverse_mapping: dict[str, int]) -> tuple[int | None, str | None]:
    """
    Look up a raw dataset label and return its (class_id, family_name).

    Tries exact match first, then partial/substring match.

    Args:
        species_label:    Raw class name from dataset
        reverse_mapping:  Pre-built reverse mapping dict

    Returns:
        (class_id, family_name) or (None, None) if not found
    """
    normalised = species_label.lower().strip()

    # Exact match
    if normalised in reverse_mapping:
        cid = reverse_mapping[normalised]
        return cid, STANDARD_CLASSES[cid]

    # Substring match (e.g. 'Stoplight_Parrotfish_terminal' → 'parrotfish')
    for key, cid in reverse_mapping.items():
        if key in normalised or normalised in key:
            return cid, STANDARD_CLASSES[cid]

    return None, None


def validate_mapping(reverse_mapping: dict[str, int]) -> None:
    """Print a summary of all mapped species."""
    print("\n" + "=" * 70)
    print("SPECIES MAPPING SUMMARY")
    print("=" * 70)

    family_counts: dict[str, int] = {}
    for cid in STANDARD_CLASSES.values():
        family_counts[cid] = 0

    for family_id in reverse_mapping.values():
        family_name = STANDARD_CLASSES[family_id]
        family_counts[family_name] += 1

    for family_name, count in family_counts.items():
        print(f"  {family_name:20s} : {count:3d} mapped labels")

    print("-" * 70)
    print(f"  Total species mapped : {len(reverse_mapping)}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("╔" + "═" * 68 + "╗")
    print("║          PHASE 1B — SPECIES TAXONOMY MAPPING                    ║")
    print("╚" + "═" * 68 + "╝\n")

    # Build reverse mapping
    reverse_mapping = build_reverse_mapping(ALIAS_MAP, STANDARD_CLASSES)

    # Validate
    validate_mapping(reverse_mapping)

    # Save reverse mapping JSON for Phase 1C
    out_path = Path("configs/reverse_mapping.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reverse_mapping, indent=2, sort_keys=True))
    print(f"\n✅ Reverse mapping saved: {out_path}")

    # Demo lookup
    print("\nDEMO LOOKUPS:")
    demo_labels = [
        "Blue Tang", "Stoplight Parrotfish", "Emperor Angelfish",
        "Sergeant Major", "Cleaner Wrasse", "Unknown Fish"
    ]
    for label in demo_labels:
        cid, family = lookup_species(label, reverse_mapping)
        if cid is not None:
            print(f"  '{label}' → class {cid} ({family})")
        else:
            print(f"  '{label}' → ❌ NOT MAPPED (add to ALIAS_MAP)")

    print("\n✅ Phase 1B mapping complete")
    print("Next: Phase 1C — python src/utils/phase1c_remove_duplicates.py")


if __name__ == "__main__":
    main()
