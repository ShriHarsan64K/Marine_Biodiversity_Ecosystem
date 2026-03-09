#!/usr/bin/env python3
"""
PHASE 1B — Download Datasets from Kaggle
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Prerequisites:
    pip install kaggle
    Place kaggle.json at ~/.kaggle/kaggle.json (chmod 600)
    Get it from: https://www.kaggle.com/settings → API → Create New API Token

Usage:
    python src/utils/phase1b_download_kaggle.py

Output directory:  dataset/raw/kaggle/
"""

import os
import json
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATASETS: list[dict] = [
    {
        "slug":      "crowww/a-large-scale-fish-dataset",
        "folder":    "large-scale-fish",
        "notes":     "9,000 images — multiple species including indicator families",
        "license":   "CC0 (Public Domain)",
        "format":    "classification (no bounding boxes — needs annotation)"
    },
    {
        "slug":      "markdaniellampa/fish-dataset",
        "folder":    "fish-species",
        "notes":     "Grouper, Parrotfish, Surgeonfish with labels",
        "license":   "CC-BY-SA 4.0",
        "format":    "classification"
    },
    {
        "slug":      "sdhayalk/trash-detection-dataset",
        "folder":    "underwater-scenes",
        "notes":     "Underwater negative samples (no fish — hard negatives for training)",
        "license":   "CC-BY",
        "format":    "object detection"
    },
]

OUTPUT_ROOT = Path("dataset/raw/kaggle")

# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def check_kaggle_credentials() -> bool:
    """Verify ~/.kaggle/kaggle.json exists and has correct permissions."""
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        print("❌ ~/.kaggle/kaggle.json not found")
        print("   Steps to fix:")
        print("   1. Go to https://www.kaggle.com/settings")
        print("   2. API section → 'Create New API Token'")
        print("   3. Move downloaded file:  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("   4. Set permissions:       chmod 600 ~/.kaggle/kaggle.json")
        return False

    # Ensure file is not world-readable
    mode = oct(cred_path.stat().st_mode)[-3:]
    if mode != "600":
        cred_path.chmod(0o600)
        print(f"  ℹ  Fixed permissions on {cred_path} (now 600)")

    print("✅ Kaggle credentials found")
    return True


def download_kaggle_dataset(slug: str, output_dir: Path) -> bool:
    """
    Download a Kaggle dataset using the CLI.

    Args:
        slug:       Dataset identifier e.g. 'crowww/a-large-scale-fish-dataset'
        output_dir: Where to unzip the dataset

    Returns:
        True if successful
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading: {slug}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug,
         "--path", str(output_dir), "--unzip"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        images = list(output_dir.rglob("*.jpg")) + list(output_dir.rglob("*.png"))
        print(f"  ✅ Downloaded — {len(images)} images in {output_dir}")
        return True
    else:
        print(f"  ❌ Failed: {result.stderr.strip()}")
        return False


def manual_download_instructions(datasets: list[dict], output_dir: Path) -> None:
    """Print step-by-step manual download instructions."""
    print("\n" + "─" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS (if Kaggle API is unavailable)")
    print("─" * 70)
    for i, ds in enumerate(datasets, 1):
        url = f"https://www.kaggle.com/datasets/{ds['slug']}"
        print(f"\n{i}. {ds['slug']}")
        print(f"   URL:     {url}")
        print(f"   License: {ds['license']}")
        print(f"   Format:  {ds['format']}")
        print(f"   Notes:   {ds['notes']}")
        print(f"   Save to: {output_dir / ds['folder']}/")


def record_provenance(datasets: list[dict], output_dir: Path,
                      results: dict) -> None:
    """Write download provenance to JSON."""
    record = {
        "source":        "Kaggle",
        "download_date": datetime.now().isoformat(),
        "datasets":      []
    }

    for ds in datasets:
        entry = {
            "slug":       ds["slug"],
            "url":        f"https://www.kaggle.com/datasets/{ds['slug']}",
            "license":    ds["license"],
            "format":     ds["format"],
            "notes":      ds["notes"],
            "downloaded": results.get(ds["folder"], False)
        }
        record["datasets"].append(entry)

    log_path = output_dir / "download_log.json"
    log_path.write_text(json.dumps(record, indent=2))
    print(f"\n✅ Provenance log saved: {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("╔" + "═" * 68 + "╗")
    print("║           PHASE 1B — KAGGLE DATASET DOWNLOAD                    ║")
    print("╚" + "═" * 68 + "╝\n")

    if not check_kaggle_credentials():
        manual_download_instructions(DATASETS, OUTPUT_ROOT)
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}

    for ds in DATASETS:
        print(f"\nDataset: {ds['folder']}")
        print(f"  License: {ds['license']}")
        print(f"  Notes:   {ds['notes']}")
        dest = OUTPUT_ROOT / ds["folder"]
        results[ds["folder"]] = download_kaggle_dataset(ds["slug"], dest)

    record_provenance(DATASETS, OUTPUT_ROOT, results)

    passed = sum(results.values())
    print("\n" + "=" * 70)
    print(f"  Downloaded {passed}/{len(DATASETS)} datasets")
    print("=" * 70)
    print("\nNext: python src/utils/phase1b_map_species.py")


if __name__ == "__main__":
    main()
