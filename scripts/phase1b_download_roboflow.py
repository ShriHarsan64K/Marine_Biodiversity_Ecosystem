#!/usr/bin/env python3
"""
PHASE 1B — Download Datasets from Roboflow Universe
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Prerequisites:
    pip install roboflow
    Set your Roboflow API key below OR export ROBOFLOW_API_KEY=<your_key>

Usage:
    python src/utils/phase1b_download_roboflow.py

Output directory:  dataset/raw/roboflow/
"""

import os
import json
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit this section
# ─────────────────────────────────────────────────────────────────────────────

# Get API key from https://app.roboflow.com/settings/api
# Leave as empty string and set env var instead for security:
#   export ROBOFLOW_API_KEY="your_key_here"
ROBOFLOW_API_KEY: str = os.environ.get("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")

# Datasets to download (YOLOv8 format)
# Find workspace/project names in the Roboflow URL:
#   https://universe.roboflow.com/<workspace>/<project>/dataset/<version>
DATASETS: list[dict] = [
    {
        "workspace": "fish-detection-pzp8o",
        "project":   "underwater-fish-detection",
        "version":   1,
        "notes":     "Underwater fish with indicator species (grouper, parrotfish, surgeonfish)"
    },
    {
        "workspace": "marine-biodiversity",
        "project":   "coral-reef-fish",
        "version":   2,
        "notes":     "Coral reef fish (butterflyfish, angelfish, damselfish)"
    },
    {
        "workspace": "tropical-fish",
        "project":   "tropical-fish-dataset",
        "version":   1,
        "notes":     "Tropical fish (wrasse, triggerfish)"
    },
]

OUTPUT_ROOT = Path("dataset/raw/roboflow")

# ─────────────────────────────────────────────────────────────────────────────
# Download function
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset(api_key: str, workspace: str, project: str, version: int,
                     output_dir: Path) -> Path | None:
    """
    Download a Roboflow dataset in YOLOv8 format.

    Args:
        api_key:    Roboflow API key
        workspace:  Roboflow workspace slug
        project:    Roboflow project slug
        version:    Dataset version number
        output_dir: Directory to save the dataset

    Returns:
        Path to downloaded dataset, or None on failure
    """
    try:
        from roboflow import Roboflow  # type: ignore

        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8", location=str(output_dir / project))
        return Path(dataset.location)

    except ImportError:
        print("  ❌ roboflow not installed — pip install roboflow")
        return None
    except Exception as exc:
        print(f"  ❌ Download failed: {exc}")
        print("  ℹ  Manual alternative:")
        print(f"     1. Visit https://universe.roboflow.com/{workspace}/{project}")
        print(f"     2. Click 'Download Dataset' → YOLOv8 format → Download ZIP")
        print(f"     3. Unzip to: {output_dir / project}/")
        return None


def manual_download_instructions(datasets: list[dict], output_dir: Path) -> None:
    """Print manual download steps for all datasets."""
    print("\n" + "─" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS (if API fails)")
    print("─" * 70)
    for i, ds in enumerate(datasets, 1):
        print(f"\n{i}. {ds['project']}")
        print(f"   URL:    https://universe.roboflow.com/{ds['workspace']}/{ds['project']}")
        print(f"   Format: YOLOv8")
        print(f"   Save to: {output_dir / ds['project']}/")
        print(f"   Notes:  {ds['notes']}")


def record_provenance(datasets: list[dict], output_dir: Path,
                      locations: dict) -> None:
    """Save a provenance JSON log for licence compliance."""
    record = {
        "source":        "Roboflow Universe",
        "download_date": datetime.now().isoformat(),
        "datasets":      []
    }

    for ds in datasets:
        entry = {
            "workspace": ds["workspace"],
            "project":   ds["project"],
            "version":   ds["version"],
            "notes":     ds["notes"],
            "url":       f"https://universe.roboflow.com/{ds['workspace']}/{ds['project']}",
            "local_path": str(locations.get(ds["project"], "not_downloaded")),
            "license":   "CC-BY 4.0 (verify per dataset)"
        }
        record["datasets"].append(entry)

    log_path = output_dir / "download_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(record, indent=2))
    print(f"\n✅ Provenance log saved: {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("╔" + "═" * 68 + "╗")
    print("║         PHASE 1B — ROBOFLOW DATASET DOWNLOAD                    ║")
    print("╚" + "═" * 68 + "╝\n")

    if ROBOFLOW_API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️  No API key set!")
        print("   Option 1: Edit ROBOFLOW_API_KEY in this script")
        print("   Option 2: export ROBOFLOW_API_KEY=<your_key>")
        print()
        manual_download_instructions(DATASETS, OUTPUT_ROOT)
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    locations: dict[str, str] = {}

    for ds in DATASETS:
        print(f"\nDownloading: {ds['project']} (version {ds['version']})")
        print(f"  Notes: {ds['notes']}")
        path = download_dataset(
            ROBOFLOW_API_KEY,
            ds["workspace"],
            ds["project"],
            ds["version"],
            OUTPUT_ROOT
        )
        if path:
            locations[ds["project"]] = str(path)
            print(f"  ✅ Saved to: {path}")

            # Count images
            images = list(path.rglob("*.jpg")) + list(path.rglob("*.png"))
            print(f"  📸 Images found: {len(images)}")
        else:
            print(f"  ⚠️  {ds['project']} not downloaded")

    record_provenance(DATASETS, OUTPUT_ROOT, locations)

    print("\n" + "=" * 70)
    print("  PHASE 1B — Roboflow download complete")
    print("=" * 70)
    print("\nNext: python src/utils/phase1b_download_kaggle.py")


if __name__ == "__main__":
    main()
