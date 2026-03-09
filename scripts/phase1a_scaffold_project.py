#!/usr/bin/env python3
"""
PHASE 1A - Project Structure Scaffold
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Creates the complete workspace directory tree, starter config files,
.gitignore, README.md, and requirements.txt.

Usage (run once from your project root):
    python scripts/phase1a_scaffold_project.py
"""

import os
import json
import textwrap
from pathlib import Path

# -----------------------------------------------------------------------------
# Directory tree
# -----------------------------------------------------------------------------

DIRECTORIES: list[str] = [
    # Raw data -- one sub-folder per collection source
    "dataset/raw/roboflow",
    "dataset/raw/kaggle",
    "dataset/raw/lila",
    "dataset/raw/inaturalist",
    # Intermediate pipeline stages
    "dataset/cleaned",
    "dataset/standardized",
    "dataset/augmented",
    # Final split ready for YOLOv8
    "dataset/processed/train/images",
    "dataset/processed/train/labels",
    "dataset/processed/val/images",
    "dataset/processed/val/labels",
    "dataset/processed/test/images",
    "dataset/processed/test/labels",
    # Source code
    "src/utils",
    "src/enhancement",
    "src/training",
    "src/evaluation",
    "src/deployment",
    # Config + scripts
    "configs",
    "scripts",
    # Model artefacts
    "models/weights",
    "models/onnx",
    # Results
    "results/cleaning",
    "results/detection",
    "results/enhancement",
    "results/tracking",
    "results/biodiversity",
    # Docs + notebooks
    "docs",
    "notebooks",
    "logs",
]

# -----------------------------------------------------------------------------
# File templates
# -----------------------------------------------------------------------------

GITIGNORE = textwrap.dedent("""\
    # Python
    __pycache__/
    *.pyc
    *.pyo
    *.pyd
    .Python
    *.egg-info/
    dist/
    build/

    # Virtual environments
    marine_venv/
    venv/
    .venv/
    env/

    # Jupyter
    .ipynb_checkpoints/
    *.ipynb_checkpoints/

    # Model weights (large files -- store in DVC or Google Drive)
    *.pt
    *.pth
    *.onnx
    *.engine

    # Datasets (too large for Git)
    dataset/raw/
    dataset/processed/
    dataset/cleaned/
    dataset/standardized/
    dataset/augmented/

    # Results / logs
    logs/
    runs/
    *.log

    # OS
    .DS_Store
    Thumbs.db

    # IDE
    .vscode/
    .idea/
    *.swp

    # Secrets (API keys, credentials)
    .env
    kaggle.json
    *.key
""")

README = textwrap.dedent("""\
    #  Marine Biodiversity Ecosystem Health Assessment

    **AI-Powered Reef Health Monitoring via Indicator Species Detection**

    > Author: Shri Harsan M | M.Tech Data Science | SRM Institute
    > Version: 2.0 | March 2026

    ---

    ## Project Overview

    This project uses deep learning (YOLOv8) to detect and track **indicator species**
    in underwater video, producing real-time reef health scores for conservation managers.

    ### Indicator Species (8 families)
    | # | Family | Ecological Role |
    |---|--------|----------------|
    | 0 | Butterflyfish | Coral health proxy |
    | 1 | Grouper | Apex predator / fishing pressure |
    | 2 | Parrotfish | Algae control / bioerosion |
    | 3 | Surgeonfish | Herbivore guild |
    | 4 | Damselfish | Habitat complexity |
    | 5 | Wrasse | Ecosystem services |
    | 6 | Triggerfish | Degradation sensitivity |
    | 7 | Angelfish | Reef structure indicator |

    ---

    ## Quick Start

    ```bash
    # 1. Create & activate virtual environment
    python3.11 -m venv marine_venv
    source marine_venv/bin/activate

    # 2. Install dependencies
    pip install -r requirements.txt

    # 3. Verify environment
    python scripts/phase1a_verify_installation.py

    # 4. Scaffold workspace (already done if you see this file)
    python scripts/phase1a_scaffold_project.py

    # 5. Download datasets
    python src/utils/phase1b_download_roboflow.py
    python src/utils/phase1b_download_kaggle.py

    # 6. Clean data
    python src/utils/phase1c_remove_duplicates.py --input_dir dataset/raw
    python src/utils/phase1c_filter_quality.py    --input_dir dataset/raw
    python src/utils/phase1c_validate_annotations.py
    python src/utils/phase1c_standardize_labels.py
    python src/utils/phase1c_generate_statistics.py

    # 7. Augment & split
    python src/utils/phase1d_analyze_balance.py
    python src/utils/phase1d_augment_dataset.py
    python src/utils/phase1d_split_dataset.py
    python src/utils/phase1d_create_data_yaml.py
    ```

    ---

    ## Repository Structure

    ```
    marine_biodiversity_project/
    +-- dataset/           # All data (gitignored)
    +-- src/               # Source code
    |   +-- utils/         # Phase 1 data pipeline
    |   +-- enhancement/   # Phase 2 image enhancement
    |   +-- training/      # Phase 3 YOLOv8 training
    |   \-- evaluation/    # Phase 3 metrics & analysis
    +-- configs/           # YAML configuration files
    +-- scripts/           # One-time setup & verification scripts
    +-- models/            # Trained weights (gitignored)
    +-- results/           # Plots, reports, metrics
    +-- docs/              # Phase completion reports
    \-- notebooks/         # Exploratory Jupyter notebooks
    ```

    ---

    ## Citation

    ```bibtex
    @misc{harsan2026marine,
      author  = {Harsan, Shri M.},
      title   = {Marine Biodiversity Ecosystem Health Assessment},
      year    = {2026},
      school  = {SRM Institute of Science and Technology},
      url     = {https://github.com/[username]/marine-biodiversity-ai}
    }
    ```
""")

REQUIREMENTS = textwrap.dedent("""\
    # -------------------------------------------------------------------------
    # Marine Biodiversity Project -- Python Requirements
    # Install: pip install -r requirements.txt
    # PyTorch (CUDA 11.8) must be installed separately -- see README.md
    # -------------------------------------------------------------------------

    # Deep learning
    # torch==2.0.1          # Install via: pip install torch==2.0.1+cu118 ...
    # torchvision==0.15.2   # Install via: pip install torchvision==0.15.2+cu118
    ultralytics==8.0.196

    # Computer vision & image processing
    opencv-python==4.8.1.78
    Pillow>=10.0.0
    imagehash>=4.3.1

    # Numerical / data science
    numpy==1.26.4
    pandas>=2.1.0
    scipy>=1.11.0

    # Visualisation
    matplotlib>=3.8.0
    seaborn>=0.13.0

    # ML utilities
    scikit-learn>=1.3.0

    # Configuration & serialisation
    PyYAML>=6.0.1
    pydantic>=2.0.0

    # Progress & logging
    tqdm>=4.66.0
    loguru>=0.7.0

    # Dataset APIs
    roboflow>=1.0.0
    kaggle>=1.5.16

    # Jupyter (optional but recommended)
    jupyterlab>=4.0.0
    ipywidgets>=8.0.0

    # Testing
    pytest>=7.4.0
""")

PROJECT_CONFIG = {
    "project": {
        "name": "Marine Biodiversity Ecosystem Health Assessment",
        "author": "Shri Harsan M",
        "institution": "SRM Institute",
        "version": "2.0",
        "date": "2026-03"
    },
    "indicator_classes": {
        "0": "butterflyfish",
        "1": "grouper",
        "2": "parrotfish",
        "3": "surgeonfish",
        "4": "damselfish",
        "5": "wrasse",
        "6": "triggerfish",
        "7": "angelfish"
    },
    "dataset": {
        "target_per_class": 800,
        "min_resolution": 480,
        "train_ratio": 0.85,
        "val_ratio": 0.08,
        "test_ratio": 0.07
    },
    "model": {
        "architecture": "yolov8s",
        "pretrained_weights": "yolov8s.pt",
        "image_size": 640,
        "batch_size": 16,
        "epochs": 100,
        "patience": 20,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.0001
    },
    "augmentation": {
        "hsv_h": 0.02,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "horizontal_flip": 0.5,
        "rotation_degrees": 15,
        "brightness": 0.2,
        "contrast": 0.2,
        "gaussian_noise_sigma": 5
    }
}

# -----------------------------------------------------------------------------
# Scaffold runner
# -----------------------------------------------------------------------------

def scaffold() -> None:
    print("+" + "=" * 68 + "+")
    print("|       MARINE BIODIVERSITY PROJECT -- WORKSPACE SCAFFOLD           |")
    print("+" + "=" * 68 + "+\n")

    # 1. Create directories
    print("Creating directory tree...")
    for d in DIRECTORIES:
        Path(d).mkdir(parents=True, exist_ok=True)
        # Keep empty dirs tracked in git
        gitkeep = Path(d) / ".gitkeep"
        if not any(Path(d).iterdir()):
            gitkeep.touch()
    print(f"  OK {len(DIRECTORIES)} directories created\n")

    # 2. Write root-level files
    files = {
        ".gitignore":            GITIGNORE,
        "README.md":             README,
        "requirements.txt":      REQUIREMENTS,
    }
    for filename, content in files.items():
        p = Path(filename)
        if not p.exists():
            p.write_text(content, encoding="utf-8")
            print(f"  OK Created {filename}")
        else:
            print(f"  -- Skipped {filename} (already exists)")

    # 3. Write project config JSON
    config_path = Path("configs/project_config.json")
    if not config_path.exists():
        config_path.write_text(json.dumps(PROJECT_CONFIG, indent=2), encoding="utf-8")
        print(f"  OK Created configs/project_config.json")
    else:
        print(f"  -- Skipped configs/project_config.json (already exists)")

    # 4. Write species mapping YAML
    species_yaml_path = Path("configs/species_mapping.yaml")
    if not species_yaml_path.exists():
        species_yaml_path.write_text(SPECIES_MAPPING_YAML, encoding="utf-8")
        print("  OK Created configs/species_mapping.yaml")
    else:
        print("  -- Skipped configs/species_mapping.yaml (already exists)")

    print("\n" + "=" * 70)
    print("  DONE: Workspace scaffold complete!")
    print("=" * 70)
    print("\nNext step:")
    print("  python scripts/phase1a_verify_installation.py")


SPECIES_MAPPING_YAML = textwrap.dedent("""\
    # Species -> Indicator Family Mapping
    # Maps granular dataset labels to 8 standardised indicator families.
    # Add / edit species names here as you discover new dataset labels.

    indicator_families:
      butterflyfish:
        class_id: 0
        family: Chaetodontidae
        ecological_role: Coral health proxy (obligate corallivores)
        species:
          - Threadfin Butterflyfish
          - Raccoon Butterflyfish
          - Ornate Butterflyfish
          - Copperband Butterflyfish
          - Teardrop Butterflyfish
          - Saddled Butterflyfish
          - Longnose Butterflyfish
          - Redfin Butterflyfish
          - Spotfin Butterflyfish
          - Foureye Butterflyfish
          - Banded Butterflyfish
          - Reef Butterflyfish

      grouper:
        class_id: 1
        family: Serranidae
        ecological_role: Apex predator / fishing pressure indicator
        species:
          - Nassau Grouper
          - Goliath Grouper
          - Red Grouper
          - Tiger Grouper
          - Black Grouper
          - Yellowfin Grouper
          - Gag Grouper
          - Coney Grouper
          - Coral Grouper
          - Peacock Grouper
          - Potato Grouper
          - Leopard Coral Grouper

      parrotfish:
        class_id: 2
        family: Scaridae
        ecological_role: Algae control / bioerosion balance
        species:
          - Stoplight Parrotfish
          - Rainbow Parrotfish
          - Blue Parrotfish
          - Queen Parrotfish
          - Midnight Parrotfish
          - Princess Parrotfish
          - Striped Parrotfish
          - Redband Parrotfish
          - Bumphead Parrotfish
          - Bicolor Parrotfish

      surgeonfish:
        class_id: 3
        family: Acanthuridae
        ecological_role: Herbivore guild / reef balance
        species:
          - Blue Tang
          - Yellow Tang
          - Sailfin Tang
          - Powder Blue Tang
          - Convict Tang
          - Naso Tang
          - Clown Tang
          - Bluespine Unicornfish
          - Brown Surgeonfish
          - Whitecheek Surgeonfish
          - Orangeband Surgeonfish

      damselfish:
        class_id: 4
        family: Pomacentridae
        ecological_role: Habitat complexity indicator
        species:
          - Sergeant Major
          - Blue Chromis
          - Green Chromis
          - Clownfish
          - Ocellaris Clownfish
          - Percula Clownfish
          - Threespot Damselfish
          - Bicolor Damselfish
          - Yellowtail Damselfish
          - Beau Gregory

      wrasse:
        class_id: 5
        family: Labridae
        ecological_role: Ecosystem services / cleaning stations
        species:
          - Cleaner Wrasse
          - Humphead Wrasse
          - Bird Wrasse
          - Napoleon Wrasse
          - Bluehead Wrasse
          - Yellowhead Wrasse
          - Creole Wrasse
          - Slippery Dick
          - Puddingwife Wrasse
          - Spanish Hogfish

      triggerfish:
        class_id: 6
        family: Balistidae
        ecological_role: Degradation sensitivity indicator
        species:
          - Picasso Triggerfish
          - Clown Triggerfish
          - Titan Triggerfish
          - Queen Triggerfish
          - Undulate Triggerfish
          - Blue Triggerfish
          - Redtooth Triggerfish
          - Crosshatch Triggerfish
          - Bridled Triggerfish
          - Black Triggerfish

      angelfish:
        class_id: 7
        family: Pomacanthidae
        ecological_role: Reef structure indicator
        species:
          - Emperor Angelfish
          - French Angelfish
          - Queen Angelfish
          - King Angelfish
          - Gray Angelfish
          - Rock Beauty
          - Passer Angelfish
          - Koran Angelfish
          - Blue-ringed Angelfish
          - Regal Angelfish
""")


if __name__ == "__main__":
    scaffold()