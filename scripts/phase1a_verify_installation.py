#!/usr/bin/env python3
"""
PHASE 1A - Environment Verification Script
Marine Biodiversity Ecosystem Health Assessment
Author: Shri Harsan M | M.Tech Data Science | SRM Institute

Run this script after setting up your environment to confirm everything is working.
Usage:
    python scripts/phase1a_verify_installation.py

Expected: 7/7 checks pass before proceeding to Phase 1B.
"""

import sys
import subprocess
import importlib
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1: Python Version
# ─────────────────────────────────────────────────────────────────────────────

def check_python_version() -> bool:
    """Verify Python 3.11+ is active."""
    print("\n" + "=" * 70)
    print("  1. Python Version Check")
    print("=" * 70)

    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")

    if version >= (3, 11):
        print("✅ Python 3.11+ detected")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} detected — need 3.11+")
        print("   Fix: sudo apt install python3.11  OR  conda create -n marine_bio python=3.11")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2: CUDA Toolkit (nvcc)
# ─────────────────────────────────────────────────────────────────────────────

def check_cuda_toolkit() -> bool:
    """Verify nvcc (CUDA compiler) is available."""
    print("\n" + "=" * 70)
    print("  2. CUDA Toolkit (nvcc)")
    print("=" * 70)

    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(result.stdout.strip())

        if "release" in result.stdout.lower():
            print("✅ CUDA compiler found")
            return True
        else:
            print("❌ nvcc output unexpected")
            return False

    except FileNotFoundError:
        print("❌ nvcc not found")
        print("   Fix: Install CUDA 11.8 toolkit and add to PATH")
        print("   export PATH=/usr/local/cuda-11.8/bin:$PATH")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3: PyTorch + CUDA Support
# ─────────────────────────────────────────────────────────────────────────────

def check_pytorch_cuda() -> bool:
    """Verify PyTorch is installed and CUDA is accessible."""
    print("\n" + "=" * 70)
    print("  3. PyTorch + CUDA Support")
    print("=" * 70)

    try:
        import torch  # type: ignore

        print(f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {device_name}")
            print(f"VRAM: {vram_gb:.2f} GB")

            if vram_gb < 5.5:
                print("⚠️  Less than 6GB VRAM — use batch_size=8 during training")
            else:
                print("✅ VRAM sufficient for batch_size=16")

            print("✅ PyTorch + CUDA working")
            return True
        else:
            print("❌ torch.cuda.is_available() returned False")
            print("   Fix: pip install torch==2.0.1 torchvision==0.15.2 "
                  "--index-url https://download.pytorch.org/whl/cu118")
            return False

    except ImportError:
        print("❌ PyTorch not installed")
        print("   Fix: pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4: Core Dependencies
# ─────────────────────────────────────────────────────────────────────────────

def check_dependencies() -> bool:
    """Verify all required Python packages are importable."""
    print("\n" + "=" * 70)
    print("  4. Core Dependencies")
    print("=" * 70)

    required = {
        "cv2":           "opencv-python",
        "numpy":         "numpy",
        "pandas":        "pandas",
        "matplotlib":    "matplotlib",
        "seaborn":       "seaborn",
        "sklearn":       "scikit-learn",
        "PIL":           "Pillow",
        "imagehash":     "imagehash",
        "yaml":          "PyYAML",
        "tqdm":          "tqdm",
        "roboflow":      "roboflow",
        "kaggle":        "kaggle",
    }

    all_ok = True
    for module, package in required.items():
        try:
            importlib.import_module(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}  →  pip install {package}")
            all_ok = False

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5: Ultralytics YOLOv8 Inference
# ─────────────────────────────────────────────────────────────────────────────

def check_yolov8_inference() -> bool:
    """Download yolov8s.pt (if needed) and run a smoke-test inference."""
    print("\n" + "=" * 70)
    print("  5. YOLOv8 Inference Test")
    print("=" * 70)

    try:
        from ultralytics import YOLO  # type: ignore
        import numpy as np

        print("Loading YOLOv8s pretrained model (downloads ~22 MB on first run)…")
        model = YOLO("yolov8s.pt")

        # Create a synthetic blank image (640×640 RGB)
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy_image, verbose=False)

        print(f"Ultralytics version: {importlib.import_module('ultralytics').__version__}")
        print("✅ YOLOv8 inference test passed")
        return True

    except Exception as exc:
        print(f"❌ YOLOv8 test failed: {exc}")
        print("   Fix: pip install ultralytics==8.0.196")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6: Project Directory Structure
# ─────────────────────────────────────────────────────────────────────────────

def check_project_structure() -> bool:
    """Confirm all expected project directories exist."""
    print("\n" + "=" * 70)
    print("  6. Project Directory Structure")
    print("=" * 70)

    required_dirs = [
        "dataset/raw",
        "dataset/processed",
        "dataset/augmented",
        "dataset/standardized",
        "src/utils",
        "src/enhancement",
        "src/training",
        "src/evaluation",
        "configs",
        "models/weights",
        "results/cleaning",
        "results/detection",
        "results/enhancement",
        "scripts",
        "docs",
        "notebooks",
    ]

    all_ok = True
    for d in required_dirs:
        path = Path(d)
        if path.exists():
            print(f"  ✅ {d}/")
        else:
            print(f"  ❌ Missing: {d}/  →  mkdir -p {d}")
            all_ok = False

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 7: Git Repository
# ─────────────────────────────────────────────────────────────────────────────

def check_git_repository() -> bool:
    """Confirm git is initialised in the current working directory."""
    print("\n" + "=" * 70)
    print("  7. Git Repository")
    print("=" * 70)

    if Path(".git").exists():
        result = subprocess.run(
            ["git", "log", "--oneline", "-3"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            print(result.stdout.strip())
            print("✅ Git repository with commits found")
        else:
            print("✅ Git repository initialised (no commits yet — run: git commit)")
        return True
    else:
        print("❌ Git not initialised")
        print("   Fix: git init && git add . && git commit -m 'Phase 1A: Environment setup'")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("╔" + "═" * 68 + "╗")
    print("║    MARINE BIODIVERSITY PROJECT — INSTALLATION VERIFICATION       ║")
    print("╚" + "═" * 68 + "╝")

    checks = {
        "Python 3.11+":       check_python_version,
        "CUDA Toolkit":       check_cuda_toolkit,
        "PyTorch + CUDA":     check_pytorch_cuda,
        "Dependencies":       check_dependencies,
        "YOLOv8 Inference":   check_yolov8_inference,
        "Project Structure":  check_project_structure,
        "Git Repository":     check_git_repository,
    }

    results: dict[str, bool] = {}
    for name, fn in checks.items():
        results[name] = fn()

    passed = sum(results.values())
    total = len(results)

    print("\n" + "=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")

    print("=" * 70)
    print(f"  PASSED: {passed}/{total} checks")
    print("=" * 70 + "\n")

    if passed == total:
        print("🎉 All checks passed! Environment is ready for Phase 1B.")
        print("\nNext Steps:")
        print("  1. Review PHASE1B_DATASET_COLLECTION.txt")
        print("  2. Start collecting datasets from Roboflow, Kaggle, etc.")
        print("  3. Document progress in docs/PHASE1_PROGRESS.md")
        return 0
    else:
        print("⚠️  Some checks failed. Fix the issues above before Phase 1B.")
        print("\nTroubleshooting:")
        print("  - CUDA: nvcc --version")
        print("  - PyTorch: pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118")
        print("  - GPU: nvidia-smi")
        return 1


if __name__ == "__main__":
    sys.exit(main())
