#!/usr/bin/env python3
"""
PHASE 2B — Enhancement Benchmark & Validation
3-Class Marine Biodiversity Project

Computes SSIM, PSNR, and UCIQE on a random sample of images,
generates side-by-side comparison plots, and saves a JSON report.

Metrics targets (from roadmap):
    SSIM  > 0.80
    PSNR  > 15 dB
    UCIQE > baseline (higher = better underwater colour quality)

Usage:
    python src/enhancement/benchmark_enhancers.py \
        --dataset_dir dataset/processed/train/images \
        --sample      20 \
        --output_dir  results/enhancement

Author: Shri Harsan M | M.Tech Data Science | SRM Institute
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path so we can import ancuti_fusion
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.enhancement.ancuti_fusion import enhance

# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM).
    Measures luminance, contrast, and structure similarity.
    Range: [0, 1], higher is better.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray1, gray2, full=True)
        return float(score)
    except ImportError:
        # Fallback: manual SSIM (single window)
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
        mu1, mu2   = g1.mean(), g2.mean()
        sig1, sig2 = g1.std(), g2.std()
        sig12      = ((g1 - mu1) * (g2 - mu2)).mean()
        num = (2*mu1*mu2 + C1) * (2*sig12 + C2)
        den = (mu1**2 + mu2**2 + C1) * (sig1**2 + sig2**2 + C2)
        return float(num / (den + 1e-10))


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) in dB.
    Measures fidelity between original and enhanced image.
    Target > 15 dB for underwater imagery.
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def compute_uciqe(bgr: np.ndarray) -> float:
    """
    Underwater Colour Image Quality Evaluation (UCIQE).
    Reference: Yang & Sowmya 2015.
    Combines chroma standard deviation, luminance contrast, and saturation mean.
    Higher = better underwater colour quality.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    # Chroma
    chroma    = np.sqrt(a**2 + b**2)
    sigma_c   = chroma.std()

    # Luminance contrast (top 1% - bottom 1%)
    l_flat    = l.flatten()
    con_l     = np.percentile(l_flat, 99) - np.percentile(l_flat, 1)

    # Saturation mean
    hsv       = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    mu_s      = hsv[:, :, 1].mean() / 255.0

    # UCIQE = c1*sigma_c + c2*con_l + c3*mu_s  (standard coefficients)
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    uciqe = c1 * (sigma_c / 128.0) + c2 * (con_l / 255.0) + c3 * mu_s
    return float(uciqe)


# ── Benchmark core ────────────────────────────────────────────────────────────

def benchmark_sample(image_paths: list, n_sample: int) -> dict:
    """
    Run enhancement on n_sample images and compute all metrics.
    Returns summary dict.
    """
    random.shuffle(image_paths)
    sample = image_paths[:n_sample]

    results = []
    print(f"\nBenchmarking {len(sample)} images...")

    for img_path in sample:
        original = cv2.imread(str(img_path))
        if original is None:
            continue
        try:
            enhanced = enhance(original)
        except Exception as e:
            print(f"  ERROR {img_path.name}: {e}")
            continue

        ssim_val  = compute_ssim(original, enhanced)
        psnr_val  = compute_psnr(original, enhanced)
        uciqe_orig = compute_uciqe(original)
        uciqe_enh  = compute_uciqe(enhanced)

        results.append({
            "file":         img_path.name,
            "ssim":         round(ssim_val,  4),
            "psnr_db":      round(psnr_val,  2),
            "uciqe_orig":   round(uciqe_orig, 4),
            "uciqe_enh":    round(uciqe_enh,  4),
            "uciqe_delta":  round(uciqe_enh - uciqe_orig, 4),
        })

    if not results:
        return {}

    ssim_vals  = [r["ssim"]        for r in results]
    psnr_vals  = [r["psnr_db"]     for r in results]
    uciqe_orig = [r["uciqe_orig"]  for r in results]
    uciqe_enh  = [r["uciqe_enh"]   for r in results]

    summary = {
        "n_images":       len(results),
        "ssim_mean":      round(np.mean(ssim_vals), 4),
        "ssim_std":       round(np.std(ssim_vals),  4),
        "ssim_target":    0.80,
        "ssim_pass":      bool(np.mean(ssim_vals) >= 0.80),
        "psnr_mean_db":   round(np.mean(psnr_vals), 2),
        "psnr_std_db":    round(np.std(psnr_vals),  2),
        "psnr_target_db": 15.0,
        "psnr_pass":      bool(np.mean(psnr_vals) >= 15.0),
        "uciqe_orig_mean":round(np.mean(uciqe_orig), 4),
        "uciqe_enh_mean": round(np.mean(uciqe_enh),  4),
        "uciqe_improved": bool(np.mean(uciqe_enh) > np.mean(uciqe_orig)),
        "per_image":      results,
    }
    return summary


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_comparison_grid(image_paths: list, output_path: Path, n: int = 4) -> None:
    """
    Save a side-by-side grid: Original | Enhanced | Diff heatmap
    for the first n images.
    """
    sample = image_paths[:n]
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = [axes]

    for row, img_path in enumerate(sample):
        original = cv2.imread(str(img_path))
        if original is None:
            continue
        enhanced = enhance(original)

        orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        enh_rgb  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        diff     = np.abs(orig_rgb.astype(np.float32) - enh_rgb.astype(np.float32))
        diff_norm = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)

        ssim_val = compute_ssim(original, enhanced)
        psnr_val = compute_psnr(original, enhanced)
        uciqe_o  = compute_uciqe(original)
        uciqe_e  = compute_uciqe(enhanced)

        axes[row][0].imshow(orig_rgb)
        axes[row][0].set_title(f"Original\nUCIQE={uciqe_o:.3f}", fontsize=9)
        axes[row][0].axis("off")

        axes[row][1].imshow(enh_rgb)
        axes[row][1].set_title(f"Enhanced (Ancuti 2012)\nUCIQE={uciqe_e:.3f}  SSIM={ssim_val:.3f}  PSNR={psnr_val:.1f}dB", fontsize=9)
        axes[row][1].axis("off")

        axes[row][2].imshow(diff_norm)
        axes[row][2].set_title("Difference (amplified)", fontsize=9)
        axes[row][2].axis("off")

    plt.suptitle("Phase 2A — Ancuti Underwater Enhancement Results", fontsize=13, y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison grid saved: {output_path}")


def save_metrics_chart(summary: dict, output_path: Path) -> None:
    """Bar chart of mean SSIM, PSNR, UCIQE with target lines."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # SSIM
    ax = axes[0]
    ax.bar(["Mean SSIM"], [summary["ssim_mean"]], color="#42A5F5", alpha=0.85)
    ax.axhline(summary["ssim_target"], color="green", linestyle="--", label=f"Target ({summary['ssim_target']})")
    ax.set_ylim(0, 1.1)
    ax.set_title("SSIM (higher = better)")
    ax.legend(fontsize=8)
    color = "green" if summary["ssim_pass"] else "red"
    ax.text(0, summary["ssim_mean"] + 0.02, f"{summary['ssim_mean']:.3f}", ha="center", color=color, fontweight="bold")

    # PSNR
    ax = axes[1]
    ax.bar(["Mean PSNR"], [summary["psnr_mean_db"]], color="#FF7043", alpha=0.85)
    ax.axhline(summary["psnr_target_db"], color="green", linestyle="--", label=f"Target ({summary['psnr_target_db']} dB)")
    ax.set_title("PSNR in dB (higher = better)")
    ax.legend(fontsize=8)
    color = "green" if summary["psnr_pass"] else "red"
    ax.text(0, summary["psnr_mean_db"] + 0.3, f"{summary['psnr_mean_db']:.1f} dB", ha="center", color=color, fontweight="bold")

    # UCIQE
    ax = axes[2]
    ax.bar(["Original", "Enhanced"],
           [summary["uciqe_orig_mean"], summary["uciqe_enh_mean"]],
           color=["#90A4AE", "#66BB6A"], alpha=0.85)
    ax.set_title("UCIQE (underwater colour quality)")
    delta = summary["uciqe_enh_mean"] - summary["uciqe_orig_mean"]
    color = "green" if delta >= 0 else "red"
    ax.text(1, summary["uciqe_enh_mean"] + 0.002,
            f"Δ{delta:+.3f}", ha="center", color=color, fontweight="bold")

    plt.suptitle("Phase 2B — Enhancement Benchmark Results", fontsize=13)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Metrics chart saved: {output_path}")


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(summary: dict) -> None:
    print("\n" + "=" * 62)
    print("  PHASE 2B — ENHANCEMENT BENCHMARK REPORT")
    print("=" * 62)
    print(f"  Images tested   : {summary['n_images']}")
    print()
    ssim_status = "PASS ✅" if summary["ssim_pass"] else "FAIL ❌"
    psnr_status = "PASS ✅" if summary["psnr_pass"] else "FAIL ❌"
    uciq_status = "IMPROVED ✅" if summary["uciqe_improved"] else "NO CHANGE ⚠"
    print(f"  SSIM mean       : {summary['ssim_mean']:.4f}  (target ≥ {summary['ssim_target']})  {ssim_status}")
    print(f"  SSIM std        : {summary['ssim_std']:.4f}")
    print(f"  PSNR mean       : {summary['psnr_mean_db']:.2f} dB  (target ≥ {summary['psnr_target_db']} dB)  {psnr_status}")
    print(f"  PSNR std        : {summary['psnr_std_db']:.2f} dB")
    print(f"  UCIQE original  : {summary['uciqe_orig_mean']:.4f}")
    print(f"  UCIQE enhanced  : {summary['uciqe_enh_mean']:.4f}  {uciq_status}")
    print("=" * 62)

    if summary["ssim_pass"] and summary["psnr_pass"]:
        print("\n  All targets met! Proceed to Phase 3 training.")
    else:
        if not summary["ssim_pass"]:
            print(f"\n  WARNING: SSIM {summary['ssim_mean']:.4f} < target 0.80")
            print("  → Consider reducing PYRAMID_LVLS or CLAHE clip limit")
        if not summary["psnr_pass"]:
            print(f"\n  WARNING: PSNR {summary['psnr_mean_db']:.2f} dB < target 15 dB")
            print("  → Consider reducing gamma or sharpening strength")

    print("\n  Next: Phase 3 — YOLOv8 Training")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2B — Enhancement benchmark")
    p.add_argument("--dataset_dir", default="dataset/processed/train/images",
                   help="Directory of images to benchmark")
    p.add_argument("--sample",      type=int, default=20,
                   help="Number of images to sample (default: 20)")
    p.add_argument("--output_dir",  default="results/enhancement",
                   help="Where to save plots and report")
    p.add_argument("--grid_n",      type=int, default=4,
                   help="Images in comparison grid (default: 4)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("╔" + "═" * 60 + "╗")
    print("║   PHASE 2B — ENHANCEMENT BENCHMARK & VALIDATION          ║")
    print("╚" + "═" * 60 + "╝")

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)

    if not dataset_dir.exists():
        print(f"ERROR: {dataset_dir} not found.")
        print("Run Phase 1D first to create dataset/processed/")
        return

    # Collect images
    exts = {".jpg", ".jpeg", ".png"}
    image_paths = [p for p in dataset_dir.rglob("*") if p.suffix.lower() in exts]

    if not image_paths:
        print(f"ERROR: No images found in {dataset_dir}")
        return

    print(f"\nFound {len(image_paths)} images in {dataset_dir}")
    print(f"Sampling {min(args.sample, len(image_paths))} for benchmark...")

    # Run benchmark
    summary = benchmark_sample(image_paths, args.sample)
    if not summary:
        print("ERROR: Benchmark failed — no images could be processed")
        return

    # Print report
    print_report(summary)

    # Save JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull report saved: {report_path}")

    # Save plots
    save_comparison_grid(image_paths, output_dir / "enhancement_comparison.png", n=args.grid_n)
    save_metrics_chart(summary, output_dir / "enhancement_metrics.png")


if __name__ == "__main__":
    main()
