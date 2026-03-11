#!/usr/bin/env python3
"""
PHASE 2A — Ancuti et al. CVPR 2012 Underwater Image Enhancement
3-Class Marine Biodiversity Project
Classes: 0=Butterflyfish, 1=Parrotfish, 2=Angelfish

Based on:
    Ancuti, C., Ancuti, C.O., Haber, T., Bekaert, P. (2012).
    Enhancing Underwater Images and Videos by Fusion. CVPR 2012.

Pipeline:
    Input Image
        ├── Input 1: White Balance (Gray World)
        │       └── CLAHE (clip=2.0, tile=8x8)
        └── Input 2: Gamma Correction (γ=1.2) + Sharpening
                └── CLAHE (clip=2.0, tile=8x8)
    Weight Maps (per input):
        ├── Laplacian Contrast Weight
        ├── Saliency Weight
        └── Saturation Weight
    Laplacian Pyramid Fusion (6 levels)
    Output: Enhanced Image

Usage (standalone):
    python src/enhancement/ancuti_fusion.py \
        --input  dataset/processed/train/images \
        --output dataset/enhanced/train/images  \
        --benchmark

Usage (as module):
    from src.enhancement.ancuti_fusion import enhance
    enhanced = enhance(bgr_image)

Author: Shri Harsan M | M.Tech Data Science | SRM Institute
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

CLAHE_CLIP   = 2.0
CLAHE_TILE   = (8, 8)
PYRAMID_LVLS = 6
GAMMA        = 1.2          # slight brightening for dark underwater scenes
SHARP_KERNEL = np.array([   # unsharp mask kernel
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0],
], dtype=np.float32)


# ── Step 1: Derive two input images ───────────────────────────────────────────

def white_balance_gray_world(bgr: np.ndarray) -> np.ndarray:
    """
    Gray World white balance.
    Scales each channel so that its mean equals the global mean.
    Corrects the blue/green colour cast typical in underwater images.
    """
    result  = bgr.astype(np.float32)
    mean_b  = result[:, :, 0].mean()
    mean_g  = result[:, :, 1].mean()
    mean_r  = result[:, :, 2].mean()
    mean_all = (mean_b + mean_g + mean_r) / 3.0

    result[:, :, 0] = np.clip(result[:, :, 0] * (mean_all / (mean_b + 1e-6)), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (mean_all / (mean_g + 1e-6)), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (mean_all / (mean_r + 1e-6)), 0, 255)
    return result.astype(np.uint8)


def gamma_sharpen(bgr: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """
    Gamma correction + unsharp mask sharpening.
    Brightens dark underwater images and restores edge detail.
    """
    # Gamma correction via LUT (fast)
    inv_gamma = 1.0 / gamma
    lut = (np.arange(256, dtype=np.float32) / 255.0) ** inv_gamma * 255.0
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    corrected = cv2.LUT(bgr, lut)

    # Unsharp mask sharpening
    sharpened = cv2.filter2D(corrected, -1, SHARP_KERNEL)
    return sharpened


def apply_clahe(bgr: np.ndarray,
                clip_limit: float = CLAHE_CLIP,
                tile_size: tuple  = CLAHE_TILE) -> np.ndarray:
    """
    Apply CLAHE to the L channel of LAB colour space.
    Enhances local contrast without over-amplifying noise.
    """
    lab    = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe  = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_eq   = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ── Step 2: Weight maps ───────────────────────────────────────────────────────

def weight_laplacian_contrast(bgr: np.ndarray) -> np.ndarray:
    """
    Laplacian contrast weight.
    High response at edges and fine texture — encourages sharp regions.
    """
    gray     = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    return laplacian


def weight_saliency(bgr: np.ndarray) -> np.ndarray:
    """
    Saliency weight using Itti-style local colour contrast.
    Gaussian blur difference approximates centre-surround saliency.
    """
    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean = cv2.GaussianBlur(lab, (0, 0), sigmaX=28)
    diff = np.sqrt(np.sum((lab - mean) ** 2, axis=2))
    return diff


def weight_saturation(bgr: np.ndarray) -> np.ndarray:
    """
    Saturation weight.
    Measures per-pixel colour vividness — rewards well-saturated regions.
    """
    f   = bgr.astype(np.float32)
    mu  = f.mean(axis=2, keepdims=True)
    sat = np.sqrt(((f - mu) ** 2).mean(axis=2))
    return sat


def compute_weights(bgr: np.ndarray) -> np.ndarray:
    """
    Combine all three weight maps, add small epsilon to avoid zeros,
    and return a single-channel float32 weight map (not yet normalised).
    """
    wl = weight_laplacian_contrast(bgr)
    ws = weight_saliency(bgr)
    wc = weight_saturation(bgr)

    # Normalise each map to [0, 1] independently
    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    combined = norm(wl) + norm(ws) + norm(wc) + 1e-6
    return combined.astype(np.float32)


# ── Step 3: Laplacian Pyramid Fusion ─────────────────────────────────────────

def build_gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    """Build a Gaussian pyramid of 'levels' levels."""
    pyr = [img.astype(np.float32)]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def build_laplacian_pyramid(img: np.ndarray, levels: int) -> list:
    """Build a Laplacian pyramid from the image."""
    gauss = build_gaussian_pyramid(img, levels)
    lpyr  = []
    for i in range(levels - 1):
        size = (gauss[i].shape[1], gauss[i].shape[0])
        up   = cv2.pyrUp(gauss[i + 1], dstsize=size)
        lpyr.append(gauss[i] - up)
    lpyr.append(gauss[-1])   # coarsest level = Gaussian residual
    return lpyr


def pyramid_fusion(inputs: list, weights: list, levels: int = PYRAMID_LVLS) -> np.ndarray:
    """
    Fuse N input images using their weight maps via Laplacian pyramid blending.

    inputs  : list of BGR float32 arrays
    weights : list of single-channel float32 weight maps (unnormalised)
    levels  : pyramid depth
    """
    # Normalise weights so they sum to 1 per pixel
    weight_sum = sum(weights)
    norm_w = [w / (weight_sum + 1e-8) for w in weights]

    # Build pyramids
    lp_inputs = [build_laplacian_pyramid(inp, levels) for inp in inputs]
    gp_weights = [build_gaussian_pyramid(w[:, :, np.newaxis], levels) for w in norm_w]

    # Fuse level by level
    fused_pyr = []
    for lvl in range(levels):
        fused_lvl = None
        for i, lp in enumerate(lp_inputs):
            layer = lp[lvl]                          # (H, W, 3)
            w_map = gp_weights[i][lvl]               # (H, W, 1)
            # Ensure spatial dims match (can differ by ±1 pixel after pyrDown)
            h, ww = layer.shape[:2]
            w_map = cv2.resize(w_map, (ww, h))
            if w_map.ndim == 2:
                w_map = w_map[:, :, np.newaxis]
            contribution = layer * w_map
            fused_lvl = contribution if fused_lvl is None else fused_lvl + contribution
        fused_pyr.append(fused_lvl)

    # Collapse pyramid
    result = fused_pyr[-1]
    for lvl in range(levels - 2, -1, -1):
        h, ww = fused_pyr[lvl].shape[:2]
        result = cv2.pyrUp(result, dstsize=(ww, h)) + fused_pyr[lvl]

    return np.clip(result, 0, 255).astype(np.uint8)


# ── Public API ────────────────────────────────────────────────────────────────

def enhance(bgr: np.ndarray) -> np.ndarray:
    """
    Full Ancuti 2012 enhancement pipeline.

    Parameters
    ----------
    bgr : np.ndarray
        Input BGR image (uint8, H×W×3).

    Returns
    -------
    np.ndarray
        Enhanced BGR image (uint8, H×W×3).
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("Input image is empty or None")

    # ── Derive two inputs ──────────────────────────────────────────────────
    inp1 = apply_clahe(white_balance_gray_world(bgr))
    inp2 = apply_clahe(gamma_sharpen(bgr))

    # ── Weight maps ────────────────────────────────────────────────────────
    w1 = compute_weights(inp1)
    w2 = compute_weights(inp2)

    # ── Pyramid fusion ─────────────────────────────────────────────────────
    result = pyramid_fusion(
        inputs  = [inp1.astype(np.float32), inp2.astype(np.float32)],
        weights = [w1, w2],
        levels  = PYRAMID_LVLS,
    )
    return result


# ── Batch processing ──────────────────────────────────────────────────────────

def enhance_directory(input_dir: Path, output_dir: Path,
                      extensions: tuple = (".jpg", ".jpeg", ".png")) -> dict:
    """
    Enhance all images in input_dir and write results to output_dir.

    Returns a dict with counts: {processed, skipped, errors}
    """
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in extensions]

    counts = {"processed": 0, "skipped": 0, "errors": 0}

    for img_path in tqdm(image_paths, desc=f"Enhancing {input_dir.name}"):
        out_path = output_dir / img_path.name
        if out_path.exists():
            counts["skipped"] += 1
            continue
        try:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                counts["errors"] += 1
                continue
            enhanced = enhance(bgr)
            cv2.imwrite(str(out_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            counts["processed"] += 1
        except Exception as e:
            print(f"  ERROR {img_path.name}: {e}")
            counts["errors"] += 1

    return counts


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ancuti 2012 underwater image enhancement"
    )
    p.add_argument("--input",     required=True,  help="Input image or directory")
    p.add_argument("--output",    required=True,  help="Output image or directory")
    p.add_argument("--benchmark", action="store_true",
                   help="Compute SSIM/PSNR on a sample and save comparison image")
    p.add_argument("--sample",    type=int, default=5,
                   help="Number of sample images for benchmark (default: 5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    inp  = Path(args.input)
    out  = Path(args.output)

    print("╔" + "═" * 60 + "╗")
    print("║     PHASE 2A — ANCUTI UNDERWATER ENHANCEMENT              ║")
    print("╚" + "═" * 60 + "╝")

    if inp.is_file():
        # Single image mode
        bgr = cv2.imread(str(inp))
        if bgr is None:
            print(f"ERROR: Could not read {inp}")
            return
        enhanced = enhance(bgr)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Enhanced image saved: {out}")

    elif inp.is_dir():
        # Batch mode
        counts = enhance_directory(inp, out)
        print(f"\nDone! Processed={counts['processed']}  "
              f"Skipped={counts['skipped']}  Errors={counts['errors']}")
    else:
        print(f"ERROR: {inp} does not exist")
        return

    if args.benchmark:
        print("\nBenchmark requested — run benchmark_enhancers.py for full metrics")

    print("\nNext: python src/enhancement/benchmark_enhancers.py")


if __name__ == "__main__":
    main()
