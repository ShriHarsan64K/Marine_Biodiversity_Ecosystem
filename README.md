# 🪸 Marine Biodiversity Ecosystem Assessment

### Real-Time Fish Detection, Tracking & Reef Health Monitoring using YOLOv8

[![Python](https://img.shields.io/badge/Python-3.11.2-blue?logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8s-Ultralytics-purple)](https://github.com/ultralytics/ultralytics)
[![mAP](https://img.shields.io/badge/mAP@0.5-95.1%25-brightgreen)](results/detection/)
[![FPS](https://img.shields.io/badge/FPS-59.3-orange)](results/detection/)
[![MHI](https://img.shields.io/badge/MHI-70.59%2F100-teal)](results/biodiversity_health/)
[![GPU](https://img.shields.io/badge/GPU-RTX%203050%206GB-green)](https://www.nvidia.com/)
[![Institute](https://img.shields.io/badge/SRM%20Institute-M.Tech%20Data%20Science-red)](https://www.srmist.edu.in/)

> **M.Tech Research Project** — Shri Harsan M | Department of Data Science | SRM Institute of Science and Technology, Chennai

---

## 🎯 Overview

This project presents an **end-to-end real-time pipeline** for marine biodiversity monitoring using deep learning. The system detects, tracks, and analyses three reef indicator fish species — Butterflyfish, Parrotfish, and Angelfish — from underwater video footage, computing a novel **Marine Health Index (MHI)** score that quantifies coral reef ecosystem health in real time.

The pipeline runs entirely on a **consumer-grade GPU (RTX 3050 6GB)** at **59.3 FPS**, making it practical for field deployment without specialised hardware. The live monitoring dashboard streams ecological metrics, biodiversity indices, trophic structure, and Detection Time Ratio (DTR) in a browser-based interface.

---

## 📊 Key Results

| Metric | Value | Details |
|--------|-------|---------|
| **mAP@0.5** | **95.1%** | YOLOv8s on 364-image test set |
| **Precision** | 93.8% | Across all 3 classes |
| **Recall** | 92.8% | Across all 3 classes |
| **FPS** | **59.3** | RTX 3050 6GB, 1920×1080 |
| **MHI Score** | **70.59 / 100** | Grade: GOOD — HEALTHY |
| **Shannon H′** | 0.999 | Excellent weighted diversity |
| **Pielou J′** | 0.909 | Even species distribution |
| **SSIM** (Enhancement) | 0.847 | Ancuti underwater enhancement |
| **PSNR** (Enhancement) | 17.7 dB | Phase 2 image quality |
| **Dataset Size** | 5,201 images | 3 species, augmented |

---

## 🐠 3 Target Species

| Class | Species | Ecological Role | Weight (w) | Emoji |
|-------|---------|-----------------|------------|-------|
| **Class 0** | Butterflyfish | Coral health indicator | **2.0** | 🦋 |
| **Class 1** | Parrotfish | Algae control indicator | **1.6** | 🦜 |
| **Class 2** | Angelfish | Reef structure indicator | **1.1** | 👼 |

> Weights reflect ecological importance in the MHI formula: `MHI = 0.30×H′_w + 0.25×Trophic + 0.20×Apex + 0.15×Presence + 0.10×Evenness`

---

## 🔬 6 Novel Contributions

1. **Weighted Marine Health Index (MHI)** — First composite 0–100 ecological health score combining weighted Shannon diversity, trophic balance, apex predator presence, indicator species presence, and evenness. Inspired by AGRRA/Reef Check belt transect methodology.

2. **Cumulative Survey Tracking** — Persistent cross-frame species counting that mirrors AGRRA underwater survey belt transect protocol, enabling quantitative biodiversity assessment from video.

3. **Detection Time Ratio (DTR)** — Novel video-only metric: `DTR = frame_of_first_complete_species_set / total_frames`. Rewards early detection of all species. Combined score = `0.70 × MHI + 0.30 × Time_Score`.

4. **Real-Time Trophic Cascade Detection** — Automated identification of herbivore/corallivore trophic imbalance with cascade failure alerts, enabling early warning of reef degradation.

5. **Degradation Robustness Testing** — Systematic evaluation of MHI stability under simulated turbidity, blur, and noise degradation conditions. Demonstrates graceful performance degradation.

6. **Consumer-GPU End-to-End Pipeline** — Complete detection → tracking → health assessment pipeline running at 59.3 FPS on RTX 3050 6GB, making real-time reef monitoring accessible without specialised hardware.

---

## 📁 Project Structure

```
marine_biodiversity_ecosystem/
├── configs/                        # YOLOv8 training YAML configs
├── data/
│   └── videos/                     # Test videos (.gitignore — not in repo)
├── dataset/
│   ├── augmented_train/            # Augmented training images + labels
│   ├── splits/                     # Train / val / test splits
│   │   ├── train/  val/  test/
│   └── standardized/               # Final merged dataset (5,201 images)
│       ├── train/  val/  test/
├── docs/                           # Project documentation (.docx files)
├── logs/                           # Training logs
├── models/
│   ├── weights/                    # best.pt — trained YOLOv8s weights
│   └── onnx/                       # ONNX export (Phase 9 edge deployment)
├── notebooks/                      # EDA and analysis notebooks
├── results/
│   ├── biodiversity_health/        # mhi_report.json, indices
│   ├── dashboard/                  # Live HTML dashboard + state files
│   ├── detection/
│   │   └── yolov8s_marine/         # Training run outputs (weights, plots)
│   ├── enhancement/                # Ancuti Phase 2 enhanced images
│   ├── preprocessing/              # Data cleaning outputs
│   ├── tracking/                   # Phase 4 ByteTrack/BoT-SORT outputs
│   └── validation/                 # Degradation robustness plots
├── src/
│   ├── dashboard/                  # phase6_tracker_dtr.py, phase6_image_test.py
│   ├── detection/                  # YOLOv8 training scripts
│   ├── enhancement/                # Ancuti underwater enhancement
│   ├── evaluation/                 # Test-set evaluation scripts
│   ├── health_index/               # MHI computation (mhi_calculator.py)
│   ├── tracking/                   # ByteTrack / BoT-SORT integration
│   └── utils/                      # Shared utilities
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11.2
- CUDA-compatible GPU (tested: NVIDIA RTX 3050 6GB)
- CUDA 11.8+ and cuDNN

### 1. Clone the repository
```powershell
git clone https://github.com/YOUR_USERNAME/marine_biodiversity_ecosystem.git
cd marine_biodiversity_ecosystem
```

### 2. Create virtual environment
```powershell
python -m venv marine_venv
marine_venv\Scripts\activate        # Windows
# source marine_venv/bin/activate   # Linux/Mac
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Option A — Test on Image Folder
```powershell
# Window 1: Start dashboard server
cd results\dashboard
python -m http.server 8080

# Window 2: Run image test
marine_venv\Scripts\activate
python src\dashboard\phase6_image_test.py
```

### Option B — Test on Video (with DTR)
```powershell
# Window 1: Start dashboard server
cd results\dashboard
python -m http.server 8080

# Window 2: Run video tracker
marine_venv\Scripts\activate
python src\dashboard\phase6_tracker_dtr.py --source data\videos\reef_test_1.mp4
```

### Open Dashboard
```
http://localhost:8080/phase6_dashboard_live.html
```

### Tracker Controls (press in OpenCV window)
| Key | Action |
|-----|--------|
| `Q` | Quit tracker |
| `P` | Pause / Resume |
| `R` | Reset counters |
| `S` | Save screenshot |

---

## 📈 Phases Completed (6 of 12)

- [x] **Phase 1** — Dataset collection & preprocessing (5,201 images, 3 species)
- [x] **Phase 2** — Underwater image enhancement (Ancuti method, SSIM=0.847)
- [x] **Phase 3** — YOLOv8s detection training (mAP@0.5=95.1%, 59.3 FPS)
- [x] **Phase 4** — Multi-object tracking (ByteTrack/BoT-SORT, 490 tracks)
- [x] **Phase 5** — Marine Health Index computation (MHI=70.59/100, GOOD)
- [x] **Phase 6** — Real-time live dashboard with DTR (browser-based, 1s refresh)
- [ ] Phase 7 — Coral segmentation *(future work)*
- [ ] Phase 8 — Temporal LSTM analysis *(future work)*
- [ ] Phase 9 — Edge deployment / ONNX *(planned)*
- [ ] Phase 10 — REST API *(planned)*
- [ ] Phase 11 — Field pilot study *(future work)*
- [ ] Phase 12 — IEEE publication *(in progress)*

---

## 🧮 MHI Formula

```
MHI = (0.30 × H′_weighted) + (0.25 × Trophic_Balance) +
      (0.20 × Apex_Predator) + (0.15 × Indicator_Presence) +
      (0.10 × Pielou_Evenness)

Combined Score (video) = 0.70 × MHI + 0.30 × Time_Score
Time_Score = (1 - DTR) × 100
DTR = frame_of_first_complete_species_set / total_frames
```

| MHI Range | Grade | Status |
|-----------|-------|--------|
| 80–100 | Excellent | 🟢 Pristine |
| 60–79 | Good | 🟢 Healthy |
| 40–59 | Fair | 🟡 Moderate stress |
| 20–39 | Poor | 🔴 Degraded |
| 0–19 | Critical | 🔴 Severely degraded |

---

## 📚 Key References

1. Qin et al. (2024) — YOLOv8-FASG for underwater fish detection, mAP 96.4% — *IEEE Access*
2. Williams et al. (2023) — Real-time reef health scoring as open problem — *Frontiers in Marine Science*
3. Raj et al. (2022) — ReefVision offline analysis system — *Ecological Informatics*
4. Du et al. (2023) — BoT-SORT multi-object tracking, MOTA 80.5 — *IEEE TMM*
5. Zhang et al. (2022) — ByteTrack — *ECCV*
6. Islam et al. (2021) — FUnIE-GAN underwater enhancement — *IEEE RAL*

> Full reference list (18 papers, 2021–2025) in `docs/IEEE_Reference_Papers.docx`

---

## ⚠️ Limitations

- Validated on 3 indicator species only (Butterflyfish, Parrotfish, Angelfish)
- Field validation against concurrent manual diver surveys outside current scope
- DTR metric applicable to video mode only; not computed for static image sets
- Model trained primarily on Indo-Pacific reef imagery

---

## 👤 Author

**Shri Harsan M**
M.Tech Data Science
SRM Institute of Science and Technology, Chennai
Department of Computing Technologies

---

## 📄 License

This project is submitted as an M.Tech research thesis. Code is available for academic use.

---

*Marine Biodiversity Ecosystem Assessment · SRM Institute · 2025*