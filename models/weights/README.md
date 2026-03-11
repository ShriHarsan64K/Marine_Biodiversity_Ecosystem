# models/weights/

Trained YOLOv8s model weights for marine fish detection.

## Expected file
```
models/weights/
└── yolov8s_marine_best.pt     (~22 MB — add to .gitignore or Git LFS)
```

## Copy from training output
```powershell
Copy-Item "results\detection\yolov8s_marine\weights\best.pt" `
          "models\weights\yolov8s_marine_best.pt"
```

## Model Details
| Property | Value |
|----------|-------|
| Architecture | YOLOv8s |
| Input size | 640 × 640 |
| Classes | 3 (Butterflyfish, Parrotfish, Angelfish) |
| mAP@0.5 | 95.1% |
| mAP@0.5:0.95 | ~68% |
| Parameters | ~11.2M |
| Training epochs | 100 |
| GPU | RTX 3050 6GB |

## Note
`.pt` files are in `.gitignore` due to size (~22 MB).
Share via Google Drive and add the link to README.