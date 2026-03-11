# models/onnx/

This folder holds the ONNX-exported version of the trained YOLOv8s model.

## Purpose
- Phase 9: Edge deployment (Raspberry Pi, Jetson Nano, mobile)
- Cross-platform inference without PyTorch dependency
- Faster inference via ONNX Runtime / TensorRT

## How to export (run from project root with venv active)

```powershell
marine_venv\Scripts\activate
python src\utils\export_onnx.py
```

Or directly:
```python
from ultralytics import YOLO
model = YOLO('models/weights/yolov8s_marine_best.pt')
model.export(format='onnx', imgsz=640, dynamic=False, simplify=True)
# Output: models/weights/yolov8s_marine_best.onnx
# Move to:  models/onnx/yolov8s_marine.onnx
```

## Expected file
```
models/onnx/
└── yolov8s_marine.onnx     (~22 MB)
```

## Note
`.onnx` files are in `.gitignore` due to size.
Share via Google Drive and link in README, or use Git LFS.