# Model Checkpoint Manifest & Provenance Guide

This document defines the cryptographic specification, expected storage paths, licensing status, and acquisition procedures for all machine learning models used in the Egyptian Vehicle Intelligence System.

---

## 1. Security & Licensing Policy

> [!IMPORTANT]
> **NEVER commit binary model checkpoints (`.pt`, `.pth`, `.onnx`, `.engine`) to public Git repositories.**
> Model weights are excluded via `.gitignore` to protect intellectual property, adhere to open-source repository size limits, and prevent accidental distribution of fine-tuned weights without explicit corporate authorization.

---

## 2. Checkpoint Inventory & Cryptographic Manifest

The system enforces cryptographic SHA-256 verification to ensure that inference behavior is bit-for-bit identical across multiple deployment workstations:

| Model Identifier | Architecture | Expected Path | SHA-256 Checksum | Size | Workflow Classification | License |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **`ealpr_cleaned_plate_detector`** | YOLOv8n (CNN, 3.2M params) | `artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt` | `1d9ab549c626ee1b708b3439ad63320515c6de105283a5c980b3c5952337b279` | 6.5 MB | **Required for normal ALPR** | AGPL-3.0 |
| **`ealpr_characters_detector`** | YOLOv8n (CNN, 3.2M params, 27 cls) | `artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt` | `875b76e18b1df480829f84809e1f26d2c38d2c92c7f66d3bd85492db14488319` | 6.5 MB | **Required for normal ALPR** | AGPL-3.0 |
| **`yolov8m_finetuned_vehicle_detector`** | YOLOv8m (CNN, 25.9M params) | `artifacts/missions/mission_013/checkpoints/yolov8m_finetuned_best.pt` | `b2e462fe4b06f1aa5479d615445c2246941d6ea1a91ebdb235e6b056788ca03d` | 52.0 MB | **Required for vehicle detection** | AGPL-3.0 |
| **`yolov8m_pretrained`** | YOLOv8m (CNN, 25.9M params) | `yolov8m.pt` | `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5` | 52.1 MB | **Required for vehicle detection** | AGPL-3.0 |
| **`rtdetr_l_finetuned_vehicle_detector`** | RT-DETR-L (Transformer, 32.8M params) | `artifacts/missions/mission_013/checkpoints/rtdetr_l_finetuned_best.pt` | `47f0cc1a288cc2d2f630f80841caa0e90b92eda025dd570424d4402836975613` | 66.2 MB | **Optional experimental** | Apache-2.0 |
| **`rtdetr_l_pretrained`** | RT-DETR-L (Transformer, 32.9M params) | `rtdetr-l.pt` | `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f` | 66.5 MB | **Optional experimental** | Apache-2.0 |
| **`yolov8n_pretrained`** | YOLOv8n (CNN, 3.2M params) | `yolov8n.pt` | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | 6.5 MB | **Obsolete or unused** | AGPL-3.0 |
| **`yolo11n_pretrained`** | YOLO11n (CNN, 2.6M params) | `yolo11n.pt` | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` | 5.6 MB | **Obsolete or unused** | AGPL-3.0 |
| **`yolov8s_pretrained`** | YOLOv8s (CNN, 11.2M params) | `yolov8s.pt` | `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a` | 22.6 MB | **Obsolete or unused** | AGPL-3.0 |

---

## 3. Obtaining & Installing Checkpoints on a New Machine

When cloning this repository on a new machine, follow these steps to place checkpoints in their expected directories:

### A. Core ALPR Production Checkpoints (Required for Golden Tests & CLI)

The End-to-End ALPR Pipeline and Golden Test suite require two trained models:

1. **Plate Detector:** Place `best.pt` into:
   ```powershell
   New-Item -ItemType Directory -Force -Path "artifacts\runs\detect\ealpr_cleaned_yolov8n\weights"
   Copy-Item "path\to\downloaded\ealpr_cleaned_best.pt" "artifacts\runs\detect\ealpr_cleaned_yolov8n\weights\best.pt"
   ```
2. **Character OCR Detector:** Place `best.pt` into:
   ```powershell
   New-Item -ItemType Directory -Force -Path "artifacts\runs\detect_chars\ealpr_characters_yolov8n\weights"
   Copy-Item "path\to\downloaded\ealpr_characters_best.pt" "artifacts\runs\detect_chars\ealpr_characters_yolov8n\weights\best.pt"
   ```

*Alternatively, you may pass explicit paths via CLI arguments:*
```powershell
python scripts/alpr_cli.py --input tests/golden/images/0010.jpg `
    --detector-weights "path/to/plate_model.pt" `
    --char-weights "path/to/char_model.pt"
```

### B. Domain-Adapted Vehicle Checkpoints (Mission 013 Track B)

To reproduce Mission 013/013A vehicle detection benchmarks:
```powershell
New-Item -ItemType Directory -Force -Path "artifacts\missions\mission_013\checkpoints"
Copy-Item "path\to\yolov8m_finetuned_best.pt" "artifacts\missions\mission_013\checkpoints\yolov8m_finetuned_best.pt"
```

### C. Upstream Pretrained COCO Baselines (Auto-Downloadable)

Pretrained COCO baseline weights can be downloaded directly from official upstream releases:

```powershell
# Download Pretrained YOLOv8m (COCO)
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt" -OutFile "yolov8m.pt"

# Download Pretrained RT-DETR-L (COCO)
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-l.pt" -OutFile "rtdetr-l.pt"

# Download Pretrained YOLOv8n (COCO)
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt" -OutFile "yolov8n.pt"
```

---

## 4. Checkpoint Verification Command

To verify that all checkpoints are correctly installed and match their cryptographic hashes:

```powershell
python scripts/verify_portability.py
```

If any required checkpoints are missing, the verification script will print the missing file path, required SHA-256 hash, and actionable instructions.
