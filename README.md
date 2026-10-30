# Vehicle Intelligence & Egyptian License Plate Recognition System

A high-performance, modular computer vision foundation designed for fixed CCTV cameras at gates and parking facilities. Initially targeted at Egyptian vehicles and Egyptian license plates with progressive identification of vehicle attributes, stickers, and permits.

---

## Architecture Overview

```
                      +-----------------------------------------+
                      |       CCTV RTSP Stream / Video / Image   |
                      +-----------------------------------------+
                                           |
                                           v
                             +--------------------------+
                             |    MediaLoader (OpenCV)  |
                             +--------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
        +-----------------------+                     +-----------------------+
        |  BaseVehicleDetector  |                     |  BaseVehicleTracker   |
        | (YOLOv10 / Mock / MOG)|                     | (Centroid / ByteTrack)|
        +-----------------------+                     +-----------------------+
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
                             +--------------------------+
                             |    BasePlateDetector     |
                             |   (Localized ROI crop)   |
                             +--------------------------+
                                           |
                                           v
                             +--------------------------+
                             |       BasePlateOCR       |
                             | (Arabic/Latin + Digits)  |
                             +--------------------------+
                                           |
                                           v
                             +--------------------------+
                             | EvidenceAggregator (FIFO)|
                             |  Multi-Frame Consensus   |
                             +--------------------------+
                                           |
                         +-----------------+-----------------+
                         |                                   |
                         v                                   v
             +-----------------------+           +-----------------------+
             | VehicleSessionSummary |           |    PrivacyRedactor    |
             |   (Structured Data)   |           |  (Blurring & Pruning) |
             +-----------------------+           +-----------------------+
```

### Modular Interfaces
- `BaseVehicleDetector`: Pluggable vehicle detection interface (`VehicleDetection`).
- `BasePlateDetector`: Pluggable Egyptian license plate detection interface (`PlateDetection`).
- `BasePlateOCR`: Replaceable OCR interface with bilingual Arabic letters, numbers, and Latin transliteration (`PlateTextResult`).
- `BaseVehicleTracker`: Multi-frame object tracking assigning persistent `track_id` (`TrackedVehicle`).
- `BaseEvidenceAggregator`: Multi-frame consensus accumulator, best-crop selector, and `needs_review` governor.
- `BaseVehicleAttributeClassifier`: Secondary classifier for Make, Model, Year range, Primary Color, and Body Type.
- `BaseStickerDetector`: Identifies parking tags, inspection decals, and commercial stickers.
- `BasePermitReader`: Windshield gate and facility access permit parsing.

---

## Hardware Targets & Sizing Analysis

| Dimension | Host Development System | Target Deployment System |
|---|---|---|
| **GPU** | NVIDIA GeForce RTX 4060 (Ada Lovelace) | NVIDIA GeForce RTX 3060 (Ampere) |
| **VRAM** | 8 GB (8,188 MB) | **12 GB (12,288 MB)** |
| **Streams Target** | Single stream / Development tests | **8–16 Gate Camera Streams** |
| **Target FPS per Gate** | 10–25 FPS | **6–10 FPS per camera** (sufficient for gates) |
| **Target VRAM Budget** | ~3.5 GB | **~7.5 GB total** (leaving 4.5 GB safety headroom) |

### Multi-Stream Scaling Strategy
1. **Gate Ingestion Downsampling**: Gate cameras do not require 30 FPS inference. Vehicles moving at 10–30 km/h are in the gate trigger zone for 2–5 seconds; sampling at 6–8 FPS provides 15–40 frames per vehicle.
2. **Conditional Attribute Execution**: Heavy OCR and Attribute classification models are **only** executed on high-quality keyframes selected by the tracking accumulator, never on every frame of every camera.
3. **Hardware NVDEC Decoding**: Hardware video decoding offloads CPU cores.

---

## Candidate Model Families Evaluation

| Task | Candidate | License | Latency (ms) | Strengths | Trade-offs |
|---|---|---|---|---|---|
| **Vehicle Detection** | **YOLOv10-M** *(Recommended)* | **Apache 2.0** | ~6–9 ms | NMS-free, ultra-fast, commercial friendly | Requires export to ONNX / TensorRT |
| Vehicle Detection | RT-DETR-R18/R50 | Apache 2.0 | ~10–16 ms | Transformer-based, high small-object recall | Higher VRAM footprint (~1.8x) |
| Vehicle Detection | YOLOv11-M | AGPL-3.0 | ~5–8 ms | State-of-the-art accuracy | AGPL viral license requires commercial fee |
| **Plate OCR** | **PaddleOCR (PP-OCRv4)** *(Recommended)* | **Apache 2.0** | ~8–14 ms | Native Arabic character set, lightweight | Preprocessing required for plate slant |
| Plate OCR | EasyOCR | Apache 2.0 | ~25–40 ms | Strong multilingual Arabic support | Slower inference speed on batch streams |
| Plate OCR | Custom CRNN (Trained on EALPR) | Apache 2.0 | **~3–5 ms** | Tailored to Egyptian plate fonts & colors | Requires supervised training pipeline |

---

## Egyptian License Plate Structure

Egyptian vehicle plates have distinct color banners indicating vehicle usage:
- **Light Blue**: Private Vehicles (ملاكي)
- **Orange**: Taxis (أجرة)
- **Red / Green**: Commercial / Transport / Buses (نقل / تجاري)
- **Dark Blue**: Police (شرطة)
- **Green**: Diplomatic Corps (هيئات دبلوماسية)
- **Yellow**: Customs (جمارك)

Our domain model standardizes transliteration from Arabic characters (e.g. `س ع د`) to Latin characters (`S E D`) and converts Eastern Arabic digits (`١ ٢ ٣ ٤`) to standardized digits (`1234`).

---

## Privacy & Data Retention Compliance

In accordance with the **Egyptian Data Protection Law (Law No. 151 of 2020)** and CCTV privacy best practices:
- **Face Blurring**: Automated blurring of driver and bystander faces before archival.
- **Background Dimming**: Masking non-vehicle regions in CCTV frames to isolate the vehicle ROI.
- **Retention Policies**: Automated purging of raw video recordings after a configurable threshold (default: 30 days) and pseudonymization of plate metadata after 7 days.
- **Plate Text Anonymization**: Cryptographic salted SHA-256 hashing for statistical aggregation.

---

## Installation & Setup

### 1. Prerequisites
- Python 3.11 (64-bit)
- NVIDIA GPU Driver (CUDA 12.x / 13.x compatible)
- Git

### 2. Environment Setup
```powershell
# Clone / enter repository
cd "C:\Users\MARWAN\computer vision"

# Create Python 3.11 virtual environment
py -3.11 -m venv .venv

# Activate environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install core dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Optional PyTorch with CUDA Acceleration
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Verification & Full Reproduction Pipeline

### 1. Automated Headline Reproduction Script
The repository provides an end-to-end PowerShell reproduction script that verifies manifest integrity, runs all unit/domain/golden regression tests, and reproduces headline detection and OCR metrics:

```powershell
# Run full reproduction pipeline
powershell -ExecutionPolicy Bypass -File scripts/reproduce_all.ps1
```

### 2. Manual Reproduction Commands
```powershell
# 1. Run all regression test suites (including 20 golden samples)
.\.venv\Scripts\python.exe -m pytest -v

# 2. Evaluate Plate Detector on Cleaned Test Split
.\.venv\Scripts\python.exe scripts/evaluate_plate_detector.py `
  --weights artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt `
  --dataset-dir artifacts/ealpr_yolo_dataset_cleaned `
  --gt-dir artifacts/splits/cleaned_ground_truth `
  --output-dir artifacts/evaluation_cleaned_verified `
  --conf 0.25

# 3. Evaluate End-to-End ALPR and OCR (Separated Part A, B, and C)
.\.venv\Scripts\python.exe scripts/audit_end_to_end_and_ocr.py

# 4. Generate 27-Class Character Confusion Matrix
.\.venv\Scripts\python.exe scripts/generate_character_confusion_matrix.py

# 5. Run Multi-Batch FP32 vs FP16 Latency & Throughput Benchmark
.\.venv\Scripts\python.exe scripts/benchmark_multi_batch_throughput.py
```

---

## Command-Line Interface (`scripts/alpr_cli.py`)

The ALPR CLI supports single images, image directories, video files, and simulated gate replay:

```powershell
# Process a single vehicle image
.\.venv\Scripts\python.exe scripts/alpr_cli.py --input path/to/vehicle.jpg --output-dir artifacts/cli_output

# Process an entire folder of vehicle images (outputs crops, JSONs, and summary.csv)
.\.venv\Scripts\python.exe scripts/alpr_cli.py --input path/to/folder --output-dir artifacts/cli_folder_output

# Process a video stream sampled at 2.0 FPS
.\.venv\Scripts\python.exe scripts/alpr_cli.py --input path/to/video.mp4 --fps 2.0 --output-dir artifacts/cli_video_output

# Simulated Gate Replay Mode (tracks vehicles, selects best frame per transit, emits JSON lines)
.\.venv\Scripts\python.exe scripts/alpr_cli.py --input path/to/gate_cctv.mp4 --fps 2.0 --gate-replay --output-dir artifacts/gate_transit_output
```

---

## Own-Data Readiness (Gate Video Tools)

### Ingest Gate Video & Drop Near-Duplicates
```powershell
# Ingests video, drops near-duplicates via dHash (dist <= 3), and generates YOLO pre-labels
.\.venv\Scripts\python.exe scripts/ingest_gate_footage.py --video path/to/gate.mp4 --output-dir data/gate_raw/ --fps 2.0
```

Folder layout created under `data/gate_raw/`:
```
data/gate_raw/
├── images/
│   ├── gate_frame_00001.jpg
│   └── gate_frame_00002.jpg
├── labels/
│   ├── gate_frame_00001.txt   # YOLO format: 0 xc yc w h
│   └── gate_frame_00002.txt
└── gate_ingest_manifest.json
```

### Evaluate on Labeled Gate Frames with Environmental Slicing
```powershell
.\.venv\Scripts\python.exe scripts/evaluate_on_gate_frames.py `
  --dataset-dir data/gate_raw `
  --tags data/gate_raw/conditions.json `
  --output-dir artifacts/evaluation_gate
```

---

## REST API Service

A thin REST API is provided in `vehicle_intelligence/api/app.py`:
- `GET /health`: Health status.
- `POST /v1/read-plate`: Uploads an image file and returns structured `ALPRPlateResult` JSON.
- Built-in `/docs` Swagger documentation page.

To start the server:
```powershell
.\.venv\Scripts\python.exe vehicle_intelligence/api/app.py --host 0.0.0.0 --port 8000
```

---

## Inference Configuration & NMS Standards

The baseline end-to-end ALPR pipeline (`EndToEndALPRPipeline`) operates under the following documented configuration:
- **Plate Detection**: `conf = 0.25`, `iou = 0.70`, `imgsz = 640`, `agnostic_nms = False`
- **Character Recognition (27 Classes)**: `conf = 0.25`, `iou = 0.70`, `imgsz = 320`, `agnostic_nms = False`
- **Default `agnostic_nms = False` Policy**:
  Class-aware Non-Maximum Suppression (`agnostic_nms=False`) is the canonical, frozen default. It ensures that spatially proximate or overlapping character detections belonging to different classes (e.g., adjacent digits and closely kerned Arabic letter strokes) are not suppressed across distinct classes. This setting reproduces the audited 93.38% (282/302) end-to-end exact match performance.


