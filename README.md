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

## Verification & Execution

### Run Automated Tests
```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```
Output:
```
============================= 24 passed in 1.20s ==============================
```

### Run Reproducible Benchmark
```powershell
# Benchmark detector and pipeline with synthetic frames
.\.venv\Scripts\python.exe scripts/run_benchmark.py --detector mock --iterations 50

# Benchmark OpenCV motion detector
.\.venv\Scripts\python.exe scripts/run_benchmark.py --detector motion --iterations 30

# Benchmark with real Egyptian vehicle image
.\.venv\Scripts\python.exe scripts/run_benchmark.py --detector mock --image "C:\Users\MARWAN\plate_project\images_raw\EALPR-master\EALPR Vechicles dataset\Vehicles\0001.jpg"
```

### Process Image or Video Feed
```powershell
.\.venv\Scripts\python.exe scripts/process_media.py "C:\Users\MARWAN\plate_project\images_raw\EALPR-master\EALPR Vechicles dataset\Vehicles\0001.jpg"
```
