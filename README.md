# Vehicle Intelligence & Egyptian License Plate Recognition System

A high-performance, modular computer vision foundation designed for fixed CCTV cameras at gates, weigh stations, and parking facilities. Engineered for Egyptian vehicles and Egyptian license plates with progressive identification of vehicle attributes, trailer types, and multi-frame plate consensus.

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
        | (YOLOv8m / RT-DETR)   |                     | (Centroid / ByteTrack)|
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

### Core Modular Components
- `BaseVehicleDetector`: Pluggable vehicle detection interface supporting fine-tuned domain models (`VehicleDetection`).
- `BasePlateDetector`: Pluggable Egyptian license plate detection interface (`PlateDetection`).
- `BasePlateOCR`: Replaceable OCR interface with bilingual Arabic letters, numbers, and Latin transliteration (`PlateTextResult`).
- `BaseVehicleTracker`: Multi-frame object tracking assigning persistent `track_id` (`TrackedVehicle`).
- `BaseEvidenceAggregator`: Multi-frame consensus accumulator, best-crop selector, and `needs_review` governor.
- `PrivacyRedactor`: Privacy-preserving facial and background redactor per Egyptian Data Protection Law (Law No. 151 of 2020).

---

## System Requirements

| Specification | Recommended GPU Profile | CPU Fallback Profile |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 (64-bit) / Ubuntu 22.04+ | Windows 11 (64-bit) / Ubuntu 22.04+ |
| **Python** | **Python 3.11 (64-bit)** (3.10+ supported) | **Python 3.11 (64-bit)** (3.10+ supported) |
| **GPU / Acceleration** | NVIDIA GeForce RTX 3060/4060 or better | Not required (CPU inference supported) |
| **VRAM / RAM** | $\ge 8\text{ GB}$ VRAM / 16 GB RAM | 16 GB RAM |
| **CUDA Driver** | CUDA 12.4+ / NVIDIA Driver 550+ | N/A |
| **Disk Space** | $\sim 5\text{ GB}$ (Environment + weights) | $\sim 3\text{ GB}$ (Environment + weights) |

---

## Quickstart & Installation (Windows 11)

### Option A: Automated Installation (Recommended)

Run the included automated PowerShell setup script from the root of the repository:

```powershell
# Open Windows PowerShell as Administrator or standard user
Set-Location "path\to\cloned\repository"

# Run automated setup (auto-detects NVIDIA GPU or CPU fallback)
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

*Optional script flags:*
- `-Mode GPU`: Force NVIDIA GPU (CUDA 12.4) PyTorch wheel installation.
- `-Mode CPU`: Force CPU-only PyTorch wheel installation.
- `-SkipTests`: Skip post-installation smoke tests.

---

### Option B: Manual Installation

#### 1. Create Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

#### 2. Install PyTorch with Hardware Acceleration

**For NVIDIA GPU (CUDA 12.4):**
```powershell
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**For CPU-Only Systems:**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

#### 3. Install Package in Editable Mode
```powershell
pip install -e . --no-deps
```

#### 4. Configure Local Environment Variables
```powershell
Copy-Item .env.example .env
```

---

## Verification & Health Check

### 1. Run Portability & Environment Audit
Verify that Python, dependencies, compute devices, OpenCV Unicode reading, and model checkpoints are properly configured:

```powershell
python scripts/verify_portability.py
```

Expected output:
```
[*] Python Runtime:       [PASS] 3.11.x (64-bit)
[*] Dependencies:         [PASS] 11 packages installed, 0 missing
[*] Hardware Compute:     [PASS] Mode: GPU_ACCELERATED (or CPU_FALLBACK)
[*] OpenCV Unicode I/O:   [PASS] Arabic path reading supported
[*] System Configuration: [PASS] Config valid
[*] Checkpoints Manifest: [PASS] Tracked models verified
[*] Golden Test Set:      [PASS] 20 regression samples verified
================================================================================
FINAL PORTABILITY VERDICT: PORTABLE
================================================================================
```

### 2. Verify GPU Acceleration
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU Only\"}')"
```

---

## Local Data & Footage Configuration

The repository uses relative paths by default and supports external storage drives via `.env`:

1. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Edit `.env` to specify your local CCTV footage or dataset directories:
   ```ini
   # Local path to gate CCTV captures
   GATE_DIR=C:/Data/TruckEntry
   # DATA_DIR=data
   # WEIGHTS_DIR=weights
   ```
3. If `GATE_DIR` is not set or not mounted, standalone ALPR inference, golden regression tests, and single-image CLI runs execute cleanly without requiring external footage.

---

## Model Checkpoints & Management

Model weights are binary files excluded from Git via `.gitignore`. Refer to [CHECKPOINTS.md](CHECKPOINTS.md) for full cryptographic SHA-256 hashes, architectures, and provenance details.

### Workflow Classification & Checkpoint Mapping:

- **Required for Normal ALPR (Inference, Golden Tests, REST API):**
  - Plate Detector (YOLOv8n): `artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt` (SHA-256: `1d9ab549...`)
  - Character OCR Detector (YOLOv8n 27-class): `artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt` (SHA-256: `875b76e1...`)
- **Required for Vehicle Detection (Gate CCTV Ingress & Tracking):**
  - Fine-Tuned Heavy Vehicle Detector (YOLOv8m): `artifacts/missions/mission_013/checkpoints/yolov8m_finetuned_best.pt` (SHA-256: `b2e462fe...`)
  - Pretrained Baseline (COCO YOLOv8m): `yolov8m.pt` (SHA-256: `5d4a90cd...`)
- **Optional Experimental (Vision Transformer Evaluation):**
  - Fine-Tuned RT-DETR-L: `artifacts/missions/mission_013/checkpoints/rtdetr_l_finetuned_best.pt` (SHA-256: `47f0cc1a...`)
  - Pretrained RT-DETR-L: `rtdetr-l.pt` (SHA-256: `6de60b10...`)
- **Obsolete or Unused Baseline Weights:**
  - `yolov8n.pt`, `yolo11n.pt`, `yolov8s.pt` (Superseded by domain fine-tuned models)

*To audit checkpoint integrity:*
```powershell
python scripts/verify_portability.py
```

---

## Running Automated Tests

Run the complete test suite (129 tests covering ALPR pipeline, character OCR, tracking, privacy, configuration, and golden regressions):

```powershell
# Run complete test suite
pytest -q

# Run regression test on 20 golden plate samples
pytest tests/test_golden_plates.py -v

# Run configuration and portability tests
pytest tests/test_config.py tests/test_domain.py -v
```

---

## Command-Line Interface (`scripts/alpr_cli.py`)

The ALPR CLI supports processing single images, image folders, and video streams:

### 1. Process a Single Image
```powershell
python scripts/alpr_cli.py --input tests/golden/images/0010.jpg --output-dir artifacts/cli_output
```
Outputs:
- JSON reading with Arabic characters, digits, confidence, and bounding box.
- Cropped plate image in `artifacts/cli_output/crops/`.
- Annotated visualization in `artifacts/cli_output/annotated/`.

### 2. Process a Folder of Images
```powershell
python scripts/alpr_cli.py --input tests/golden/images --output-dir artifacts/cli_folder_output
```
Outputs per-image JSONs and a consolidated summary CSV table (`summary.csv`).

### 3. Process Video Stream with Simulated Gate Replay
Tracks vehicles through the gate transit zone, selects the optimal keyframe, resolves multi-frame character conflicts, and emits structured JSON Lines:

```powershell
python scripts/alpr_cli.py --input path/to/gate_cctv.mp4 --fps 2.0 --gate-replay --output-dir artifacts/gate_output
```

*Customizing Checkpoint Locations:*
```powershell
python scripts/alpr_cli.py --input tests/golden/images/0010.jpg `
    --detector-weights "path/to/plate_best.pt" `
    --char-weights "path/to/chars_best.pt" `
    --device cpu
```

---

## REST API Service

The system provides a lightweight FastAPI service for integration with barrier gate controllers:

```powershell
# Start local API server
python vehicle_intelligence/api/app.py --host 0.0.0.0 --port 8000
```
- Interactive Swagger documentation: `http://localhost:8000/docs`
- Health probe: `GET /health`
- Plate reading endpoint: `POST /v1/read-plate`

---

## Operational Scopes, Limitations & Licensing Notices

1. **Heavy Vehicle Domain Specialization:**  
   The vehicle detection model (`yolov8m_finetuned_best.pt`) was adapted on heavy commercial trucks entering gate facilities. It is **not** validated for general passenger cars or motorcycles.
2. **Deep-Night IR Sample Bounds:**  
   Evaluation on deep-night infrared captures is exploratory ($n=2$ frames in test set).
3. **Privacy Compliance:**  
   Face blurring and plate data retention pruning must remain enabled in production per Egyptian Law No. 151 of 2020.
4. **Licensing Manifest:**  
   - Core repository codebase: **Apache-2.0**.
   - RT-DETR-L upstream architecture: **Apache-2.0**.
   - YOLOv8 architecture & weights: **AGPL-3.0** (Commercial deployment requires an enterprise license from Ultralytics).
