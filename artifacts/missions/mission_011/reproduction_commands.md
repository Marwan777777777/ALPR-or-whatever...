# Mission 011: Reproduction Commands

This document provides exact, deterministic commands to reproduce all evaluations, domain-shift experiments, model benchmarks, and integration tests conducted during Mission 011.

---

## 1. Environment & Python Interpreter Verification

Activate the project virtual environment on Windows PowerShell:

```powershell
Set-Location "c:\Users\MARWAN\computer vision"
& ".\.venv\Scripts\python.exe" --version
& ".\.venv\Scripts\python.exe" -c "import torch, cv2, ultralytics; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected environment output:
- Python: `3.11.9`
- PyTorch: `2.6.0+cu124`
- Hardware: `NVIDIA GeForce RTX 4060 Laptop GPU (8187 MiB VRAM)`
- CUDA: `12.4`

---

## 2. Step 1: Baseline 340-Frame Evaluation

Evaluate the existing vehicle detector (`yolov8n.pt` with COCO vehicle classes) across all 340 chronological gate captures in `D:/TruckEntry/`.
Uses `MediaLoader.load_image` to safely bypass Windows non-ASCII / Unicode path issues.

```powershell
& ".\.venv\Scripts\python.exe" scripts/evaluate_baseline_340.py
```

Outputs generated:
- `artifacts/missions/mission_011/vehicle_detection_results.csv`
- Baseline metrics summary printed to stdout:
  - Total frames: 340
  - Daylight (06:30 - 17:30): 203/209 frames detected (97.13%)
  - Deep Night IR (00:00 - 06:30): 0/10 frames detected (0.00%)
  - Evening IR (17:30 - 24:00): 29/121 frames detected (23.97%)
  - Combined IR Failure Rate: 102 misses out of 131 IR frames (77.86%)

---

## 3. Step 2: Controlled Preprocessing Domain-Shift Experiments

Run controlled experiments (one variable modified at a time: CLAHE, Gamma correction, AutoContrast, Bilateral filter, resolution changes, confidence thresholding) on a representative stratified test set:

```powershell
& ".\.venv\Scripts\python.exe" scripts/run_preprocessing_experiments.py
```

Outputs generated:
- `artifacts/missions/mission_011/preprocessing_experiments.csv`
- Visual side-by-side comparative crops and full frames in:
  `artifacts/missions/mission_011/representative_detection_examples/`

---

## 4. Step 3: Alternative Pretrained Vehicle Detector Evaluation

Benchmark candidate models (`yolov8n`, `yolov8s`, `yolov8m`, `yolo11n`, `rtdetr-l`) on identical gate frames, measuring vehicle detection yield, latency, parameters, and commercial licensing compatibility:

```powershell
& ".\.venv\Scripts\python.exe" scripts/evaluate_alternative_detectors.py
```

Outputs generated:
- `artifacts/missions/mission_011/vehicle_detector_comparison.csv`
- Visual detector bounding box overlays in:
  `artifacts/missions/mission_011/representative_detection_examples/`

---

## 5. Step 4: Unit, Regression, and Integration Tests

Run the newly implemented Mission 011 tests alongside the Mission 010 vehicle tracking regression test suite:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests/test_mission_011_vehicle_detection.py tests/test_mission_010_vehicle_tracking.py -v
```

Expected result: 12 passed in ~6 seconds.

Run the complete repository regression test suite to ensure zero regressions across plate detection, character OCR, pipeline orchestration, tracking, and evaluation:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q -rs
```

Expected result: 105 passed, 1 warning in ~15 seconds.

---

## 6. Step 5: Verify Deliverables Integrity

Verify that all required Mission 011 deliverables exist:

```powershell
Get-ChildItem -Path "artifacts/missions/mission_011"
```

Deliverables checklist:
1. `MISSION_011_REPORT.md`
2. `vehicle_detector_comparison.csv`
3. `preprocessing_experiments.csv`
4. `vehicle_detection_results.csv`
5. `representative_detection_examples/` (30 visual demonstration images)
6. `licensing_notes.md`
7. `reproduction_commands.md`
8. `integration_test_results.txt`
