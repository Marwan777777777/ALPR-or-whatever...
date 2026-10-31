# Mission 012B: Reproduction Commands

This document provides exact, deterministic commands to reproduce all evaluations, temporal leakage audits, association reconciliations, split benchmarks, and regression tests for Mission 012B.

---

## 1. Environment & Hardware Verification

Verify Python environment, PyTorch version, and GPU acceleration on Windows PowerShell:

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

## 2. Execute Temporal Split Integrity & Anti-Leakage Audit

Verify deterministic timestamp parsing, 20.0s gap event clustering, and zero intra-event leakage between Train, Validation, and Test partitions:

```powershell
& ".\.venv\Scripts\python.exe" scripts/verify_temporal_split_integrity.py
```

Outputs generated:
- `artifacts/missions/mission_012b/temporal_split_integrity.json`

---

## 3. Run Split-Specific Vehicle Detector Benchmark

Benchmark YOLOv8n, YOLOv8m, and RT-DETR-L separately across the Training set (196 frames), Validation set (72 frames), Held-Out Test set (72 frames), and Entire Dataset (340 frames):

```powershell
& ".\.venv\Scripts\python.exe" scripts/run_mission_012b_split_benchmark.py
```

Outputs generated:
- `artifacts/missions/mission_012b/split_specific_benchmark.csv`
- `artifacts/missions/mission_012b/per_lighting_test_metrics.csv`
- `artifacts/missions/mission_012b/per_class_test_metrics.csv`
- `artifacts/missions/mission_012b/per_visibility_test_metrics.csv`

---

## 4. Run Association Benchmark Reconciliation & NMS Multi-Vehicle Audit

Reconcile all 322 plate observations, separate geometric containment from true association accuracy, and audit NMS deduplication on all multi-vehicle frames:

```powershell
& ".\.venv\Scripts\python.exe" scripts/reconcile_association_and_nms_analysis.py
```

Outputs generated:
- `artifacts/missions/mission_012b/association_reconciliation.csv`
- `artifacts/missions/mission_012b/association_test_metrics.csv`
- `artifacts/missions/mission_012b/nms_multi_vehicle_analysis.md`

---

## 5. Generate Annotation Quality Review Queue

Compile the prioritized review queue covering all 17 adjudication flags, rare classes, geometry extremes, and camera truncation justifications:

```powershell
& ".\.venv\Scripts\python.exe" scripts/build_annotation_review_queue.py
```

Outputs generated:
- `artifacts/missions/mission_012b/annotation_review_queue.csv`

---

## 6. Run Mission 012B Automated Regression Tests

Run the newly implemented Mission 012B unit and integration tests:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests/test_mission_012b_benchmark.py -v
```

Expected result: All tests pass cleanly.

Run the complete repository regression test suite to ensure zero regressions across plate detection, character OCR, pipeline orchestration, tracking, and evaluation:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q -rs
```

Expected result: 111 passed, 1 warning.
