# Mission 012: Reproduction Commands

This document provides exact, deterministic commands to reproduce all independent vehicle ground truth annotations, quality control validations, benchmark splits, detector evaluations, plate association benchmarks, and regression tests for Mission 012.

---

## 1. Environment & Hardware Verification

Verify Python environment and GPU acceleration on Windows PowerShell:

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

## 2. Launch Independent Vehicle Annotation Interface

Launch the dedicated local web server to inspect and interactively annotate the 340 gate captures in `D:\TruckEntry`:

```powershell
& ".\.venv\Scripts\python.exe" scripts/serve_vehicle_annotation_interface.py 8090
```

Open a web browser at:
`http://localhost:8090/`

Interface features:
- Anonymized sample presentation (`SAMPLE-001` to `SAMPLE-340`) with zero model predictions or legacy plate strings in the DOM.
- Full 4K canvas drawing, resizing handles, multi-vehicle labeling.
- Class selection (`heavy_truck`, `light_commercial`, `passenger_car`, `bus`, `microbus_van`, `motorcycle`, `tricycle`, `other_vehicle`).
- Visibility selection (`fully_visible`, `partially_visible`, `heavily_occluded`, `truncated_by_frame`).
- Real-time brightness & contrast adjustment sliders (essential for dark night IR frames).
- One-click export to JSON and CSV.

---

## 3. Generate Anti-Leakage Benchmark Splits

Partition the 340 gate captures into temporal event clusters (gap > 20s) and allocate to Train (57.65%), Validation (21.18%), and Test (21.18%) stratified across lighting regimes:

```powershell
& ".\.venv\Scripts\python.exe" scripts/generate_benchmark_splits.py
```

Outputs generated:
- `artifacts/missions/mission_012/benchmark_splits.json`

---

## 4. Build & Validate Vehicle Ground Truth Dataset

Generate the verified independent ground truth across all 340 gate captures and execute automated quality control:

```powershell
& ".\.venv\Scripts\python.exe" scripts/build_vehicle_ground_truth.py
```

Outputs generated:
- `artifacts/missions/mission_012/vehicle_ground_truth.json`
- `artifacts/missions/mission_012/annotations/annotations.csv`
- `artifacts/missions/mission_012/annotation_validation_report.json`

---

## 5. Execute Comprehensive Detector Benchmarking & Association Evaluation

Evaluate YOLOv8n, YOLOv8m, and RT-DETR-L on the independent ground truth across all 340 captures, measuring object precision, recall, mAP@0.50, mAP@0.50:0.95, per-lighting metrics, and plate-to-vehicle association containment:

```powershell
& ".\.venv\Scripts\python.exe" scripts/run_mission_012_benchmark.py
```

Outputs generated:
- `artifacts/missions/mission_012/detector_benchmark.csv`
- `artifacts/missions/mission_012/per_lighting_metrics.csv`
- `artifacts/missions/mission_012/per_class_metrics.csv`
- `artifacts/missions/mission_012/association_benchmark.csv`
- `artifacts/missions/mission_012/representative_examples/` (visual overlay JPEGs)

---

## 6. Run Integration & Regression Test Suite

Run the Mission 012 test suite alongside Missions 010 and 011:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests/test_mission_010_vehicle_tracking.py tests/test_mission_011_vehicle_detection.py tests/test_mission_012_benchmark.py -v
```

Expected result: 15 passed in ~6 seconds.

Run the complete repository regression test suite:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q -rs
```

Expected result: 108 passed, 1 warning in ~14 seconds.

---

## 7. Deliverables Checklist Verification

Verify all Mission 012 deliverables exist in `artifacts/missions/mission_012/`:

```powershell
Get-ChildItem -Path "artifacts/missions/mission_012"
```

Checklist:
1. `MISSION_012_REPORT.md`
2. `annotation_interface/` (`index.html`, `styles.css`, `annotator.js`)
3. `annotations/` (`annotations.csv`)
4. `annotation_validation_report.json`
5. `benchmark_splits.json`
6. `vehicle_ground_truth.json`
7. `detector_benchmark.csv`
8. `per_class_metrics.csv`
9. `per_lighting_metrics.csv`
10. `association_benchmark.csv`
11. `representative_examples/` (5 visual demonstration images)
12. `reproduction_commands.md`
13. `integration_test_results.txt`
