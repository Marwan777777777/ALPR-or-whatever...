# Mission 010: Reproduction Commands

This document provides the exact, deterministic commands to reproduce all test suites, gate validation pipelines, and artifact generation for **Mission 010: Vehicle-Level Detection, Tracking, and Plate Association**.

## Environment Prerequisites

- Windows 11 / PowerShell
- Python 3.11 virtual environment at `.venv`
- PyTorch 2.6.0+cu124 with CUDA 12.4
- NVIDIA GeForce RTX 4060 Laptop GPU (or CPU execution)

---

## 1. Execute Mission 010 Deterministic Synthetic Integration Tests

Runs all 8 required vehicle-level tracking and plate association scenarios without GPU or external media dependencies:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_mission_010_vehicle_tracking.py -v --tb=short
```

Expected output:
```text
tests/test_mission_010_vehicle_tracking.py::test_scenario_1_single_vehicle_single_plate_multiple_frames PASSED
tests/test_mission_010_vehicle_tracking.py::test_scenario_2_two_vehicles_in_same_frame PASSED
tests/test_mission_010_vehicle_tracking.py::test_scenario_3_two_vehicles_passing_close_together PASSED
tests/test_mission_010_vehicle_tracking.py::test_scenario_4_intermittent_plate_detection PASSED
tests/test_mission_010_vehicle_tracking.py::test_scenario_5_plate_detected_without_vehicle_detection PASSED
tests/test_mission_010_vehicle_tracking.py::test_scenario_6_ambiguous_plate_vehicle_association PASSED
tests/test_mission_010_vehicle_tracking.py::test_scenario_7_conflicting_ocr_observations_for_one_vehicle PASSED
tests/test_mission_010_vehicle_tracking.py::test_scenario_8_temporary_vehicle_detection_loss_and_recovery PASSED
============================== 8 passed in 0.04s ==============================
```

---

## 2. Execute Full Repository Regression Test Suite

Verifies that all 101 tests across the entire repository (domain, config, detectors, tracker, aggregator, API, triage audits, reading order) pass with zero regressions:

```powershell
.venv\Scripts\python.exe -m pytest -q -rs
```

Expected output:
```text
======================= 101 passed, 1 warning in 8.89s =======================
```

---

## 3. Run Gate Validation and CSV Deliverables Generation

Executes chronological sorting, vehicle detection, vehicle tracking, plate-to-vehicle association, and transit event consolidation across all 340 gate captures in `D:/TruckEntry`:

```powershell
.venv\Scripts\python.exe scripts/run_mission_010_validation.py
```

Generated outputs in `artifacts/missions/mission_010/`:
- `vehicle_detection_results.csv`
- `vehicle_track_analysis.csv`
- `plate_vehicle_association_results.csv`
- `event_examples.csv`

---

## 4. Analyze Chronological Gaps and Trajectory Feasibility

Verifies timestamp intervals across gate frame captures to confirm non-continuous snapshot behavior:

```powershell
.venv\Scripts\python.exe scratch/analyze_gate_timestamps.py
```
