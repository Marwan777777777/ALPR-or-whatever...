# Mission 009: Reproduction Commands and Verification Guide

This document provides exact, reproducible command-line instructions to run all Mission 009 benchmarks, pipeline integrations, discrepancy reconciliations, and integration tests.

---

## 1. Environment & Pre-requisites

Ensure you are in the repository root with the project virtual environment active:

```powershell
# Activate virtual environment
& .venv\Scripts\Activate.ps1

# Verify Python version, PyTorch version, and GPU availability
.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 2. Mission 008 Discrepancy Resolution & 77-Crop Comparison (Objective 1)

Executes the per-sample evaluation comparing character NMS IoU 0.70 vs IoU 0.45 across all 77 in-vocab readable gate triage crops:

```powershell
.venv\Scripts\python.exe scripts/export_iou_70_vs_45_comparison.py
```

*Output Artifact:* `artifacts/missions/mission_009/iou_70_vs_45_per_sample.csv`

---

## 3. End-to-End Gate Replay Benchmark (Objective 3)

Measures complete pipeline latency (loading/decoding, plate detection, crop extraction, character OCR, tracking, event emission) across Configurations A, B, and C on identical inputs:

```powershell
.venv\Scripts\python.exe scripts/run_gate_replay_benchmark_009.py
```

*Output Artifacts:*
- `artifacts/missions/mission_009/end_to_end_benchmark.json`
- `artifacts/missions/mission_009/per_configuration_results.csv`

---

## 4. Event-Level Multi-Frame Replay Analysis (Objective 4)

Replays multi-frame vehicle transit bursts to evaluate tracking stability, evidence upgrading, conflicting read preservation, and duplicate event prevention:

```powershell
.venv\Scripts\python.exe scripts/run_event_replay_analysis_009.py
```

*Output Artifact:* `artifacts/missions/mission_009/event_replay_analysis.csv`

---

## 5. Integration Test Suite Execution (Objective 5)

Runs the comprehensive deterministic integration test suite covering configuration propagation, NMS/resolution parameters, multi-frame tracking, OCR conflict preservation, and evidence persistence:

```powershell
# Run Mission 009 integration test suite
.venv\Scripts\python.exe -m pytest tests/test_mission_009_integration.py -v

# Run full project test suite (all 93 tests)
.venv\Scripts\python.exe -m pytest -q -rs
```

*Output Artifact:* `artifacts/missions/mission_009/integration_test_results.txt`

---

## 6. CLI Execution (Production Pipeline)

### 6.1 Production Configuration (Configuration B: 640 / char_iou 0.45)
```powershell
.venv\Scripts\python.exe scripts/alpr_cli.py --input artifacts/triage/samples/frame_001.jpg --output-dir artifacts/cli_output --det-imgsz 640 --char-iou 0.45 --agnostic-nms
```

### 6.2 High-Resolution Option (Configuration C: 1280 / char_iou 0.45)
```powershell
.venv\Scripts\python.exe scripts/alpr_cli.py --input artifacts/triage/samples/frame_001.jpg --output-dir artifacts/cli_output --det-imgsz 1280 --char-iou 0.45 --agnostic-nms
```

### 6.3 Historical Baseline Reproduction (Configuration A: 640 / char_iou 0.70 / no-agnostic-nms)
```powershell
.venv\Scripts\python.exe scripts/alpr_cli.py --input artifacts/triage/samples/frame_001.jpg --output-dir artifacts/cli_output --det-imgsz 640 --char-iou 0.70 --no-agnostic-nms
```

### 6.4 Gate Transit Tracking Replay
```powershell
.venv\Scripts\python.exe scripts/alpr_cli.py --input artifacts/triage/samples/ --output-dir artifacts/cli_output --gate-replay --det-imgsz 1280 --char-iou 0.45
```
