# Mission 008: Reproduction Commands and Verification Guide

This document provides exact, reproducible command-line instructions to reproduce all Mission 008 results, evaluations, and experiments.

---

## 1. Environment & Pre-requisites

Ensure you are in the repository root with the project virtual environment active:

```powershell
# Activate project virtual environment
& .venv\Scripts\Activate.ps1

# Verify Python version and CUDA availability
.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 2. Controlled NMS Evaluation (Objective 1)

Re-runs character recognition across all 81 gate triage crops comparing:
- Historical baseline: `agnostic_nms=False` (67/77 = 87.01%, CER = 0.0214)
- Mission 008 intended: `agnostic_nms=True` (73/77 = 94.81%, CER = 0.0097)
- Verifies the 5 corrected crops individually (Rows 25, 45, 47, 94, 96).

```powershell
.venv\Scripts\python.exe scripts/evaluate_gate_crops_nms.py
```
*Output Artifact:* `artifacts/missions/mission_008/nms_comparison.json`

---

## 3. Plate Detection Miss Diagnostics (Objective 2)

Generates the observable failure mode classification table for all 18 visible-plate misses:

```powershell
.venv\Scripts\python.exe scripts/export_detection_miss_analysis.py
```
*Output Artifact:* `artifacts/missions/mission_008/detection_miss_analysis.csv`

---

## 4. Remaining OCR Error Analysis (Objective 3)

Generates the error taxonomy and pattern analysis for all remaining OCR failures under class-agnostic NMS:

```powershell
.venv\Scripts\python.exe scripts/export_ocr_error_analysis.py
```
*Output Artifact:* `artifacts/missions/mission_008/ocr_error_analysis.csv`

---

## 5. Targeted Controlled Experiments (Objective 4)

Executes the three diagnostic-driven improvement experiments:
1. **Experiment A:** Character Detector NMS IoU Threshold Tuning (`iou` in `[0.70, 0.60, 0.55, 0.50, 0.45]`) on 300 validation crops, verifying improvement on gate triage.
2. **Experiment B:** Crop Border Padding (`0%`, `4%`, `6%`, `8%`, `10%`) on gate frames.
3. **Experiment C:** Plate Detector Input Resolution (`640` vs `960` vs `1280`) on 4K gate frames.

```powershell
.venv\Scripts\python.exe scripts/run_targeted_experiments_008.py
```
*Output Artifact:* `artifacts/missions/mission_008/experiment_results.json`

---

## 6. CLI Pipeline Execution (Historical vs Intended)

### 6.1 Intended Mission 008 Configuration (`agnostic_nms=True`)
Runs the full end-to-end pipeline on an image or directory using the new default class-agnostic NMS:

```powershell
.venv\Scripts\python.exe scripts/alpr_cli.py --input artifacts/triage/samples/frame_001.jpg --output-dir artifacts/cli_output --device cuda:0 --agnostic-nms
```

### 6.2 Historical Baseline Configuration (`agnostic_nms=False`)
Reproduces the historical baseline behavior via the explicit `--no-agnostic-nms` flag:

```powershell
.venv\Scripts\python.exe scripts/alpr_cli.py --input artifacts/triage/samples/frame_001.jpg --output-dir artifacts/cli_output --device cuda:0 --no-agnostic-nms
```

---

## 7. Full Test Suite Verification

Runs the complete test suite including new Mission 008 regression tests:

```powershell
.venv\Scripts\python.exe -m pytest -q -rs
```
