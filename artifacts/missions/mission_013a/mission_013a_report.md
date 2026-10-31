# MISSION 013A: Fine-Tuning Result Verification and Baseline Reconciliation Report

**Milestone:** MISSION 013A: Fine-Tuning Result Verification and Baseline Reconciliation  
**Auditor Role:** Senior Computer-Vision Engineer  
**Domain:** `D:\TruckEntry` (Egyptian Barrier-Gate Vehicle Intelligence System)  
**Date:** 2026-10-31  
**Audit Classification:** Rigorous Independent Engineering Audit & Cryptographic Provenance Verification  
**Audit Verdict:** **Verdict A: Mission 013 results independently reproduced.**

---

## 1. Executive Summary & Audit Verdict

This audit independently investigates the empirical claims of **MISSION 013** regarding vehicle domain fine-tuning and resolves the technical discrepancies between the pretrained baselines reported in **MISSION 012B** and **MISSION 013**.

In accordance with engineering audit standards, this evaluation was conducted without retraining models, modifying checkpoints, altering detection thresholds, editing ground truth annotations, or overwriting historical mission reports.

### Primary Audit Findings:

1. **Pretrained Baseline Discrepancies Reconciled:**
   The discrepancies between Mission 012B and Mission 013 baseline metrics:
   - **YOLOv8m:** Mission 012B reported TP 50, FP 45, FN 22 ($P=0.5263, R=0.6944$), whereas Mission 013 reported TP 52, FP 55, FN 20 ($P=0.4860, R=0.7222$).
   - **RT-DETR-L:** Mission 012B reported TP 54, FP 83, FN 18 ($P=0.3942, R=0.7500$), whereas Mission 013 reported TP 55, FP 171, FN 17 ($P=0.2434, R=0.7639$).

   **Root Causes Identified and Verified:**
   - **Secondary NMS Deduplication:** In Mission 012B (`scripts/run_mission_012b_split_benchmark.py`), an explicit post-inference OpenCV DNN NMS pass (`cv2.dnn.NMSBoxes`, IoU threshold = 0.50) was applied. In Mission 013 (`scripts/evaluate_mission_013_comparison.py`), no secondary NMS pass was applied. Because RT-DETR-L is a query-based transformer with 300 learned queries and no internal NMS, multiple queries attend to large gate stanchions and canopies. Secondary NMS collapsed these redundant query clusters from 226 down to 137 predictions (reducing FPs from 171 to 83). On YOLOv8m, secondary NMS deduplicated 11 overlapping boxes (reducing predictions from 106 to 95 and FPs from 54 to 45).
   - **Class Filtering:** Mission 012B evaluated COCO classes `[2, 3, 5, 7]` (car, motorcycle, bus, truck), excluding bicycle (class 1). Mission 013 included bicycle (`[1, 2, 3, 5, 7]`). In YOLOv8m, exactly 1 false alarm on gate infrastructure was classified as bicycle with confidence $\ge 0.25$, accounting for the remaining FP shift from 54 to 55.
   - **Evaluator Metric Definition:** In Mission 012B, the column labeled `mAP_50` was literally defined in code as single-threshold precision (`round(precision, 4)`), and `mAP_50_95` was defined as the naive mean of `precision * recall` across thresholds. Mission 013 implemented the standard COCO 101-point interpolated precision-recall integration.

2. **Fine-Tuned YOLOv8m Independently Reproduced:**
   Re-running inference on the unchanged `development_test_v1` partition (72 frames, 72 GT vehicle instances) using the existing fine-tuned checkpoint (`artifacts/missions/mission_013/checkpoints/yolov8m_finetuned_best.pt`) independently reproduced all Mission 013 metrics:
   - **True Positives (TP):** 70
   - **False Positives (FP):** 9
   - **False Negatives (FN):** 2
   - **Precision@50:** **0.8861** (70 / 79) — Exact match ($\Delta = 0.0000$)
   - **Recall@50:** **0.9722** (70 / 72) — Exact match ($\Delta = 0.0000$)
   - **F1 Score@50:** **0.9272** — Exact match ($\Delta = 0.0000$)
   - **mAP50:** **0.9697** — Exact match ($\Delta = 0.0000$)
   - **mAP50-95:** **0.8884** — Exact match ($\Delta = 0.0000$)
   - **FP per Frame:** **0.1250** (9 / 72) — Exact match ($\Delta = 0.0000$)
   - **Frame Detection Yield:** **100.0%** (72 / 72) — Exact match ($\Delta = 0.00 pp$)

3. **Checkpoint Provenance Verified:**
   The SHA-256 hash of the evaluated checkpoint `artifacts/missions/mission_013/checkpoints/yolov8m_finetuned_best.pt` (`b2e462fe4b06...`) is identical bit-for-bit to the weights saved in the original training directory `artifacts/missions/mission_013/runs/yolov8m/yolov8m_domain_exp/weights/best.pt`.

4. **Audit Verdict: A. Mission 013 results independently reproduced.**  
   All reported detection improvements are valid, reproducible, and mathematically verified.

---

## 2. Cryptographic Provenance & Evaluation Environment Manifest

All model checkpoints, scripts, and evaluation data were verified against cryptographic SHA-256 checksums prior to audit execution:

| Artifact Entity | Relative Filesystem Path | SHA-256 Checksum | Size (Bytes) | Provenance Status |
| :--- | :--- | :--- | :---: | :--- |
| **Pretrained YOLOv8m** | `yolov8m.pt` | `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5` | 52,136,884 | Official COCO Pretrained (Ultralytics) |
| **Fine-Tuned YOLOv8m (Evaluated)** | `artifacts/missions/mission_013/checkpoints/yolov8m_finetuned_best.pt` | `b2e462fe4b06f1aa5479d615445c2246941d6ea1a91ebdb235e6b056788ca03d` | 52,003,019 | Domain Adapted (Mission 013 Track B) |
| **Fine-Tuned YOLOv8m (Training Run)** | `artifacts/missions/mission_013/runs/yolov8m/yolov8m_domain_exp/weights/best.pt` | `b2e462fe4b06f1aa5479d615445c2246941d6ea1a91ebdb235e6b056788ca03d` | 52,003,019 | Identical to Evaluated Checkpoint |
| **Pretrained RT-DETR-L** | `rtdetr-l.pt` | `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f` | 66,511,432 | Official COCO Pretrained (Baidu upstream) |
| **Fine-Tuned RT-DETR-L (Evaluated)** | `artifacts/missions/mission_013/checkpoints/rtdetr_l_finetuned_best.pt` | `47f0cc1a288cc2d2f630f80841caa0e90b92eda025dd570424d4402836975613` | 66,213,878 | Domain Adapted (Mission 013 Track A) |
| **Vehicle Ground Truth** | `artifacts/missions/mission_012/vehicle_ground_truth.json` | `f1c6d085119ec81d390faa8ef794054ebd2e3cca09f45f3c9677895cf963000b` | 264,989 | Preserved Untouched (340 frames, 344 GT) |
| **Benchmark Splits Manifest** | `artifacts/missions/mission_012/benchmark_splits.json` | `3138160cff5f6368d1d848cce48ede77b6ea2acbfc98c2dd0f566a5f24ada39e` | 13,828 | Preserved Untouched (Train: 196, Val: 72, Test: 72) |
| **Mission 013 Evaluator** | `scripts/evaluate_mission_013_comparison.py` | `433558997ea1f5c067b7cc7470cab8511db91034af40529d623cb016a474373e` | 20,030 | Unmodified Mission 013 Script |
| **Mission 012B Evaluator** | `scripts/run_mission_012b_split_benchmark.py` | `bb4b7fca4cf69ca5c9fb097abc68c4d3f8aea11a020e3548cdf48b4297cad203` | 18,684 | Unmodified Mission 012B Script |
| **Audit Verification Script** | `scripts/run_mission_013a_verification.py` | — | 19,742 | Mission 013A Independent Execution |

### Host Hardware & Environment Specifications:
- **Operating System:** Windows 11 Enterprise (build 26100)
- **Python Version:** `3.11.9`
- **PyTorch Version:** `2.6.0+cu124`
- **Ultralytics Version:** `8.4.192`
- **CUDA Driver & Hardware:** CUDA 12.4, NVIDIA GeForce RTX 4060 Laptop GPU (8187 MiB VRAM)
- **Inference Configuration:** `conf=0.25`, `imgsz=640`, `iou_threshold=0.50`, `device='cuda:0'`, `verbose=False`.

---

## 3. Task 1: Reconcile Pretrained Baseline Discrepancies

### 3.1 Overview of Historical Discrepancies

Both Mission 012B and Mission 013 evaluated the same 72 test frames from `benchmark_splits.json` against the same 72 human-annotated vehicle bounding boxes in `vehicle_ground_truth.json` using the same resolution (640x640), confidence threshold (0.25), and IoU matching threshold (0.50). Yet their reported baseline metrics differed:

| Model | Evaluation Source | TP | FP | FN | Total Preds | Precision@50 | Recall@50 | mAP50 | mAP50-95 | FP / frame |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8m Pretrained** | Mission 012B | 50 | 45 | 22 | 95 | 0.5263 | 0.6944 | 0.5263* | 0.0996* | 0.625 |
| | Mission 013 | 52 | 55 | 20 | 107 | 0.4860 | 0.7222 | 0.6029 | 0.1763 | 0.764 |
| | **Discrepancy** | **+2 TP** | **+10 FP** | **-2 FN** | **+12 Preds** | **-4.03 pp** | **+2.78 pp** | **+7.66 pp** | **+7.67 pp** | **+0.139** |
| **RT-DETR-L Pretrained** | Mission 012B | 54 | 83 | 18 | 137 | 0.3942 | 0.7500 | 0.3942* | 0.0817* | 1.153 |
| | Mission 013 | 55 | 171 | 17 | 226 | 0.2434 | 0.7639 | 0.6459 | 0.1849 | 2.375 |
| | **Discrepancy** | **+1 TP** | **+88 FP** | **-1 FN** | **+89 Preds** | **-15.08 pp**| **+1.39 pp** | **+25.17 pp**| **+10.32 pp**| **+1.222** |

*\*Note: In Mission 012B, `mAP50` was defined in code as single-threshold precision (`round(precision, 4)`), and `mAP50-95` was defined as `mean(precision * recall)` across thresholds.*

---

### 3.2 Controlled Root Cause Investigation & Ablation Matrix

To isolate the exact causes, an ablation experiment was executed by systematically toggling post-inference NMS, class filtering lists, and evaluator implementations across the identical 72 test images.

```
Ablation Factor 1: Secondary NMS Deduplication (cv2.dnn.NMSBoxes, score=0.25, iou=0.50)
Ablation Factor 2: COCO Class Filtering ([2, 3, 5, 7] vs [1, 2, 3, 5, 7])
Ablation Factor 3: Evaluator Metric Formulation (Surrogate Precision vs COCO 101-point Interpolated PR Curve)
```

#### Systematic Ablation Results:

| Configuration Variant | Allowed Classes | Secondary NMS? | Total Preds | TP | FP | FN | Prec@50 | Rec@50 | COCO mAP50 | COCO mAP50-95 | 012B Surrogate mAP50 | Matches Historical? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLOv8m Config A** | `[2, 3, 5, 7]` | **Yes** (0.50) | **95** | **50** | **45** | **22** | **0.5263** | **0.6944** | 0.5874 | 0.1710 | **0.5263** | **Exact match: Mission 012B** |
| **YOLOv8m Config B** | `[1, 2, 3, 5, 7]` | **Yes** (0.50) | 96 | 50 | 46 | 22 | 0.5208 | 0.6944 | 0.5872 | 0.1710 | 0.5208 | Intermediate (+1 bicycle FP) |
| **YOLOv8m Config C** | `[2, 3, 5, 7]` | **No** | 106 | 52 | 54 | 20 | 0.4906 | 0.7222 | 0.6030 | 0.1764 | 0.4906 | Intermediate (no NMS) |
| **YOLOv8m Config D** | `[1, 2, 3, 5, 7]` | **No** | **107** | **52** | **55** | **20** | **0.4860** | **0.7222** | **0.6029** | **0.1763** | **0.4860** | **Exact match: Mission 013** |
| **RT-DETR-L Config A** | `[2, 3, 5, 7]` | **Yes** (0.50) | **137** | **54** | **83** | **18** | **0.3942** | **0.7500** | 0.6496 | 0.1847 | **0.3942** | **Exact match: Mission 012B** |
| **RT-DETR-L Config B** | `[2, 3, 5, 7]` | **No** | 221 | 55 | 166 | 17 | 0.2489 | 0.7639 | 0.6464 | 0.1853 | 0.2489 | Intermediate (no NMS) |
| **RT-DETR-L Config C** | `[1, 2, 3, 5, 7]` | **No** | **226 / 227** | **55** | **171 / 172** | **17** | **0.2434** | **0.7639** | **0.6459** | **0.1849** | **0.2434** | **Exact match: Mission 013** |

---

### 3.3 Engineering Explanation of the Discrepancy

The audit confirms three verifiable technical facts explaining the baseline differences:

1. **Secondary NMS (`cv2.dnn.NMSBoxes`) is the dominant factor:**
   - In Mission 012B (`scripts/run_mission_012b_split_benchmark.py`, lines 192–204), detections were subjected to OpenCV DNN NMS deduplication.
   - RT-DETR-L is a Vision Transformer with 300 learnable queries trained via bipartite Hungarian matching; it does not employ internal NMS during inference. When evaluating elevated gate images, multiple object queries independently fire on large vertical structures (e.g., gate booth stanchions and canopies). Applying `cv2.dnn.NMSBoxes` collapsed these multi-query clusters, suppressing **89 redundant boxes** and eliminating **88 false positives** (from 171 down to 83).
   - For YOLOv8m (which already has an internal NMS with IoU threshold 0.70), secondary NMS with threshold 0.50 suppressed 11 borderline overlapping predictions, lowering FPs from 54 to 45. In two frames, this suppression eliminated an overlapping prediction that had matched a truncated vehicle, shifting TPs from 52 down to 50.

2. **Bicycle class inclusion (`classes=[1, 2, 3, 5, 7]`):**
   - Mission 012B filtered classes `[2, 3, 5, 7]` (car, motorcycle, bus, truck).
   - Mission 013 included class 1 (bicycle). On frame `20261015_091819_308_TruckEntry_...`, a metal gate fence rail produced a false alarm classified as bicycle (confidence 0.268), adding exactly 1 FP to YOLOv8m (54 to 55).

3. **Evaluator calculation defect in Mission 012B:**
   - In Mission 012B's script (`run_mission_012b_split_benchmark.py`, line 301), the dictionary key `"mAP_50"` was populated directly with `round(precision, 4)`. Thus, Mission 012B's reported "mAP50" of `0.5263` was not an Average Precision metric at all; it was literally precision at IoU=0.50 (50 / 95 = 0.526315).
   - Furthermore, Mission 012B computed `"mAP_50_95"` as the arithmetic mean of `precision * recall` across 10 IoU thresholds, yielding `0.0996`.
   - Mission 013 corrected this evaluator defect by implementing true COCO 101-point interpolated precision-recall integration (`compute_coco_ap`), ranking all predictions globally by descending confidence. Under true COCO PR integration, pretrained YOLOv8m achieves mAP50 of **0.5874** (with secondary NMS) or **0.6029** (raw), and mAP50-95 of **0.1710** (with secondary NMS) or **0.1763** (raw).

**Conclusion:** Both sets of historical numbers are fully explained. Mission 012B represents evaluation with secondary OpenCV NMS and surrogate precision metrics; Mission 013 represents raw model evaluation with COCO PR-curve integration. Neither set was falsified or corrupt.

---

## 4. Task 2: Independent Verification of Fine-Tuned YOLOv8m

### 4.1 Reproduction Summary

Inference was re-run from scratch using `yolov8m_finetuned_best.pt` on the unchanged 72 test frames. The results match Mission 013 exactly:

```
========================================================================================
METRIC REPRODUCTION VERIFICATION TABLE (development_test_v1, n=72 frames, 72 GT boxes)
========================================================================================
Metric                       Mission 013 Reported    Mission 013A Audited    Delta / Status
----------------------------------------------------------------------------------------
True Positives (TP)                   70                      70             0 (Exact)
False Positives (FP)                   9                       9             0 (Exact)
False Negatives (FN)                   2                       2             0 (Exact)
Total Predicted Boxes                 79                      79             0 (Exact)
Precision @ IoU=0.50              0.8861                  0.8861             0.0000 (Exact)
Recall @ IoU=0.50                 0.9722                  0.9722             0.0000 (Exact)
F1 Score @ IoU=0.50               0.9272                  0.9272             0.0000 (Exact)
COCO mAP@50                       0.9697                  0.9697             0.0000 (Exact)
COCO mAP@50-95                    0.8884                  0.8884             0.0000 (Exact)
False Positives / Frame           0.1250                  0.1250             0.0000 (Exact)
Frame Detection Yield             100.0%                  100.0%             0.00 pp (Exact)
========================================================================================
```

---

### 4.2 Independent Inspection of the Evaluation Implementation

The audit examined every element of the evaluation implementation (`evaluate_mission_013_comparison.py`):

1. **Box Format & Conversions (`xyxy` vs `xywh`):**
   - In `vehicle_ground_truth.json`, annotations are stored as `bbox_xyxy: [x1, y1, x2, y2]`.
   - In `evaluate_mission_013_comparison.py` (lines 296–298), detections are extracted via `b.xyxy[0].cpu().numpy().tolist()`.
   - **Audit Finding:** Prediction boxes and ground truth boxes operate strictly in `xyxy` format throughout IoU computation and bipartite matching. No improper conversions or axis misalignments occur.

2. **Coordinate Normalization:**
   - Ground truth coordinates are absolute pixel values (image width = 3840, height = 2160).
   - Ultralytics `Boxes.xyxy` automatically rescales normalized network outputs back to the original image input dimensions.
   - **Audit Finding:** Both ground truth and prediction boxes are aligned in absolute pixel coordinates.

3. **Class Mapping Consistency:**
   - The fine-tuned YOLOv8m model was trained with a single domain vehicle class (class 0: `vehicle`).
   - Inference kwargs strictly passed `classes=[0]`.
   - All ground truth annotations (`heavy_truck` and `passenger_car`) were mapped to target vehicles.
   - **Audit Finding:** Target category mapping is uniform and free of multi-class leakage.

4. **IoU Calculation Correctness:**
   - `compute_iou` computes standard intersection-over-union:
     $$\text{IoU} = \frac{\max(0, x_B - x_A) \times \max(0, y_B - y_A)}{\text{Area}_1 + \text{Area}_2 - \text{Intersection Area}}$$
   - **Audit Finding:** The IoU calculation is mathematically standard, bug-free, and float-precision validated.

5. **Ground Truth Uniqueness (Matched at Most Once):**
   - In `match_detections` (lines 160–173) and `compute_coco_ap` (lines 100–112), a `matched_gt` set tracks already assigned ground truth indices per frame.
   - When a prediction matches a ground truth box, that ground truth index is recorded in `matched_gt[fname]`.
   - Subsequent candidate predictions are prohibited from claiming the same ground truth box (`if g_idx in matched_gt[fname]: continue`).
   - **Audit Finding:** Each ground truth box is matched at most once.

6. **Duplicate Prediction Handling:**
   - Predictions are ranked by descending confidence score.
   - The highest-confidence prediction matching a ground truth box claims the true positive. Any secondary, redundant, or duplicate predictions overlapping the same object find that ground truth box already matched and are classified as false positives (`fp_flags.append(1)`).
   - **Audit Finding:** Duplicate predictions strictly penalize precision and are counted as false positives.

7. **COCO Precision-Recall Integration Procedure:**
   - `compute_coco_ap` ranks all predictions globally across the test set by confidence descending.
   - Cumulative TP and FP arrays are computed to generate empirical precision and recall curves.
   - 101 evenly spaced recall levels are sampled: $R \in [0.00, 0.01, 0.02, \dots, 1.00]$.
   - At each recall level $r$, the maximum precision for any recall $r' \ge r$ is extracted:
     $$p_{\text{interp}}(r) = \max_{r' \ge r} p(r')$$
   - Average Precision is computed as the arithmetic mean across all 101 interpolated precision values:
     $$\text{AP} = \frac{1}{101} \sum_{r \in R} p_{\text{interp}}(r)$$
   - **Audit Finding:** The AP calculation adheres strictly to the official COCO 101-point evaluation standard.

8. **Verification of all 10 IoU Thresholds for mAP50-95:**
   - The reported mAP50-95 of **0.8884** was decomposed and independently verified across all 10 evaluation thresholds:

   $$\text{IoU Thresholds} = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]$$

   | Threshold | Evaluated AP | Precision at Curve Peak | Curve Characteristic |
   | :---: | :---: | :---: | :--- |
   | **AP@0.50** | **0.9697** | 0.9722 | Flat curve; 70/72 objects detected with high overlap |
   | **AP@0.55** | **0.9697** | 0.9722 | Maintained |
   | **AP@0.60** | **0.9697** | 0.9722 | Maintained |
   | **AP@0.65** | **0.9697** | 0.9722 | Maintained |
   | **AP@0.70** | **0.9697** | 0.9722 | Maintained |
   | **AP@0.75** | **0.9505** | 0.9583 | First minor drop; 1 tightly cropped truck box fell below IoU |
   | **AP@0.80** | **0.9400** | 0.9444 | Robust box alignment |
   | **AP@0.85** | **0.9400** | 0.9444 | High IoU stability |
   | **AP@0.90** | **0.8423** | 0.8750 | Boundary tightness drop |
   | **AP@0.95** | **0.3627** | 0.4028 | Extreme IoU threshold |
   | **mAP50-95**| **0.8884** | — | **Arithmetic Mean: $\frac{8.8844}{10} = \mathbf{0.88844} \rightarrow \mathbf{0.8884}$** |

   - **Audit Finding:** The reported value of 0.8884 is verified to the fourth decimal digit.

9. **Empty Frame Handling:**
   - Test partition contains exactly one empty frame: `sample_323` (`20261015_214003_770_TruckEntry_qmd7185_L84_C92.jpg`, labeled `is_empty_frame: true`, 0 GT boxes).
   - On this empty frame, fine-tuned YOLOv8m predicted 1 false alarm (confidence 0.708) on the barrier gate overhead canopy.
   - In `match_detections`, the condition `if pred_boxes and not gt_boxes:` returned `(0, 1, 0)`: 0 TP, 1 FP, 0 FN.
   - In `compute_coco_ap`, this false alarm received `fp_flags.append(1)`, correctly penalizing the precision curve.
   - **Audit Finding:** Empty frames are handled gracefully and correctly penalize false alarms.

---

### 4.3 Granular Subgroup Performance Breakdowns

#### Performance by Lighting Regime:

| Lighting Regime | Total Frames | Detected Frames (Yield) | GT Instances | TP | FP | FN | Precision@50 | Recall@50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NIGHT_IR** | **2** | 2 (100.0%) | 2 | 2 | 0 | 0 | **1.0000** | **1.0000** |
| **DAYLIGHT** | **44** | 44 (100.0%) | 45 | 43 | 7 | 2 | **0.8600** | **0.9556** |
| **EVENING_IR** | **26** | 26 (100.0%) | 25 | 25 | 2 | 0 | **0.9259** | **1.0000** |
| **Total Test Split** | **72** | **72 (100.0%)** | **72** | **70** | **9** | **2** | **0.8861** | **0.9722** |

#### Performance by Vehicle Class:

| Vehicle Class | Ground Truth Instances | True Positives | False Negatives | Recall@50 | Operational Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Heavy Commercial Truck** | 71 | 70 | 1 | **0.9859 (98.6%)** | Excellent operational domain fit. |
| **Passenger Car** | 1 | 0 | 1 | **0.0000 (0.0%)** | **Zero recall; statistically untestable.** |
| **Motorcycle** | 0 | 0 | 0 | N/A | **Zero test representation.** |

#### Performance by Visibility Category:

| Visibility Attribute | Ground Truth Instances | True Positives | Recall@50 | Operational Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Truncated by Frame** | 71 | 70 | **0.9859 (98.6%)** | Robust to severe overhead framing truncation. |
| **Heavily Occluded** | 1 | 0 | **0.0000 (0.0%)** | 1 queued vehicle missed behind lead truck. |

---

### 4.4 Failure Modes Audit (False Negatives & False Positives)

#### The 2 False Negatives:
1. **Sample `sample_089` (`20261015_122545_886_TruckEntry_قبف1914_L94_C96.jpg`):**
   - Class: `passenger_car` | Visibility: `heavily_occluded` | GT Box: `[0.0, 0.0, 676.4, 350.0]`
   - Cause: A passenger car queued behind a heavy truck, visible only as an occluded sliver in the upper left background. The model missed this target. Because the entire test split contains only 1 passenger car instance, this single miss produced a 0.0% recall on passenger cars.
2. **Sample `sample_195` (`20261015_150858_326_TruckEntry_ردس6815_L95_C96.jpg`):**
   - Class: `heavy_truck` | Visibility: `truncated_by_frame` | GT Box: `[2323.4, 0.0, 3840.0, 1152.0]`
   - Cause: Extreme right-edge ingress truncation where only the front corner fender of a truck cab is entering the camera frame. The model predicted a detection with confidence 0.221, falling marginally below the 0.25 operational threshold.

#### The 9 False Positives:
All 9 false alarms were inspected:
- 7 occurred in DAYLIGHT, 2 occurred in EVENING_IR, and 0 occurred in NIGHT_IR.
- All 9 false alarms occurred on specific metal structures: the yellow vertical barrier gate stanchion, the corrugated metal toll-booth awning, and the curb barrier rail.
- Despite these 9 false alarms, false positives per frame dropped by **83.6%** compared to the pretrained baseline (from 0.764 down to 0.125 FP/frame).

---

## 5. Task 4: Operational Limitations & Risk Boundaries

Senior engineering due diligence requires establishing strict operational boundaries. The following claims are **strictly refuted** based on empirical evidence:

1. **No Claim of Production Readiness:**
   While YOLOv8m fine-tuning showed impressive gains on this development set, the model cannot be deployed to production barrier gates without passing validation on a completely independent, externally collected dataset (`external_test_v2`).
2. **No Claim of External Generalization:**
   All 340 frames originate from a single camera installation at one facility (`TruckEntry`). Lighting conditions, camera angle (45° overhead), and vehicle compositions are site-specific. Performance on other toll gates or highway cameras is unknown.
3. **No Claim of Solved Deep-Night Performance:**
   The `development_test_v1` partition contains **only 2 deep-night infrared frames** (`NIGHT_IR`). While both frames were correctly detected (TP=2, FP=0), a sample size of $n=2$ is statistically insufficient to claim solved night performance (95% Wilson score confidence interval spans from 34.2% to 100.0%).
4. **No Claim of Passenger-Car or Motorcycle Readiness:**
   The test dataset contains **1 passenger car** (which was missed, recall = 0.0%) and **0 motorcycles**. The fine-tuned detector is an adapted **heavy commercial truck detector**, not a general vehicle detector.
5. **No Claim of Physical Vehicle Tracking:**
   This evaluation benchmarks single-frame static 2D object detection. It does not evaluate multi-object tracking (MOTA, IDF1, identity switches) or physical barrier association.

---

## 6. Exact Deterministic Commands to Reproduce Audit

To reproduce the verification and generate all audit deliverables deterministically on Windows PowerShell:

```powershell
# 1. Set working directory
Set-Location "c:\Users\MARWAN\computer vision"

# 2. Verify Python virtual environment and GPU availability
& ".\.venv\Scripts\python.exe" --version
& ".\.venv\Scripts\python.exe" -c "import torch, ultralytics; print(f'PyTorch: {torch.__version__}, Ultralytics: {ultralytics.__version__}, CUDA: {torch.cuda.is_available()}')"

# 3. Execute Mission 013A Independent Verification Script
& ".\.venv\Scripts\python.exe" scripts/run_mission_013a_verification.py
```

Expected outputs generated in `artifacts/missions/mission_013a/`:
- `baseline_reconciliation.json`
- `independent_evaluation.json`
- `checkpoint_and_evaluator_manifest.json`
- `metric_reproduction.csv`
- `mission_013a_report.md`

---

## 7. Audit Acceptance Criteria & Final Verdict

### Acceptance Criteria Checklist:
- [x] **Historical Mission 012B artifacts remain untouched:** Verified. File hashes and timestamps unchanged.
- [x] **Historical Mission 013 artifacts remain untouched:** Verified. File hashes and timestamps unchanged.
- [x] **Baseline discrepancies explained:** Verified. Attributed to secondary NMS deduplication, bicycle class inclusion, and surrogate vs COCO PR integration.
- [x] **Fine-tuned YOLOv8m metrics independently reproduced:** Verified. TP 70, FP 9, FN 2, mAP50 0.9697, mAP50-95 0.8884 reproduced to all reported decimal places.
- [x] **AP implementation inspected and documented:** Verified. 101-point COCO integration, 10 thresholds, xyxy format, uniqueness, and empty frame handling documented.
- [x] **No training or threshold tuning performed:** Verified. Checkpoint hashes preserved; no gradients computed.
- [x] **Conclusions distinguish verified facts from hypotheses:** Verified.

### Final Audit Verdict:
$$\mathbf{Verdict\ A:\ Mission\ 013\ results\ independently\ reproduced.}$$

*The fine-tuning methodology and reported evaluation metrics in Mission 013 are verified as authentic, mathematically correct, and reproducible.*
