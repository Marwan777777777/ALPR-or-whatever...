# MISSION 013: Exploratory Vehicle Domain Fine-Tuning Report

**Milestone:** MISSION 013: Exploratory Vehicle Domain Fine-Tuning  
**Domain:** `D:\TruckEntry` (Egyptian Barrier-Gate Vehicle Intelligence System)  
**Date:** 2026-10-31  
**Status:** Completed & Validated  
**Classification:** Exploratory Domain Adaptation Experiment (Pre-Production Milestone)

---

## 1. Executive Summary

Mission 013 executes the first controlled domain-adaptation experiment for the Egyptian barrier-gate vehicle detection system. The operational domain (`D:\TruckEntry`) presents severe computer vision challenges: an elevated 45° overhead camera angle, harsh evening and deep-night infrared (IR) sensor regimes, extreme heavy truck dominance (97.7%), framing truncation (98.6% of vehicles truncated), and structural false alarms on gate infrastructure.

Two independent experimental tracks were benchmarked before and after conservative domain transfer learning:
- **Track A (RT-DETR-L):** Vision Transformer architecture (HGNetv2 backbone + AIFI hybrid encoder + Hungarian bipartite matcher), 32.8M parameters, Apache-2.0 upstream provenance.
- **Track B (YOLOv8m):** Deep CNN comparative baseline (CSPDarknet backbone + decoupled anchor-free head + Task-Aligned Assigner), 25.9M parameters, AGPL-3.0 provenance.

### Core Empirical Findings:
1. **Fine-Tuned YOLOv8m Emerges as the Decisive Winner:**
   - **Precision@50** increased from **0.4860 to 0.8861** (+40.01 percentage points).
   - **Recall@50** increased from **0.7222 to 0.9722** (+25.00 percentage points).
   - **F1 Score** rose from **0.5810 to 0.9272** (+0.3462).
   - **mAP50** jumped from **0.6029 to 0.9697** (+36.68 percentage points).
   - **mAP50-95** soared from **0.1763 to 0.8884** (+71.21 percentage points).
   - **False Positives per Frame** dropped by **83.6%**, from **0.764 to 0.125** FP/frame.
   - **Frame Detection Yield** reached **100.0%** (72/72 development test frames detected).
   - **Inference Latency** maintained real-time edge execution: **p50 = 18.42 ms**, **p95 = 26.38 ms** (~51 FPS) on the RTX 4060 GPU with 499.85 MiB peak VRAM.
2. **Fine-Tuned RT-DETR-L Experienced Representation Collapse:**
   - Pretrained RT-DETR-L suffered heavy false alarms on barrier gate stanchions (2.375 FP/frame, precision 0.2434).
   - Under conservative fine-tuning on 196 training samples, the vision transformer cross-scale encoder and Hungarian bipartite assignment suffered from sample-scarcity collapse: query confidence scores shrunk below the 0.25 operational threshold on evaluation frames, driving recall down to 0.0556 and yield down to 5.56%.
3. **Severe Operational Scopes & Limitations:**
   - The adaptation is valid **only for heavy commercial trucks** (dump trucks, flatbeds, multi-axle trailers). Ground truth rare classes (passenger cars $n=1$, motorcycles $n=0$) had 0% recall, establishing that the dataset cannot validate general vehicle detection.
   - Deep-night IR evaluation sample size is limited ($n=2$), requiring subsequent external validation.
4. **Final Recommendation:** **YOLOv8m fine-tuned is recommended as the leading experimental candidate.** Production validation remains strictly contingent on the external, independently collected `external_test_v2` dataset.

---

## 2. Experimental Boundaries & Split Integrity

The experimental dataset comprises 340 gate captures partitioned using the Mission 012/012C verified chronological event-block clustering protocol (20-second consecutive gap rule, zero intra-event leakage):

| Split Name | Frames Count | Temporal Events | Vehicle Instances | Primary Operational Function |
| :--- | :---: | :---: | :---: | :---: |
| **Training (`train`)** | **196** | 175 | 200 | Exclusively permitted to update model weights. |
| **Validation (`val`)** | **72** | 58 | 72 | Checkpoint selection, loss monitoring, early stopping. |
| **Development Test (`development_test_v1`)** | **72** | 58 | 72 | Internal post-training comparison only; not general proof. |
| **Total** | **340** | **291** | **344** | Pairwise mutually disjoint; zero leakage verified. |

### Anti-Leakage & Ground Truth Integrity:
- **No Ground Truth Modification:** The human-annotated vehicle ground truth (`vehicle_ground_truth.json`, SHA-256: `f1c6d085...`) was preserved bit-for-bit without alteration or model-assisted pseudo-labeling.
- **Strict Data Isolation:** Training and validation were conducted from isolated disk directories (`artifacts/missions/mission_013/dataset/`). Development test images were never exposed during gradient descent.
- **Original Pretrained Weights Preserved:** Pretrained checkpoints in the repository root (`rtdetr-l.pt` and `yolov8m.pt`) were cryptographically verified before and after training; their SHA-256 hashes remained identical.

---

## 3. Cryptographic Provenance & Licensing Manifest

All model checkpoints, source architectures, and configurations are cryptographically traced:

| Model Entity | Checkpoint RelPath | SHA-256 Checksum | Parameters | License Status |
| :--- | :--- | :--- | :---: | :--- |
| **Pretrained RT-DETR-L** | `rtdetr-l.pt` | `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f` | 32,970,476 | Apache-2.0 upstream (Baidu) / AGPL-3.0 runner |
| **Fine-Tuned RT-DETR-L** | `artifacts/missions/mission_013/checkpoints/rtdetr_l_finetuned_best.pt` | `47f0cc1a288cc2d2f630f80841caa0e90b92eda025dd570424d4402836975613` | 32,808,131 | Apache-2.0 architecture + fine-tuned domain weights |
| **Pretrained YOLOv8m** | `yolov8m.pt` | `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5` | 25,902,640 | AGPL-3.0 Ultralytics |
| **Fine-Tuned YOLOv8m** | `artifacts/missions/mission_013/checkpoints/yolov8m_finetuned_best.pt` | `b2e462fe4b06f1aa5479d615445c2246941d6ea1a91ebdb235e6b056788ca03d` | 25,856,899 | AGPL-3.0 Ultralytics |

### Upstream Track A Provenance:
The upstream RT-DETR repository was cloned from official Baidu sources (`lyuwenyu/RT-DETR`) at `vendor/rtdetr_upstream` (commit `df2a96c8c3a098567a1149bf0248da6efe6d495d`, Apache-2.0). Because Torchvision 0.21.0 deprecates `torchvision.transforms.v2.datapoints` in favor of `tv_tensors`, domain execution used the PyTorch execution wrapper while preserving clean weight state dictionaries.

---

## 4. Pre-Training Baseline Performance & Latency Profiles

Baseline profiling was conducted on an NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) under PyTorch 2.6.0+cu124, using CUDA synchronization, 50 warm-up runs, and 200 timed iterations at 640x640 resolution:

| Metric | Pretrained RT-DETR-L (COCO) | Pretrained YOLOv8m (COCO) | Delta (YOLO vs RT-DETR) |
| :--- | :---: | :---: | :---: |
| **Median Latency (p50)** | **45.45 ms** | **18.20 ms** | -27.25 ms (2.5x faster) |
| **95th Percentile Latency (p95)** | **58.16 ms** | **24.30 ms** | -33.86 ms |
| **Mean Latency ± Std** | 47.03 ± 5.05 ms | 19.02 ± 3.73 ms | -28.01 ms |
| **Mean FPS** | 21.3 FPS | 52.6 FPS | +31.3 FPS |
| **Peak Inference VRAM** | 243.85 MiB | 398.33 MiB | +154.48 MiB |
| **Test Precision@50** | 0.2434 (55 TP, 171 FP) | 0.4860 (52 TP, 55 FP) | +0.2426 (+24.26 pp) |
| **Test Recall@50** | 0.7639 (55/72) | 0.7222 (52/72) | -0.0417 (-4.17 pp) |
| **Test F1 Score** | 0.3691 | 0.5810 | +0.2119 |
| **Test mAP50** | 0.6459 | 0.6029 | -0.0430 |
| **Test mAP50-95** | 0.1849 | 0.1763 | -0.0086 |
| **Test False Positives / Frame** | **2.375** (171 FP) | **0.764** (55 FP) | -1.611 (-67.8% FP) |
| **Test Frame Yield** | 91.67% (66/72) | 86.11% (62/72) | -5.56 pp |

**Baseline Observation:** Pretrained RT-DETR-L had a higher detection yield (91.7% vs 86.1%), but suffered massive false alarm rates (2.38 FP/frame vs 0.76 FP/frame), detecting gate booth roofs, corrugated barriers, and road surface curbs as vehicles.

---

## 5. Domain Fine-Tuning Execution & Hyperparameters

Both models were trained using conservative transfer learning configurations designed to prevent catastrophic forgetting and maintain training stability on 196 domain images:

| Hyperparameter | Track A: RT-DETR-L | Track B: YOLOv8m | Rationale & Engineering Justification |
| :--- | :---: | :---: | :--- |
| **Base Weights** | `rtdetr-l.pt` (preserved) | `yolov8m.pt` (preserved) | Initialized from official COCO pretrained weights. |
| **Dataset Configuration** | `dataset.yaml` (1 class: `vehicle`) | `dataset.yaml` (1 class: `vehicle`) | Unified target vehicle category mapping. |
| **Epochs Scheduled** | 25 | 25 | Conservative epoch budget for small sample adaptation. |
| **Epochs Completed** | 22 (Early stopped at ep 14) | 25 (Full run) | Monitored on 72 validation frames; early stopped on no gain. |
| **Batch Size** | 4 | 8 | Batch 4 on RT-DETR-L to preserve transformer memory bounds. |
| **Input Resolution** | 640x640 | 640x640 | Maintained identical resolution across baseline and training. |
| **Optimizer** | AdamW | AdamW | Weight decay regularization on transfer learning heads. |
| **Initial LR (`lr0`)** | **0.0001** (1e-4) | **0.001** (1e-3) | Lower LR for sensitive vision transformer attention queries. |
| **Final LR Factor (`lrf`)** | 0.01 | 0.01 | Cosine/linear schedule down to 1% of initial LR. |
| **Warmup Epochs** | 3.0 | 3.0 | Smooth initialization for newly adapted classification heads. |
| **Early Stopping Patience**| 8 | 8 | Prevents overfitting on validation performance plateau. |
| **Dataloader Workers** | 0 | 0 | Windows in-process execution (avoids WinError 1455 memory exhaustion). |
| **Deterministic Seed** | 42 | 42 | Full reproducibility. |
| **Training Duration** | 776.7 s (~12.9 min) | 493.9 s (~8.2 min) | GPU compute on RTX 4060 Laptop GPU. |

---

## 6. Comprehensive 4-Model Comparative Evaluation

Post-training evaluation was performed across all 72 frames in `development_test_v1` (and cross-checked against 72 frames in `validation`). Object-level detection metrics are evaluated at confidence threshold = 0.25 and IoU threshold = 0.50. mAP50 and mAP50-95 are computed via standard COCO 101-point interpolated precision-recall integration.

### 6.1 Primary Benchmark Summary Table (`development_test_v1`)

| Metric | Pretrained RT-DETR-L | Fine-Tuned RT-DETR-L | Pretrained YOLOv8m | Fine-Tuned YOLOv8m | YOLOv8m Fine-Tuning Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Precision@50** | 0.2434 | 0.2222 | 0.4860 | **0.8861** | **+40.01 pp (+82.3%)** |
| **Recall@50** | 0.7639 | 0.0556 | 0.7222 | **0.9722** | **+25.00 pp (+34.6%)** |
| **F1 Score** | 0.3691 | 0.0889 | 0.5810 | **0.9272** | **+0.3462 (+59.6%)** |
| **mAP50** | 0.6459 | 0.0421 | 0.6029 | **0.9697** | **+36.68 pp (+60.8%)** |
| **mAP50-95** | 0.1849 | 0.0307 | 0.1763 | **0.8884** | **+71.21 pp (+403.9%)** |
| **True Positives (TP)** | 55 | 4 | 52 | **70** | **+18 TPs** |
| **False Positives (FP)** | 171 | 14 | 55 | **9** | **-46 FPs (-83.6%)** |
| **False Negatives (FN)** | 17 | 68 | 20 | **2** | **-18 FNs (-90.0%)** |
| **False Positives / Frame**| 2.375 | 0.194 | 0.764 | **0.125** | **-0.639 FP/frame** |
| **Frame Detection Yield** | 91.67% | 5.56% | 86.11% | **100.0%** | **+13.89 pp** |
| **Median Latency (p50)** | 45.45 ms | 45.95 ms | 18.20 ms | **18.42 ms** | +0.22 ms |
| **95th Percentile Latency**| 58.16 ms | 56.80 ms | 24.30 ms | **26.38 ms** | +2.08 ms |
| **Peak GPU VRAM** | 243.85 MiB | 335.39 MiB | 398.33 MiB | **499.85 MiB** | +101.52 MiB |

---

## 7. Granular Performance Breakdowns

### 7.1 Performance by Lighting Condition (`development_test_v1`)

| Lighting Condition | Model Candidate | Frames / GT | Detected Frames (Yield) | TP | FP | FN | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deep-Night IR** | Pretrained RT-DETR-L | 2 / 2 | 2 (100.0%) | 1 | 1 | 1 | 0.5000 | 0.5000 |
| | Fine-Tuned RT-DETR-L | 2 / 2 | 0 (0.0%) | 0 | 0 | 2 | 0.0000 | 0.0000 |
| | Pretrained YOLOv8m | 2 / 2 | 2 (100.0%) | 1 | 1 | 1 | 0.5000 | 0.5000 |
| | **Fine-Tuned YOLOv8m** | **2 / 2** | **2 (100.0%)** | **2** | **0** | **0** | **1.0000** | **1.0000** |
| **Evening IR** | Pretrained RT-DETR-L | 26 / 26 | 20 (76.9%) | 14 | 59 | 12 | 0.1918 | 0.5385 |
| | Fine-Tuned RT-DETR-L | 26 / 26 | 2 (7.7%) | 2 | 9 | 24 | 0.1818 | 0.0769 |
| | Pretrained YOLOv8m | 26 / 26 | 16 (61.5%) | 13 | 12 | 13 | 0.5200 | 0.5000 |
| | **Fine-Tuned YOLOv8m** | **26 / 26** | **26 (100.0%)** | **25** | **2** | **1** | **0.9259** | **0.9615** |
| **Daylight** | Pretrained RT-DETR-L | 44 / 44 | 44 (100.0%) | 40 | 111 | 4 | 0.2649 | 0.9091 |
| | Fine-Tuned RT-DETR-L | 44 / 44 | 2 (4.5%) | 2 | 5 | 42 | 0.2857 | 0.0455 |
| | Pretrained YOLOv8m | 44 / 44 | 44 (100.0%) | 38 | 42 | 6 | 0.4750 | 0.8636 |
| | **Fine-Tuned YOLOv8m** | **44 / 44** | **44 (100.0%)** | **43** | **7** | **1** | **0.8600** | **0.9773** |

**Lighting Analysis:**
- In Evening IR, Fine-Tuned YOLOv8m eliminated detection dropouts, lifting yield from 61.5% to **100.0%**, recovering 12 previously missed vehicles and slashing false alarms from 12 to 2.
- In Deep-Night IR, both ground truth trucks were detected with zero false alarms (2 TP, 0 FP). Note: as documented in `limitations.md`, $n=2$ frames provides directional evidence but lacks statistical power for general night claims.

---

### 7.2 Performance by Vehicle Class (`development_test_v1`)

| Vehicle Class | Ground Truth Instances | Pretrained RT-DETR-L (Recall) | Fine-Tuned RT-DETR-L (Recall) | Pretrained YOLOv8m (Recall) | Fine-Tuned YOLOv8m (Recall) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Heavy Commercial Truck (`heavy_truck`)** | 71 | 77.46% (55/71) | 5.63% (4/71) | 73.24% (52/71) | **98.59% (70/71)** |
| **Passenger Car (`passenger_car`)** | 1 | 0.0% (0/1) | 0.0% (0/1) | 0.0% (0/1) | **0.0% (0/1)** |

**Class Imbalance Impact:**
- Because heavy trucks comprise 97.5% of the training data and 98.6% of the test data, the fine-tuned model has successfully specialized into an ingress heavy-commercial vehicle detector.
- The single passenger car instance in the test split was missed by all four models. **This proves conclusively that the current experiment does not establish passenger car detection capability.**

---

### 7.3 Performance by Visibility & Truncation Category (`development_test_v1`)

| Visibility Attribute | Ground Truth Instances | Pretrained RT-DETR-L (Recall) | Fine-Tuned RT-DETR-L (Recall) | Pretrained YOLOv8m (Recall) | Fine-Tuned YOLOv8m (Recall) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Truncated by Frame Boundary** | 71 | 77.46% (55/71) | 5.63% (4/71) | 73.24% (52/71) | **98.59% (70/71)** |
| **Heavily Occluded** | 1 | 0.0% (0/1) | 0.0% (0/1) | 0.0% (0/1) | **0.0% (0/1)** |

**Truncation Handling:**
Under the 45° overhead camera geometry, vehicles entering the frame are virtually always truncated. Pretrained COCO models frequently failed to trigger or split the truck into partial fragments. Fine-Tuned YOLOv8m successfully internalizes this truncation prior, detecting 70 out of 71 truncated trucks (98.59% recall).

---

## 8. Diagnostic Analysis: Architectural Divergence

The stark contrast between Track A and Track B provides a critical empirical lesson in computer vision domain adaptation:

```
[Target Domain Feeds: 196 Training Samples]
        |
        +---> Track A: RT-DETR-L (Vision Transformer)
        |       - Hybrid Encoder + 300 Object Queries + Hungarian Bipartite Matcher
        |       - SENSITIVITY: Transformer attention maps require thousands of images.
        |       - RESULT: Query confidence shrinkage -> Recall collapsed to 5.56%.
        |
        +---> Track B: YOLOv8m (CSPDarknet CNN)
                - Decoupled Anchor-Free Head + Task-Aligned Assigner
                - STABILITY: Convolutional inductive bias preserves spatial locality.
                - RESULT: Smooth transfer learning -> Recall 97.22%, Precision 88.61%, mAP 0.9697.
```

1. **Why RT-DETR-L Suffered Representation Collapse:**
   - RT-DETR relies on a set-prediction loss via Hungarian bipartite matching across 300 fixed queries. In a full COCO pretraining regime with millions of annotated instances, queries learn distinct spatial and scale specializations.
   - When exposed to a very small domain dataset (196 frames) where almost every image contains exactly one large heavy truck, Hungarian assignment gradients repeatedly penalize 299 out of 300 queries as background negatives.
   - With an AdamW learning rate of 1e-4, the decoder query classification weights rapidly saturated toward the negative logit prior, causing confidence scores to collapse below 0.25. (At conf=0.01, RT-DETR-L outputs boxes, but they are noisy and unfocused). Vision transformers require extensive regularization or frozen-backbone linear probing when fine-tuning on fewer than 1,000 images.
2. **Why YOLOv8m Succeeded Decisively:**
   - YOLOv8m utilizes a decoupled anchor-free convolutional head with a Task-Aligned Assigner that dynamically balances classification and localization alignment.
   - The inductive bias of convolutional feature hierarchies (preserving translation equivariance and local spatial features) prevents catastrophic representation collapse even with small sample sizes.
   - The model learned to suppress static gate stanchions and booth roofs while consolidating truncated truck bodies into single coherent bounding boxes, delivering an 83.6% reduction in false alarms and 100% frame yield.

---

## 9. Verification & Automated Test Suite

A dedicated regression test suite was implemented in `tests/test_mission_013_domain_training.py` and executed against the repository:
1. `TestMission013Configurations`: Validates hyperparameter YAMLs (epochs=25, lr, seed=42, deterministic=True, workers=0).
2. `TestMission013CheckpointPreservation`: Asserts cryptographic bit-exact immutability of original `rtdetr-l.pt` and `yolov8m.pt` checkpoints.
3. `TestMission013DataSplitIsolation`: Validates that train (196), val (72), and dev-test (72) are pairwise mutually disjoint with zero leakage.
4. `TestMission013EvaluationAccounting`: Enforces mathematical closure on evaluation metrics ($TP + FN = GT$, precision, recall, F1, lighting and class totals).
5. `TestMission013ManifestAndLicensing`: Enforces SHA-256 traceability, parameter counts, and license declarations.

**Full Test Suite Execution Result:**
- `tests/test_mission_013_domain_training.py`: **8 / 8 PASSED** (0.15s).
- Full Repository Regression Suite: **129 / 129 PASSED**, 0 failed, 1 warning (15.68s).

---

## 10. Final Engineering Decision & Next Steps

### 10.1 Engineering Decision
Based on the empirical evidence gathered in Mission 013:

> **RECOMMENDATION: Fine-Tuned YOLOv8m is designated as the LEADING EXPERIMENTAL CANDIDATE.**  
> **Track A (RT-DETR-L) is REJECTED for this domain adaptation regime.**

### Evidence Basis:
1. **Accuracy & Box Quality:** Fine-Tuned YOLOv8m achieved **0.8861 Precision**, **0.9722 Recall**, **0.9272 F1**, **0.9697 mAP50**, and **0.8884 mAP50-95** on `development_test_v1`, outperforming all other configurations by a wide margin.
2. **False Alarm Rejection:** Slashed false positives from 2.38 FP/frame (RT-DETR baseline) and 0.76 FP/frame (YOLO baseline) down to **0.125 FP/frame** (only 9 false alarms across 72 test frames).
3. **Operational Speed:** Runs at **18.42 ms p50 latency (~51 FPS)** on an RTX 4060 GPU, compared to RT-DETR-L's 45.95 ms (~21 FPS), fitting comfortably within real-time edge processing constraints.
4. **Failure of RT-DETR-L:** RT-DETR-L demonstrated severe architectural sensitivity to low-sample domain adaptation, suffering optimization collapse and failing to meet basic recall criteria (5.56%).

### 10.2 Explicit Pre-Production Caveats
- **No Production Certification:** This milestone is strictly an internal domain adaptation experiment.
- **Dependency on `external_test_v2`:** Model generalization capability must be independently validated on the upcoming `external_test_v2` dataset (collected across distinct temporal days, weather conditions, and camera angles) before any deployment recommendation can be made.
- **Truck Domain Constraint:** The model is certified strictly as an ingress heavy-commercial vehicle detector; it must not be deployed as an autonomous security detector for passenger vehicles, motorcycles, or pedestrians without dedicated retraining on balanced domain datasets.
