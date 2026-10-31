# Mission 013: Experimental Limitations and Domain Scope

**Milestone:** MISSION 013: Exploratory Vehicle Domain Fine-Tuning  
**Target Domain:** `D:\TruckEntry` (Egyptian Barrier-Gate Vehicle Intelligence System)  
**Date:** 2026-10-30  
**Classification:** Exploratory Experimental Milestone (Pre-Production)

---

## 1. Scope & Objective Statement

This document records the empirical constraints, data boundaries, and operational limitations identified during Mission 013 domain fine-tuning. The experiments conducted herein represent an exploratory domain-adaptation milestone on the fixed barrier-gate camera feed at `D:\TruckEntry`.

**Under no circumstances should the results in Mission 013 be interpreted as deployment readiness, production certification, or guaranteed real-world generalization across arbitrary Egyptian roadside or toll environments.**

---

## 2. Dataset Boundaries & Extreme Class Imbalance

### 2.1 Empirical Distribution
The experimental dataset comprises 340 frames partitioned into three temporally isolated, mutually disjoint splits:
- **Training Set:** 196 frames, 175 temporal events, 200 vehicle instances.
- **Validation Set:** 72 frames, 58 temporal events, 72 vehicle instances.
- **Development Test Set (`development_test_v1`):** 72 frames, 58 temporal events, 72 vehicle instances.

The distribution of vehicle classes across the splits reveals severe imbalance:

| Vehicle Class | Training Count | Validation Count | Dev Test Count | Overall % |
| :--- | :---: | :---: | :---: | :---: |
| **Heavy Commercial Truck (`heavy_truck`)** | **195** (97.5%) | **70** (97.2%) | **71** (98.6%) | **97.7%** |
| Passenger Car (`passenger_car`) | 4 (2.0%) | 1 (1.4%) | 1 (1.4%) | 1.7% |
| Motorcycle (`motorcycle`) | 1 (0.5%) | 0 (0.0%) | 0 (0.0%) | 0.3% |
| Light Commercial (`light_commercial`) | 0 (0.0%) | 1 (1.4%) | 0 (0.0%) | 0.3% |
| **Total** | **200** | **72** | **72** | **100.0%** |

### 2.2 Operational Consequence
1. **Domain Adaptation is Valid Only for Heavy Commercial Ingress:** The fine-tuning process adapts model priors almost exclusively to large heavy trucks (dump trucks, flatbeds, multi-axle trailers).
2. **Zero Generalization to Rare Classes:** The experiment provides **no empirical evidence** of reliable passenger-car, motorcycle, or light-commercial detection. The models must not be assumed capable of detecting general traffic.

---

## 3. Deep-Night IR Sample Scarcity

### 3.1 Empirical Breakdown
The lighting condition distribution in the evaluation splits is as follows:
- **Daylight:** 44 frames (61.1% of dev-test)
- **Evening IR:** 26 frames (36.1% of dev-test)
- **Deep-Night IR:** 2 frames (2.8% of dev-test, exactly 2 vehicle instances)

### 3.2 Statistical Power Limitation
While the deep-night IR captures represent the most challenging operational regime (monochrome sensor noise, harsh headlights, extreme bloom, deep pitch-black background), an evaluation sample of $n = 2$ instances has negligible statistical power. A single false positive or miss alters the metric by 50 percentage points.
- **Yield can be measured:** Both models successfully trigger on the 2 night frames.
- **Generalization cannot be asserted:** Claiming robust nighttime detection based on 2 frames would be statistically invalid. Independent validation on an expanded nighttime dataset is mandatory before deployment.

---

## 4. Vehicle Truncation & Framing Geometry

### 4.1 Ingress Gate Geometry
The camera is mounted at an elevated overhead angle (~45°) immediately adjacent to the barrier arm and security booth. As vehicles approach the gate:
- **Frame Truncation:** In `development_test_v1`, 71 out of 72 vehicles (98.6%) are partially truncated by the image boundaries (cabs entering the frame, tail ends leaving the frame, or side panels clipped).
- **Extreme Scale Variations:** Approaching trucks occupy up to 60-80% of the vertical frame dimension, whereas entering trailers or distant vehicles occupy substantially smaller pixel areas.

### 4.2 Impact on Bounding-Box Quality
Pretrained COCO detectors are biased toward fully visible, canonical vehicle bodies in typical street-level perspective. Under the 45° overhead barrier ingress:
- Pretrained models frequently split a single articulated truck into multiple partial detections (e.g., cab as "truck", trailer as separate "truck" or "bus").
- Fine-tuning successfully consolidates vehicle bounds, but boundary placement on severely clipped cabs exhibits higher variance than on centered vehicles.

---

## 5. False Positive and Structural Distractor Characteristics

### 5.1 Overhead Structural Distractors
The camera field of view includes static high-contrast infrastructure:
- Concrete guard barriers and curbs.
- The barrier gate housing, mechanical arm, and counterweight.
- The security booth roof and corrugated metal panels.
- Overhead lighting stanchions and cabling.

### 5.2 Model Vulnerability
- **Pretrained RT-DETR-L** suffered extreme false-positive vulnerability (2.29 false positives per frame on dev-test), as its Hungarian matcher and multi-scale transformer queries hallucinated "trucks" and "cars" on the gate booth, perimeter walls, and barrier rails.
- Fine-tuning on target domain negatives and positives suppresses these structural false alarms, but operational confidence thresholds must be carefully maintained.

---

## 6. Annotation Provenance and Inter-Annotator Reliability

1. **Single-Annotator Ground Truth:** Ground truth annotations (340 frames, 344 instances) were established by a single independent annotator during Mission 012 and verified for temporal integrity in Missions 012B and 012C.
2. **Absence of Inter-Annotator Reliability Metrics:** Because annotations were produced by a single human annotator, inter-coder agreement statistics (Cohen's $\kappa$ or Fleiss' $\kappa$) could not be computed. Ground truth box boundaries reflect one consistent human annotator's interpretation of truncated boundaries.
3. **Ground Truth Preservation:** Ground truth was strictly preserved and never modified, smoothed, or regenerated based on model inferences.

---

## 7. Inference Latency & Hardware Constraints

1. **Hardware Profile:** All latency profiles were measured on an NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) under CUDA 12.4 and PyTorch 2.6.
2. **Computational Footprint:**
   - **RT-DETR-L:** ~32.8M parameters, ~110 GFLOPs. Measured latency: p50 ~ 54.6 ms (~18 FPS).
   - **YOLOv8m:** ~25.9M parameters, ~79 GFLOPs. Measured latency: p50 ~ 17.9 ms (~55 FPS).
3. **Deployment Feasibility:** While RT-DETR-L fits within the 8GB VRAM envelope (~3.5GB peak training, ~0.3GB inference), its 54 ms latency is triple that of YOLOv8m. On embedded edge hardware (e.g. Jetson Orin Nano / edge PC), RT-DETR-L would require TensorRT INT8 optimization to achieve real-time gate frame rates (>25 FPS).

---

## 8. Deployment Conclusion

Mission 013 establishes that conservative domain transfer learning substantially improves vehicle detection yield, bounding-box precision, and structural false-alarm rejection on `D:\TruckEntry`.

However:
1. **No Production Model is Declared:** Production validation remains strictly contingent on the independently collected, fully unseen `external_test_v2` dataset.
2. **Operational Scope:** The current models are specialized heavy-vehicle gate ingress detectors and must be paired with downstream plate association logic rather than relied upon as autonomous security agents.
