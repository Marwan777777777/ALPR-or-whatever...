# Licensing & Commercial Deployment Audit: Vehicle Detection Models

## Executive Summary

Mission 011 requires a strict legal and architectural assessment of candidate vehicle detection models. Introducing external neural network checkpoints or third-party computer vision libraries into an enterprise barrier-gate / ALPR product can create legal liability, copyright infringement risks, or forced open-source copyleft disclosure.

This document analyzes the code, weights, and runtime licenses of all candidate vehicle detectors considered in Mission 011.

---

## 1. Candidate Model Licensing Comparison

| Model Architecture | Repository / Author | Code License | Pretrained Weights License | Commercial SaaS Compatibility | Copyleft Risk Level |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **YOLOv8** (v8n, v8s, v8m, v8x) | Ultralytics Inc. | **AGPL-3.0** | AGPL-3.0 / Commercial | **CONDITIONAL** (Requires Enterprise License) | **CRITICAL (High)** |
| **YOLO11** (11n, 11s, 11m) | Ultralytics Inc. | **AGPL-3.0** | AGPL-3.0 / Commercial | **CONDITIONAL** (Requires Enterprise License) | **CRITICAL (High)** |
| **RT-DETR** (rtdetr-l, rtdetr-x) | Baidu / PaddlePaddle / Ultralytics wrapper | **Apache-2.0** (Upstream) | Apache-2.0 | **YES** (Permissive Open Source) | **LOW (Safe)** |
| **YOLOv10** | THU-MIG (Tsinghua University) | **Apache-2.0** | Apache-2.0 | **YES** (Permissive Open Source) | **LOW (Safe)** |
| **Custom Domain-Trained Weights** | In-House / Proprietary | Proprietary | Proprietary (Trained on Egyptian Gate Imagery) | **YES** (Full Commercial Ownership) | **ZERO (Safe)** |

---

## 2. Detailed Legal Analysis by Architecture

### 2.1 Ultralytics YOLOv8 & YOLO11 (AGPL-3.0)
- **Legal Structure:** The open-source code and official weights published by Ultralytics are licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
- **The Network Trigger (Section 13):**
  Unlike the standard GPL, AGPL-3.0 explicitly states that if the software is run on a server and accessed by users over a network (e.g. REST API, RTSP microservice, barrier-gate cloud telemetry, or SaaS dashboard), the **complete source code of the entire interacting application must be made available** under AGPL-3.0 to all network users.
- **Commercial Impact:**
  Deploying YOLOv8 or YOLO11 in a closed-source commercial gate-management product without an Ultralytics Enterprise Commercial License is a direct license violation that would legally compel the company to open-source its proprietary Egyptian ALPR algorithms, reading order heuristics, database schemas, and API code.
- **Mitigation:**
  Ultralytics models are retained solely for internal research, non-commercial benchmarking, and validation. In production, commercial deployments must either purchase an Ultralytics commercial exception or utilize permissively licensed models.

### 2.2 Baidu RT-DETR (Apache-2.0)
- **Legal Structure:**
  RT-DETR was developed by Baidu and released under the **Apache License 2.0**.
- **Commercial Permissions:**
  Apache 2.0 explicitly grants:
  - Commercial use without royalties.
  - Distribution without disclosing proprietary source code.
  - Modification and private internal deployment.
  - Patent grant protections against upstream patent claims.
- **Attribution Obligation:**
  Retain original copyright and license notices in third-party attribution documentation. No copyleft obligations apply to the host application or backend databases.

### 2.3 YOLOv10 (Apache-2.0)
- **Legal Structure:**
  Developed by Tsinghua University's Machine Intelligence Group (THU-MIG) and released under **Apache-2.0**.
- **Commercial Status:**
  Safe for proprietary commercial embedding, ONNX export, and TensorRT runtime acceleration.

---

## 3. Repository Architectural Safeguards

To prevent license contamination across repository layers:

1. **Interface Isolation via `BaseVehicleDetector`:**
   The entire application (tracking, plate association, OCR, REST API, database storage) depends strictly on the abstract interface `BaseVehicleDetector` defined in [interfaces.py](file:///c:/Users/MARWAN/computer%20vision/vehicle_intelligence/core/interfaces.py). No higher-level module imports `ultralytics` or model-specific dependencies directly.
2. **Transparent Licensing Introspection:**
   `PyTorchYoloVehicleDetector` implements `get_licensing_info()`:
   ```python
   info = detector.get_licensing_info()
   # Returns: {"architecture": "...", "license": "Apache-2.0" | "AGPL-3.0", "commercial_compatible": "YES" | "NO"}
   ```
   This prevents silent deployment of AGPL-3.0 weights in commercial production builds.
3. **Domain-Specific Training Strategy:**
   When fine-tuning on Egyptian gate imagery (Section 5 of Mission 011), the training pipeline will target **RT-DETR or YOLOv10 backbones**, producing 100% proprietary checkpoint weights owned entirely by the product team with zero copyleft contamination.
