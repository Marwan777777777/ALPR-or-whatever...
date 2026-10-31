# Component-Level Intellectual Property & Licensing Matrix

**Document Purpose:** Objective legal-engineering audit of computer vision codebases, neural network checkpoints, runtime dependencies, export runtimes, and training datasets.  
**Disclaimer:** This technical audit represents software engineering license inspection and due diligence. It does NOT constitute formal legal advice or legal guarantees. Commercial deployment decisions should be verified by corporate legal counsel.

---

## 1. Executive Licensing Summary & Principles

1. **Separation of Layers:** A deep learning deployment consists of multiple distinct legal layers:
   - **Source Code License:** The terms governing the Python/C++ code executing the inference or training.
   - **Checkpoint / Weight Terms:** The license governing the neural network weight values, which may be derived from third-party datasets (such as COCO, ImageNet, or proprietary crawls).
   - **Dataset Rights:** The license or data collection consent governing training and evaluation imagery.
   - **Runtime & Export Tooling:** The licenses of underlying inference engines (PyTorch, ONNX Runtime, TensorRT, CUDA).
2. **Correction of Past Overstatements:** Mission 011 stated that "RT-DETR eliminates viral SaaS licensing liability." This statement is legally imprecise and is hereby formally qualified: While upstream Baidu PaddleDetection RT-DETR code is published under Apache-2.0 (permissive), running RT-DETR through the `ultralytics` Python package invokes Ultralytics' codebase, which remains under AGPL-3.0 unless enterprise-licensed. A clean commercial implementation requires using the standalone native Apache-2.0 RT-DETR repository (or an Apache-2.0 export) without importing AGPL packages.

---

## 2. Component-Level Licensing Matrix

| System Component | Upstream Source & Repository | Commit / Version Tested | Checkpoint Identity & SHA-256 | Code License | Weight License & Training Provenance | Commercial Deployment Status | Unresolved Legal / Technical Questions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n** (Detector Baseline) | [Ultralytics GitHub](https://github.com/ultralytics/ultralytics) | `ultralytics==8.3.82` | `yolov8n.pt`<br>`f59b3d833e2ff32e` | **AGPL-3.0** | AGPL-3.0 (Ultralytics Pretrained on COCO subset) | **CONDITIONAL** (Requires Commercial Exception) | AGPL Section 13 Network Trigger applies if exposed via API / microservice. Cannot be embedded in proprietary commercial binary without Ultralytics commercial license. |
| **YOLOv8m** (Detector Medium) | [Ultralytics GitHub](https://github.com/ultralytics/ultralytics) | `ultralytics==8.3.82` | `yolov8m.pt`<br>`5d4a90cdc7a21786` | **AGPL-3.0** | AGPL-3.0 (Ultralytics Pretrained on COCO subset) | **CONDITIONAL** (Requires Commercial Exception) | Same viral copyleft risks as YOLOv8n. Closed-source SaaS deployment requires enterprise license purchase. |
| **RT-DETR-L** (Ultralytics Wrapper) | [Ultralytics GitHub](https://github.com/ultralytics/ultralytics) | `ultralytics==8.3.82` | `rtdetr-l.pt`<br>`6de60b10d4bc566f` | **AGPL-3.0** (Ultralytics Wrapper Code) | Apache-2.0 (Baidu weights re-packaged) | **CONDITIONAL** (AGPL applies to Ultralytics wrapper) | Running Baidu's weights through `from ultralytics import YOLO` subjects execution to Ultralytics AGPL-3.0 license. Must use native Baidu code or direct ONNX export to claim Apache-2.0. |
| **RT-DETR-L** (Native Baidu Upstream) | [PaddleDetection RT-DETR](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rtdetr) | `PaddleDetection v2.6` | Official Baidu Checkpoint (`rtdetr_r50vd_6x_coco.pdparams`) | **Apache-2.0** | Apache-2.0 (Pretrained on COCO, permissive) | **COMMERCIALLY COMPATIBLE** (Permissive) | Requires PaddlePaddle or custom PyTorch/ONNX runtime without importing Ultralytics. Full commercial rights, private modification permitted without source disclosure. |
| **YOLOv10** (Candidate Permissive YOLO) | [THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10) | `release v1.0` (Tsinghua University) | `yolov10n.pt` / `yolov10s.pt` | **Apache-2.0** | Apache-2.0 (Permissive COCO weights) | **COMMERCIALLY COMPATIBLE** (Permissive) | Developed by Tsinghua University under Apache-2.0. Must verify that no GPL/AGPL submodules are imported during training or inference. |
| **Production Plate Detector** | In-House (`artifacts/runs/detect/ealpr_cleaned_yolov8n/`) | Model trained on EALPR cleaned split | `best.pt` (PyTorch checkpoint) | Architecture code depends on runtime | Trained weights owned by repository owner; training data based on EALPR dataset | **INTERNAL RESEARCH SAFE**; commercial runtime depends on engine | EALPR dataset licensing: EALPR is an academic dataset; commercial distribution requires checking EALPR author terms of use or replacing with 100% proprietary gate data. |
| **Production Character Detector** | In-House (`artifacts/runs/detect_chars/ealpr_characters_yolov8n/`) | Model trained on 27 Egyptian classes | `best.pt` (PyTorch checkpoint) | Architecture code depends on runtime | Trained weights owned by repository owner | **INTERNAL RESEARCH SAFE**; same runtime and dataset considerations as plate model | Same as above. Character vocabulary is proprietary to Egyptian plates. |
| **PyTorch Runtime** | [PyTorch Foundation](https://github.com/pytorch/pytorch) | `torch==2.6.0+cu124` | Binary distribution | **Modified BSD (3-Clause BSD)** | Open Source (Permissive) | **COMMERCIALLY COMPATIBLE** | Safe for commercial use and binary redistribution with standard attribution notice. |
| **OpenCV Runtime** | [OpenCV Foundation](https://github.com/opencv/opencv) | `opencv-python==4.10.0.84` | Binary wheel | **Apache-2.0** (OpenCV 4.5+) | Open Source (Permissive) | **COMMERCIALLY COMPATIBLE** | Free commercial use without royalties. Retain copyright notice. |
| **ONNX Runtime** | [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime) | Standalone engine | C++ / Python binaries | **MIT License** | Open Source (Permissive) | **COMMERCIALLY COMPATIBLE** | Extremely permissive. Ideal target for exporting trained checkpoints into embedded commercial runtime. |
| **NVIDIA TensorRT** | [NVIDIA Corporation](https://developer.nvidia.com/tensorrt) | Proprietary SDK | Platform binaries | **NVIDIA TensorRT License** | Proprietary Commercial | **COMMERCIALLY COMPATIBLE** | Permitted for commercial deployment on NVIDIA hardware. Requires accepting NVIDIA Software License Agreement. |
| **Egyptian Gate Dataset** (`D:\TruckEntry`) | In-House Facility Surveillance | 340 chronological captures (2026-10-15) | Raw JPEG captures + Mission 012 ground truth | **Proprietary** (Private Enterprise Asset) | Private Facility Asset | **FULL COMMERCIAL OWNERSHIP** | Proprietary enterprise surveillance asset recorded on facility premises. Zero third-party copyright restrictions. Complies with local data retention policy. |

---

## 3. Specific Analysis: Ultralytics AGPL-3.0 vs. Network Trigger (Section 13)

### The Legal Mechanism
The GNU Affero General Public License (AGPL-3.0) contains Section 13 ("Remote Network Interaction"):
> *"Notwithstanding any other provision of this License, if you modify the Program, your modified version must prominently offer all users interacting with it remotely through a computer network [...] a facility for obtaining the Corresponding Source of your version through some standard or customary means of facilitating copying software..."*

### Practical Implication for Barrier-Gate ALPR
1. **Networked Gate Telemetry / API:** If our Egyptian ALPR system is exposed via a REST API, web dashboard, or RTSP streaming microservice that communicates with remote gate controllers, AGPL-3.0 is triggered. The company would be legally obligated to release the **complete source code of the entire interacting application** (including proprietary Arabic character reading order heuristics, database models, and tracking logic) under AGPL-3.0.
2. **Current Repository Status:** In this engineering phase, Ultralytics is utilized as a research prototyping library. To achieve a legally defensible commercial deployment:
   - **Option A:** Retain Ultralytics and purchase a commercial enterprise license exception from Ultralytics Inc.
   - **Option B (Recommended):** Train using permissively licensed backbones (Baidu Apache-2.0 RT-DETR or Tsinghua Apache-2.0 YOLOv10) and export final models to **ONNX Runtime (MIT)** or **TensorRT (Proprietary/Commercial)**, removing Ultralytics entirely from production runtime dependencies.

---

## 4. Pretrained Weights vs. Training Data Lineage

Even if code is permissively licensed (e.g. Apache-2.0), pretrained checkpoints inherit usage restrictions from the datasets on which they were originally trained:
- **COCO Pretrained Weights:** Microsoft COCO dataset annotations are licensed under Creative Commons Attribution 4.0 (CC BY 4.0), which permits commercial use with attribution. Checkpoints pretrained on standard COCO (such as official Baidu RT-DETR weights) are commercially usable.
- **Academic Dataset Restrictions:** If any model is fine-tuned on non-commercial academic datasets (e.g. academic datasets prohibiting commercial use), the resulting checkpoint cannot be used in a commercial product.
- **Clean In-House Gate Weights:** Checkpoints fine-tuned exclusively on our proprietary `D:\TruckEntry` gate captures inherit 100% proprietary ownership, providing maximum legal safety.

---

## 5. Architectural Safeguards Implemented

To enforce clean separation in code:
1. **Abstract Contract Isolation:** All client modules (ALPR pipeline, tracking, association, API) interact exclusively with `BaseVehicleDetector`. No higher-level business logic imports Ultralytics directly.
2. **Licensing Introspection:** `PyTorchYoloVehicleDetector.get_licensing_info()` transparently discloses checkpoint architecture, license, and commercial compatibility at runtime.
3. **Exploratory Fine-Tuning Strategy:** Mission 013 training will benchmark Apache-2.0 candidates to prepare an exportable, dependency-clean commercial production pipeline.
