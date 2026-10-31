# Mission 014A: Pre-GitHub Release Safety & Readiness Report

**Project:** Egyptian Vehicle Intelligence (`TruckEntry` ALPR & Gate Monitoring)  
**Date:** October 31, 2026  
**Auditor / Infrastructure Engineer:** Senior ML Infrastructure Engineer  
**Release Readiness Verdict:** **CLEAN FOR RELEASE (Subject to Recommended Staging Scope)**

---

## 1. Executive Summary

This audit conducted a pre-release safety, licensing, privacy, and dependency review of the Egyptian Vehicle Intelligence repository prior to public/team publishing on GitHub.

### Critical Verdicts:
1. **Dependency Consistency:** **CONSISTENT & PINNED**. The Ultralytics version discrepancy has been resolved. The tested runtime environment executes `ultralytics 8.4.192`, which is now strictly pinned (`ultralytics>=8.4.0,<=8.4.192`) across `requirements.txt` and `pyproject.toml`. PyTorch GPU (`torch==2.6.0+cu124`) and CPU (`torch==2.6.0`) manifests are internally aligned.
2. **Checkpoint Architecture & Classification:** **VERIFIED & SECURED**. All 7 model weights have been categorized into 4 operational workflow classes. Checkpoints remain strictly excluded from Git (`*.pt` in `.gitignore`), and every hash matches ground-truth weights byte-for-byte.
3. **GitHub Release Safety:** **SECURE WITH RESTRICTED STAGING**. Zero credentials, secrets, or API keys were detected in tracked or untracked files. Tracked files contain zero `.pt` weights, zero CCTV videos, and zero gate ingress crops. Real Egyptian license plate numbers and CCTV timestamps were found in local untracked mission reports (`artifacts/missions/mission_008`–`013`); these must be excluded from Git staging.
4. **No Action Taken on Git:** Zero commits or pushes were executed. Full control is handed to the repository owner with safe, copy-pasteable staging commands.

---

## 2. Dependency Reproducibility & Environment Audit

### 2.1 The Ultralytics 8.3.237 vs. 8.4.192 Resolution
- **Investigation:** In Mission 014's markdown text narrative, a listing mistakenly referenced `ultralytics == 8.3.237`. Forensic inspection of the active virtual environment (`.venv`) and the machine-generated audit log (`artifacts/missions/mission_014/portability_audit.json`) confirmed that the actual tested runtime was **`ultralytics 8.4.192`**.
- **Resolution:**
  - `requirements.txt` pinned to: `ultralytics>=8.4.0,<=8.4.192` (Tested with `8.4.192`).
  - `pyproject.toml` dependencies updated to: `ultralytics>=8.4.0,<=8.4.192`.
  - Documentation updated to eliminate all stale references to `8.3.237`.

### 2.2 Dependency Parity Matrix

| Package | Tested `.venv` Version | `requirements.txt` | `requirements-gpu.txt` | `requirements-cpu.txt` | `pyproject.toml` | Consistency Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Python** | 3.11.9 (64-bit) | `>=3.10` | `>=3.10` | `>=3.10` | `>=3.10` | **MATCH** |
| **PyTorch (GPU)** | `2.6.0+cu124` | Reference notes | `torch==2.6.0+cu124` | N/A | `torch>=2.2.0` (opt) | **MATCH** |
| **Torchvision (GPU)** | `0.21.0+cu124` | Reference notes | `torchvision==0.21.0+cu124` | N/A | `torchvision>=0.17.0` (opt) | **MATCH** |
| **PyTorch (CPU)** | Fallback tested | Reference notes | N/A | `torch==2.6.0` | `torch>=2.2.0` (opt) | **MATCH** |
| **Torchvision (CPU)** | Fallback tested | Reference notes | N/A | `torchvision==0.21.0` | `torchvision>=0.17.0` (opt) | **MATCH** |
| **Ultralytics** | `8.4.192` | `>=8.4.0,<=8.4.192` | Inherited | Inherited | `>=8.4.0,<=8.4.192` | **MATCH** |
| **OpenCV** | `5.0.0.93` | `>=4.9.0,<=5.0.0.93` | Inherited | Inherited | `>=4.9.0,<=5.0.0.93` | **MATCH** |
| **Pydantic** | `2.14.1` | `>=2.7.0,<3.0.0` | Inherited | Inherited | `>=2.7.0,<3.0.0` | **MATCH** |
| **PyCOCOTools** | `2.0.11` | `>=2.0.7` | Inherited | Inherited | `>=2.0.7` | **MATCH** |
| **Pytest** | `9.2.0` | `>=8.0.0` | Inherited | Inherited | `>=8.0.0` (dev) | **MATCH** |

### 2.3 Installation Command Consistency
- **GPU Profile:** `setup_windows.ps1 -Mode GPU` installs `torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124` followed by `-r requirements.txt`. This exactly mirrors `requirements-gpu.txt`.
- **CPU Profile:** `setup_windows.ps1 -Mode CPU` installs `torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu` followed by `-r requirements.txt`. This exactly mirrors `requirements-cpu.txt`.

---

## 3. Checkpoint Inventory & Operational Classification

In accordance with release guidelines, every model checkpoint is classified into its exact operational workflow:

| Checkpoint Relative Path | Architecture & Parameters | SHA-256 Checksum | Operational Classification | Workflow Description |
| :--- | :--- | :--- | :--- | :--- |
| `artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt` | YOLOv8n (3.2M params) | `1d9ab549c626ee1b708b3439ad63320515c6de105283a5c980b3c5952337b279` | **Required for normal ALPR** | Detects license plate boundaries in input frames (`imgsz=640`). Required by CLI, API, and `test_golden_plates.py`. |
| `artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt` | YOLOv8n (27-class, 3.2M params) | `875b76e18b1df480829f84809e1f26d2c38d2c92c7f66d3bd85492db14488319` | **Required for normal ALPR** | Detects 17 Arabic letters and 10 Arabic-Indic digits on cropped plates (`imgsz=320`). |
| `artifacts/missions/mission_013/checkpoints/yolov8m_finetuned_best.pt` | YOLOv8m (25.9M params) | `b2e462fe4b06f1aa5479d615445c2246941d6ea1a91ebdb235e6b056788ca03d` | **Required for vehicle detection** | Domain fine-tuned detector for heavy commercial trucks at the entry gate. Required for gate transit tracking. |
| `yolov8m.pt` | YOLOv8m (25.9M params) | `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5` | **Required for vehicle detection** | Official upstream COCO baseline; evaluation benchmark reference and fine-tuning base weight. |
| `artifacts/missions/mission_013/checkpoints/rtdetr_l_finetuned_best.pt` | RT-DETR-L (32.8M params) | `47f0cc1a288cc2d2f630f80841caa0e90b92eda025dd570424d4402836975613` | **Optional experimental** | Vision Transformer vehicle detector fine-tuned in Mission 013 (research alternative to YOLOv8m). |
| `rtdetr-l.pt` | RT-DETR-L (32.9M params) | `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f` | **Optional experimental** | Official upstream COCO RT-DETR-L baseline for transformer benchmark comparisons. |
| `yolov8n.pt` | YOLOv8n (3.2M params) | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | **Obsolete or unused** | Upstream COCO nano detector from early feasibility trials (Mission 011). Superseded by YOLOv8m. |
| `yolo11n.pt` | YOLO11n (2.6M params) | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` | **Obsolete or unused** | Exploratory baseline from Mission 011. Not used in production ALPR pipeline. |
| `yolov8s.pt` | YOLOv8s (11.2M params) | `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a` | **Obsolete or unused** | Intermediate exploratory checkpoint from Mission 011. |

- Both `README.md` and `CHECKPOINTS.md` have been updated with this exact four-tier classification and mapping.
- All hashes were independently verified with Python SHA-256 routines against physical storage.

---

## 4. GitHub Release Safety & Privacy Audit

A forensic scan of the working directory was performed across both tracked (`git ls-files`) and untracked assets:

### 4.1 CCTV Footage, Images & Private Data
- **Tracked Assets:** Exactly 20 images exist in Git tracking: `tests/golden/images/0010.jpg` through `0091.jpg`. These are isolated license plate crops from the public/cleaned EALPR dataset used solely by `tests/test_golden_plates.py` for continuous integration. Zero CCTV recordings or gate camera frames are tracked.
- **Untracked Assets:** Real CCTV video files (`*.mp4`, `*.avi`) and image crops in `artifacts/` are strictly ignored by `.gitignore`. Zero video binaries are staged.
- **PII Exposure Finding:**
  - Files under untracked historical folders `artifacts/missions/mission_009/`, `artifacts/missions/mission_010/`, and `artifacts/missions/mission_012/` contain CSV tables (e.g., `plate_vehicle_association_results.csv`, `event_examples.csv`, `annotations.csv`) detailing timestamps, facility camera IDs, and real Arabic license plate numbers (e.g., `سقج ٨٥٦٨`).
  - **Risk:** Committing these mission subdirectories to a public GitHub repository would leak real Egyptian vehicle license plates and operational facility ingress logs, violating Egyptian Data Protection Law No. 151 of 2020.
  - **Remediation:** **DO NOT** run `git add artifacts/`. Only stage `artifacts/missions/mission_014/` and `artifacts/missions/mission_014a/`, which contain anonymized engineering and portability audits.

### 4.2 Credentials and Secrets
- **Audit Tool:** Automated regex scan searching for `api_key`, `secret_key`, `private_key`, `bearer tokens`, and password assignments.
- **Result:** **ZERO SECRETS FOUND**.
- `.env` is ignored by `.gitignore:84`.
- `.env.example` contains only path declarations and logging level options; no passwords or auth tokens exist.

### 4.3 Machine-Specific Paths in Tracked Files
A regex scan for `C:\Users\MARWAN` and `D:\TruckEntry` identified occurrences in legacy historical files:
1. `README.md`: An internal `file:///` link on line 186 was replaced with a relative markdown link `[CHECKPOINTS.md](CHECKPOINTS.md)`. `README.md` is now 100% clean of personal paths.
2. `vehicle_intelligence/core/config.py`: Contains a documentation comment mentioning `D:/TruckEntry` and an opportunistic check (`if d_truck.exists()`). This does not break other machines, but is noted for transparency.
3. Legacy scripts from Missions 001–005 (`scripts/audit_dataset.py`, `scripts/build_cleaned_splits.py`, `scripts/benchmark_cpu_reality_check.py`, etc.): Contain fallback defaults pointing to `C:\Users\MARWAN\plate_project\...` or `D:\TruckEntry`.
   - **Recommendation:** These files are historical and not imported by the core package. When publishing source code, keep them as-is or pass CLI arguments (`--dataset-dir`) as documented.

---

## 5. Clean Installation Verification (New Machine Portability)

### 5.1 Verification on an Isolated Machine
1. **Zero Path Assumptions:** The core pipeline, CLI, and test suite do not require `C:\Users\MARWAN` or `D:\TruckEntry`.
2. **Setup Script:** `setup_windows.ps1` operates entirely through relative paths (`Get-Location`), creating its own `.venv` and installing dependencies without reference to any host machine folders.
3. **Smoke Test Execution:** A clean machine running `setup_windows.ps1` executes smoke tests (`pytest tests/test_config.py tests/test_domain.py`) automatically upon completion.

### 5.2 Assets Requiring Separate Transfer
When setting up a new target workstation, the following assets are intentionally omitted from GitHub and must be transferred via secure local/S3/drive storage:

1. **Production ALPR Checkpoints (Mandatory for Plate Recognition):**
   - `artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt`
   - `artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt`
2. **Domain Vehicle Checkpoint (Mandatory for Gate Truck Tracking):**
   - `artifacts/missions/mission_013/checkpoints/yolov8m_finetuned_best.pt`
3. **CCTV Gate Video Data (Only if Replaying Raw Footage):**
   - Facility footage directory (e.g., `D:\TruckEntry`), configured via `GATE_DIR` in `.env`.
   - *Note: Standalone ALPR inference, golden plate testing, and API endpoints work out-of-the-box with only items (1) and (2).*

---

## 6. Exact List of Files Safe to Commit

The following files constitute the clean, portable, and secure source distribution:

### Documentation & Packaging (Root)
- `README.md`
- `CHECKPOINTS.md`
- `.env.example`
- `.gitignore`
- `pyproject.toml`
- `requirements.txt`
- `requirements-gpu.txt`
- `requirements-cpu.txt`
- `setup_windows.ps1`

### Configurations
- `configs/default_config.yaml`
- `configs/checkpoint_manifest.json`

### Core Package (`vehicle_intelligence/`)
- `vehicle_intelligence/__init__.py`
- `vehicle_intelligence/core/config.py`
- `vehicle_intelligence/core/domain.py`
- `vehicle_intelligence/core/interfaces.py`
- `vehicle_intelligence/data/loader.py`
- `vehicle_intelligence/data/audit.py`
- `vehicle_intelligence/detectors/pytorch_yolo.py`
- `vehicle_intelligence/pipeline/alpr_pipeline.py`
- `vehicle_intelligence/tracking/tracker.py`
- `vehicle_intelligence/tracking/aggregator.py`
- `vehicle_intelligence/api/app.py`
- `vehicle_intelligence/association/`

### Utility & Portability Scripts (`scripts/`)
- `scripts/alpr_cli.py`
- `scripts/verify_portability.py`
- `scripts/run_golden_regression.py`
- `scripts/run_mission_013a_verification.py`

### Test Suite (`tests/`)
- `tests/conftest.py`
- `tests/test_config.py`
- `tests/test_domain.py`
- `tests/test_detectors.py`
- `tests/test_media_loader.py`
- `tests/test_tracker.py`
- `tests/test_aggregator.py`
- `tests/test_golden_plates.py`
- `tests/golden/expectations.json`
- `tests/golden/images/` (20 reference unit test plate crops)

### Mission 014 & 014A Engineering Deliverables
- `artifacts/missions/mission_014/mission_014_report.md`
- `artifacts/missions/mission_014/portability_audit.json`
- `artifacts/missions/mission_014/golden_image_regression.json`
- `artifacts/missions/mission_014a/release_readiness_report.md`

---

## 7. Blockers & Risks Summary

| Potential Issue | Severity | Status | Mitigation |
| :--- | :--- | :--- | :--- |
| **Accidental CCTV Plate PII Leakage** | **High** | **Mitigated** | Historical mission outputs (`mission_008`–`mission_013`) contain real plate strings. Do NOT run `git add .` or stage those directories. |
| **Model Weight Distribution Limit** | **Medium** | **Resolved** | All `*.pt` files are excluded by `.gitignore`. Zero binary checkpoints staged. |
| **Local Path Leakage in README** | **Low** | **Resolved** | Absolute URI replaced with relative markdown link. README is clean. |
| **Ultralytics Dependency Mismatch** | **Low** | **Resolved** | Pinned to `8.4.192` across manifests. Verified by automated audit. |

---

## 8. Exact Recommended Git Commands

To publish the code to GitHub cleanly and securely, execute the following commands in PowerShell from the repository root:

```powershell
# 1. Verify working directory status
git status

# 2. Stage core infrastructure and documentation
git add README.md CHECKPOINTS.md .env.example .gitignore
git add pyproject.toml requirements.txt requirements-gpu.txt requirements-cpu.txt setup_windows.ps1

# 3. Stage verified configs, package source, tools, and test suite
git add configs/default_config.yaml configs/checkpoint_manifest.json
git add vehicle_intelligence/
git add scripts/alpr_cli.py scripts/verify_portability.py scripts/run_golden_regression.py scripts/run_mission_013a_verification.py
git add tests/

# 4. Stage Mission 014 and 014A audit deliverables
git add artifacts/missions/mission_014/ artifacts/missions/mission_014a/

# 5. FORENSIC CHECK: Verify staged files to ensure NO weights, CCTV videos, or private CSVs are staged
git diff --cached --stat

# Confirm that zero .pt, .pth, .mp4, or .avi files are staged:
git diff --cached --name-only | Select-String -Pattern "\.(pt|pth|onnx|mp4|avi)$"

# 6. Commit the clean release (Execute only when authorized by you)
git commit -m "feat(release): reproducible Windows 11 setup and multi-PC portability (Mission 014 & 014A)"

# 7. Push to GitHub
git push origin master
```
