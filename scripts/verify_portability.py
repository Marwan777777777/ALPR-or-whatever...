"""Portability & Environment Verification Script.

Audits repository setup on any target machine (Windows 11 / Linux):
1. Python runtime & platform encoding.
2. Core dependencies installation and versions.
3. PyTorch installation & hardware acceleration (CUDA GPU vs CPU fallback).
4. OpenCV Unicode path decoding & Arabic filename support.
5. System configuration validation (SystemConfig & default_config.yaml).
6. Model checkpoint existence, alternative path resolution, and SHA-256 integrity.
7. Golden regression dataset availability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8")


ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT_DIR / "configs" / "checkpoint_manifest.json"
CONFIG_PATH = ROOT_DIR / "configs" / "default_config.yaml"
GOLDEN_EXPECTATIONS = ROOT_DIR / "tests" / "golden" / "expectations.json"
GOLDEN_IMAGES_DIR = ROOT_DIR / "tests" / "golden" / "images"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class PortabilityAuditor:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.info: List[str] = []

    def audit_python_runtime(self) -> Dict[str, Any]:
        major = sys.version_info.major
        minor = sys.version_info.minor
        micro = sys.version_info.micro
        version_str = f"{major}.{minor}.{micro}"

        status = "PASS"
        if major < 3 or (major == 3 and minor < 10):
            status = "FAIL"
            self.errors.append(f"Python 3.10+ is required. Found Python {version_str}")
        elif minor != 11:
            self.warnings.append(f"Python 3.11 is recommended for production parity. Current: {version_str}")

        is_64bit = sys.maxsize > 2**32
        if not is_64bit:
            status = "FAIL"
            self.errors.append("64-bit Python architecture is required.")

        return {
            "version": version_str,
            "architecture": "64-bit" if is_64bit else "32-bit",
            "platform": platform.platform(),
            "status": status,
        }

    def audit_dependencies(self) -> Dict[str, Any]:
        required_packages = [
            ("numpy", "numpy"),
            ("cv2", "opencv-python"),
            ("PIL", "pillow"),
            ("pydantic", "pydantic"),
            ("pydantic_settings", "pydantic-settings"),
            ("yaml", "pyyaml"),
            ("dotenv", "python-dotenv"),
            ("pytest", "pytest"),
            ("ultralytics", "ultralytics"),
            ("torch", "torch"),
            ("torchvision", "torchvision"),
        ]

        installed = {}
        missing = []

        for mod_name, pkg_name in required_packages:
            try:
                mod = __import__(mod_name)
                ver = getattr(mod, "__version__", "installed")
                installed[pkg_name] = ver
            except ImportError:
                missing.append(pkg_name)

        if missing:
            self.errors.append(f"Missing core Python packages: {', '.join(missing)}. Run: pip install -r requirements.txt")

        return {
            "installed_count": len(installed),
            "missing_count": len(missing),
            "missing_packages": missing,
            "packages": installed,
            "status": "PASS" if not missing else "FAIL",
        }

    def audit_compute_device(self) -> Dict[str, Any]:
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram_gb = round(props.total_memory / (1024**3), 2)
                cuda_version = torch.version.cuda or "unknown"
                mode = "GPU_ACCELERATED"
            else:
                device_count = 0
                device_name = "CPU Only"
                vram_gb = 0.0
                cuda_version = "None"
                mode = "CPU_FALLBACK"
                self.warnings.append("CUDA is not available. System will execute inference in CPU fallback mode.")

            return {
                "mode": mode,
                "cuda_available": cuda_available,
                "cuda_version": cuda_version,
                "device_count": device_count,
                "device_name": device_name,
                "vram_gb": vram_gb,
                "status": "PASS",
            }
        except Exception as e:
            self.errors.append(f"Failed to inspect PyTorch hardware acceleration: {e}")
            return {"mode": "ERROR", "status": "FAIL", "error": str(e)}

    def audit_opencv_unicode_io(self) -> Dict[str, Any]:
        """Validates that OpenCV and MediaLoader correctly handle Windows non-ASCII / Unicode paths."""
        try:
            import cv2
            import numpy as np

            # Create synthetic test image
            synth_img = np.zeros((100, 100, 3), dtype=np.uint8)
            synth_img[20:80, 20:80] = [0, 255, 128]

            with tempfile.TemporaryDirectory() as tmp_dir:
                # Test Arabic filename on disk
                arabic_name = "اختبار_لوحة_ترخيص_١٢٣.jpg"
                test_path = Path(tmp_dir) / arabic_name

                # Write image safely using imencode
                success, buf = cv2.imencode(".jpg", synth_img)
                if not success:
                    raise RuntimeError("cv2.imencode failed to encode test image.")
                with open(test_path, "wb") as f:
                    f.write(buf.tobytes())

                # Test decoding via MediaLoader
                from vehicle_intelligence.pipeline.media_loader import MediaLoader
                loaded = MediaLoader.load_image(test_path)
                if loaded is None or loaded.shape != (100, 100, 3):
                    raise RuntimeError(f"Decoded image shape mismatch: {getattr(loaded, 'shape', None)}")

            return {
                "unicode_read_support": "PASS",
                "arabic_path_tested": arabic_name,
                "status": "PASS",
            }
        except Exception as e:
            self.errors.append(f"OpenCV Unicode path decoding check failed: {e}")
            return {"unicode_read_support": "FAIL", "error": str(e), "status": "FAIL"}

    def audit_configuration(self) -> Dict[str, Any]:
        try:
            from vehicle_intelligence.core.config import SystemConfig

            # 1. Default initialization
            default_cfg = SystemConfig()
            default_cfg.ensure_directories()

            # 2. YAML file initialization
            if CONFIG_PATH.is_file():
                yaml_cfg = SystemConfig.from_yaml(CONFIG_PATH)
                yaml_status = "LOADED_VALID"
            else:
                yaml_cfg = None
                yaml_status = "FILE_NOT_FOUND"
                self.warnings.append(f"Default configuration file missing at: {CONFIG_PATH}")

            # 3. Check for .env file
            env_file = ROOT_DIR / ".env"
            env_example = ROOT_DIR / ".env.example"
            has_env = env_file.is_file()
            has_env_example = env_example.is_file()

            if not has_env and has_env_example:
                self.info.append("Notice: Local '.env' not found. You can copy '.env.example' to '.env' to configure local paths.")

            # 4. Check gate footage directory resolution
            gate_dir = default_cfg.resolve_gate_dir()
            gate_status = "CONFIGURED" if (gate_dir and gate_dir.is_dir()) else "NOT_MOUNTED"
            if gate_status == "NOT_MOUNTED":
                self.info.append("Gate CCTV directory (GATE_DIR / D:/TruckEntry) is not mounted on this machine. Full gate replay benchmarks will require configuring GATE_DIR in .env.")

            return {
                "system_config_validation": "PASS",
                "yaml_config_status": yaml_status,
                "has_env_file": has_env,
                "has_env_example": has_env_example,
                "gate_footage_dir": str(gate_dir) if gate_dir else None,
                "gate_dir_status": gate_status,
                "status": "PASS",
            }
        except Exception as e:
            self.errors.append(f"Configuration validation failed: {e}")
            return {"system_config_validation": "FAIL", "error": str(e), "status": "FAIL"}

    def audit_checkpoints(self) -> Dict[str, Any]:
        if not MANIFEST_PATH.is_file():
            self.errors.append(f"Checkpoint manifest missing at: {MANIFEST_PATH}")
            return {"status": "FAIL", "error": "MANIFEST_MISSING"}

        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        checkpoints = manifest_data.get("checkpoints", {})
        results = {}
        missing_core = []
        missing_optional = []

        for cid, info in checkpoints.items():
            expected_rel = info["expected_relative_path"]
            target_path = ROOT_DIR / expected_rel
            alt_paths = [ROOT_DIR / p for p in info.get("alternative_paths", [])]
            category = info.get("category", "unknown")
            expected_hash = info.get("sha256")

            # Check primary and alternative locations
            found_path = None
            if target_path.is_file():
                found_path = target_path
            else:
                for alt in alt_paths:
                    if alt.is_file():
                        found_path = alt
                        break

            if found_path is not None:
                actual_hash = compute_sha256(found_path)
                hash_match = (actual_hash == expected_hash)
                if not hash_match:
                    self.errors.append(
                        f"Checkpoint '{cid}' hash mismatch at {found_path}! Expected: {expected_hash[:16]}..., Found: {actual_hash[:16]}..."
                    )
                    status_str = "HASH_MISMATCH"
                else:
                    status_str = "VERIFIED_PRESENT"

                results[cid] = {
                    "model_name": info["model_name"],
                    "category": category,
                    "resolved_path": str(found_path.relative_to(ROOT_DIR)).replace("\\", "/"),
                    "sha256": actual_hash,
                    "hash_verified": hash_match,
                    "status": status_str,
                }
            else:
                results[cid] = {
                    "model_name": info["model_name"],
                    "category": category,
                    "resolved_path": None,
                    "status": "MISSING",
                    "obtaining_instructions": info.get("obtaining_instructions"),
                }
                if category == "core_production":
                    missing_core.append(info["model_name"])
                    self.warnings.append(
                        f"Core checkpoint missing: '{info['model_name']}' at '{expected_rel}'. Advice: {info.get('obtaining_instructions')}"
                    )
                else:
                    missing_optional.append(info["model_name"])

        overall_status = "PASS"
        if missing_core:
            overall_status = "WARNING"
            self.warnings.append(f"{len(missing_core)} core checkpoints are not present. See CHECKPOINTS.md for acquisition.")

        return {
            "total_checkpoints_defined": len(checkpoints),
            "missing_core_count": len(missing_core),
            "missing_optional_count": len(missing_optional),
            "checkpoints": results,
            "status": overall_status,
        }

    def audit_golden_regression(self) -> Dict[str, Any]:
        if not GOLDEN_EXPECTATIONS.is_file():
            self.warnings.append(f"Golden expectations file missing at: {GOLDEN_EXPECTATIONS}")
            return {"status": "MISSING", "error": "EXPECTATIONS_NOT_FOUND"}

        try:
            with open(GOLDEN_EXPECTATIONS, "r", encoding="utf-8") as f:
                expectations = json.load(f)

            if len(expectations) != 20:
                self.warnings.append(f"Expected 20 golden samples, found {len(expectations)}")

            missing_imgs = []
            for item in expectations:
                img_p = GOLDEN_IMAGES_DIR / item["filename"]
                if not img_p.is_file():
                    missing_imgs.append(item["filename"])

            if missing_imgs:
                self.warnings.append(f"Missing {len(missing_imgs)} golden images in tests/golden/images/")

            return {
                "expectations_count": len(expectations),
                "missing_images_count": len(missing_imgs),
                "status": "PASS" if not missing_imgs else "WARNING",
            }
        except Exception as e:
            self.errors.append(f"Failed to inspect golden dataset: {e}")
            return {"status": "FAIL", "error": str(e)}

    def run_full_audit(self) -> Dict[str, Any]:
        print("=" * 80)
        print("EGYPTIAN VEHICLE INTELLIGENCE: PORTABILITY & REPRODUCIBILITY AUDIT")
        print("=" * 80)

        py_res = self.audit_python_runtime()
        print(f"[*] Python Runtime:       [{py_res['status']}] {py_res['version']} ({py_res['architecture']}) on {py_res['platform']}")

        dep_res = self.audit_dependencies()
        print(f"[*] Dependencies:         [{dep_res['status']}] {dep_res['installed_count']} packages installed, {dep_res['missing_count']} missing")

        dev_res = self.audit_compute_device()
        print(f"[*] Hardware Compute:     [{dev_res['status']}] Mode: {dev_res['mode']} | Device: {dev_res['device_name']} | VRAM: {dev_res['vram_gb']} GB")

        io_res = self.audit_opencv_unicode_io()
        print(f"[*] OpenCV Unicode I/O:   [{io_res['status']}] Arabic path reading supported")

        cfg_res = self.audit_configuration()
        print(f"[*] System Configuration: [{cfg_res['status']}] Config valid | Gate footage: {cfg_res['gate_dir_status']}")

        ckpt_res = self.audit_checkpoints()
        print(f"[*] Checkpoints Manifest: [{ckpt_res['status']}] {ckpt_res['total_checkpoints_defined']} tracked (Missing core: {ckpt_res['missing_core_count']}, optional: {ckpt_res['missing_optional_count']})")

        gold_res = self.audit_golden_regression()
        print(f"[*] Golden Test Set:      [{gold_res['status']}] {gold_res.get('expectations_count', 0)} regression samples verified")

        print("-" * 80)
        if self.errors:
            print(f"[!] BLOCKING ISSUES ({len(self.errors)}):")
            for err in self.errors:
                print(f"    - {err}")
        else:
            print("[+] Zero blocking installation errors detected.")

        if self.warnings:
            print(f"[i] ADVISORY NOTICES ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"    - {w}")

        final_verdict = "PORTABLE" if not self.errors else "NON_PORTABLE"
        print("=" * 80)
        print(f"FINAL PORTABILITY VERDICT: {final_verdict}")
        print("=" * 80)

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "verdict": final_verdict,
            "python_runtime": py_res,
            "dependencies": dep_res,
            "compute_device": dev_res,
            "unicode_io": io_res,
            "configuration": cfg_res,
            "checkpoints": ckpt_res,
            "golden_regression": gold_res,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def main():
    parser = argparse.ArgumentParser(description="Audit Vehicle Intelligence system portability.")
    parser.add_argument("--json", type=Path, help="Optional output JSON path for audit report")
    parser.add_argument("--verbose", action="store_true", help="Print verbose package and checkpoint details")
    args = parser.parse_args()

    auditor = PortabilityAuditor(verbose=args.verbose)
    report = auditor.run_full_audit()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved portability audit report to: {args.json}")

    sys.exit(0 if not auditor.errors else 1)


if __name__ == "__main__":
    main()
