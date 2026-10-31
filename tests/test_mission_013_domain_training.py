"""Mission 013: Automated Regression Tests.

Validates:
1. Training configurations (YOLOv8m and RT-DETR-L).
2. Original pretrained checkpoint preservation (SHA-256 immutability).
3. Dataset split isolation and zero-leakage enforcement.
4. Evaluation accounting closure (precision, recall, F1, frame counts, breakdowns).
5. Checkpoint manifest integrity and licensing metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
M013_DIR = ROOT_DIR / "artifacts" / "missions" / "mission_013"
CONFIG_DIR = M013_DIR / "training_configurations"
DATASET_DIR = M013_DIR / "dataset"
CKPT_DIR = M013_DIR / "checkpoints"
EVAL_DIR = M013_DIR / "evaluation_results"

FROZEN_YOLO_SHA = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
FROZEN_RTDETR_SHA = "6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class TestMission013Configurations:
    """Verifies that hyperparameter configurations are reproducible and compliant."""

    def test_yolov8m_configuration(self):
        cfg_path = CONFIG_DIR / "yolov8m_train_config.yaml"
        assert cfg_path.is_file(), f"Missing configuration: {cfg_path}"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert cfg["model_id"] == "yolov8m_finetuned"
        assert cfg["epochs"] == 25
        assert cfg["batch"] == 8
        assert cfg["imgsz"] == 640
        assert cfg["optimizer"] == "AdamW"
        assert cfg["lr0"] == 0.001
        assert cfg["seed"] == 42
        assert cfg["deterministic"] is True
        assert cfg["workers"] == 0

    def test_rtdetr_l_configuration(self):
        cfg_path = CONFIG_DIR / "rtdetr_l_train_config.yaml"
        assert cfg_path.is_file(), f"Missing configuration: {cfg_path}"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert cfg["model_id"] == "rtdetr_l_finetuned"
        assert cfg["epochs"] == 25
        assert cfg["batch"] == 4  # conservative for transformer memory stability
        assert cfg["imgsz"] == 640
        assert cfg["optimizer"] == "AdamW"
        assert cfg["lr0"] == 0.0001
        assert cfg["seed"] == 42
        assert cfg["deterministic"] is True
        assert cfg["workers"] == 0


class TestMission013CheckpointPreservation:
    """Verifies that original pretrained weights were preserved without modification."""

    def test_original_yolov8m_hash_preserved(self):
        yolo_path = ROOT_DIR / "yolov8m.pt"
        assert yolo_path.is_file()
        current_sha = sha256_file(yolo_path)
        assert current_sha == FROZEN_YOLO_SHA, (
            f"Original yolov8m.pt was corrupted or modified! Expected {FROZEN_YOLO_SHA}, got {current_sha}"
        )

    def test_original_rtdetr_l_hash_preserved(self):
        rtdetr_path = ROOT_DIR / "rtdetr-l.pt"
        assert rtdetr_path.is_file()
        current_sha = sha256_file(rtdetr_path)
        assert current_sha == FROZEN_RTDETR_SHA, (
            f"Original rtdetr-l.pt was corrupted or modified! Expected {FROZEN_RTDETR_SHA}, got {current_sha}"
        )


class TestMission013DataSplitIsolation:
    """Verifies strict split containment and that training cannot touch dev-test."""

    def test_dataset_split_boundaries(self):
        splits_path = ROOT_DIR / "artifacts" / "missions" / "mission_012" / "benchmark_splits.json"
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

        train_set = set(splits["train"]["filenames"])
        val_set = set(splits["val"]["filenames"])
        test_set = set(splits["test"]["filenames"])

        assert len(train_set) == 196
        assert len(val_set) == 72
        assert len(test_set) == 72
        assert len(train_set | val_set | test_set) == 340

        # Strict zero-leakage assertions
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_isolated_yolo_dataset_structure(self):
        dataset_yaml = DATASET_DIR / "dataset.yaml"
        assert dataset_yaml.is_file()

        with open(dataset_yaml, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)

        assert data_cfg["names"] == {0: "vehicle"}

        train_imgs = list((DATASET_DIR / "images" / "train").glob("*.jpg"))
        val_imgs = list((DATASET_DIR / "images" / "val").glob("*.jpg"))
        test_imgs = list((DATASET_DIR / "images" / "test").glob("*.jpg"))

        assert len(train_imgs) == 196
        assert len(val_imgs) == 72
        assert len(test_imgs) == 72

        # Check labels count matches
        train_labels = list((DATASET_DIR / "labels" / "train").glob("*.txt"))
        val_labels = list((DATASET_DIR / "labels" / "val").glob("*.txt"))
        test_labels = list((DATASET_DIR / "labels" / "test").glob("*.txt"))

        assert len(train_labels) == 196
        assert len(val_labels) == 72
        assert len(test_labels) == 72


class TestMission013EvaluationAccounting:
    """Verifies mathematical closure and consistency of evaluation metrics."""

    def test_evaluation_accounting_closure(self):
        full_eval_path = EVAL_DIR / "full_evaluation_results.json"
        if not full_eval_path.is_file():
            pytest.skip("Full evaluation results not yet generated.")

        with open(full_eval_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        for model_id, m_data in results.items():
            test_ev = m_data["development_test_v1"]

            tp = test_ev["true_positives_50"]
            fp = test_ev["false_positives_50"]
            fn = test_ev["false_negatives_50"]
            gt = test_ev["total_ground_truth_vehicles"]
            frames = test_ev["total_frames"]

            assert frames == 72
            assert gt == 72
            assert tp + fn == gt, f"{model_id}: TP + FN must equal ground truth"

            expected_p = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
            expected_r = round(tp / gt, 4) if gt > 0 else 0.0
            expected_f1 = round(2 * expected_p * expected_r / (expected_p + expected_r), 4) if (expected_p + expected_r) > 0 else 0.0

            assert abs(test_ev["precision_50"] - expected_p) <= 1e-3
            assert abs(test_ev["recall_50"] - expected_r) <= 1e-3
            assert abs(test_ev["f1_score_50"] - expected_f1) <= 1e-3

            # Lighting closure
            lb = test_ev["lighting_breakdown"]
            assert lb["DAYLIGHT"]["frames"] + lb["EVENING_IR"]["frames"] + lb["NIGHT_IR"]["frames"] == 72
            assert lb["DAYLIGHT"]["frames"] == 44
            assert lb["EVENING_IR"]["frames"] == 26
            assert lb["NIGHT_IR"]["frames"] == 2

            # Class closure
            cb = test_ev["class_breakdown"]
            assert cb["heavy_truck"]["ground_truth_instances"] == 71
            assert cb["passenger_car"]["ground_truth_instances"] == 1


class TestMission013ManifestAndLicensing:
    """Verifies license metadata and cryptographic traceability."""

    def test_checkpoint_manifest_traceability(self):
        manifest_path = M013_DIR / "checkpoint_manifest.json"
        if not manifest_path.is_file():
            pytest.skip("Checkpoint manifest not yet generated.")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        expected_models = [
            "rtdetr_l_pretrained",
            "rtdetr_l_finetuned",
            "yolov8m_pretrained",
            "yolov8m_finetuned",
        ]
        for mid in expected_models:
            assert mid in manifest, f"Missing model in manifest: {mid}"
            entry = manifest[mid]
            assert "sha256" in entry and len(entry["sha256"]) == 64
            assert entry["parameters_count"] > 20_000_000
            assert "license" in entry and len(entry["license"]) > 0
