"""Tests for dataset preparation and YOLO export integrity."""

import json
from pathlib import Path
import pytest
import yaml

from vehicle_intelligence.evaluation.detection_eval import GroundTruthPlate


def test_dataset_yaml_structure_and_paths():
    yaml_path = Path("artifacts/ealpr_yolo_dataset/dataset.yaml")
    assert yaml_path.is_file(), "dataset.yaml was not generated"

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "train" in data
    assert "val" in data
    assert "test" in data
    assert "names" in data
    assert data["names"][0] == "license_plate"

    dataset_root = Path(data["path"])
    assert (dataset_root / data["train"]).is_dir()
    assert (dataset_root / data["val"]).is_dir()
    assert (dataset_root / data["test"]).is_dir()


def test_exported_ground_truth_jsons():
    splits_dir = Path("artifacts/splits")

    for split in ["train", "val", "test"]:
        gt_file = splits_dir / f"{split}_ground_truth_plates.json"
        assert gt_file.is_file(), f"Missing ground truth file: {gt_file}"

        with open(gt_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        assert len(records) > 0
        # Validate that each entry parses cleanly as a GroundTruthPlate
        for item in records[:10]:
            gt = GroundTruthPlate(**item)
            assert gt.class_name == "license_plate"
            assert 0.0 <= gt.box.x1 <= 1.0
            assert 0.0 <= gt.box.y1 <= 1.0
            assert gt.box.x2 >= gt.box.x1
            assert gt.box.y2 >= gt.box.y1


def test_split_counts_match_manifest():
    manifest_file = Path("artifacts/splits/split_manifest.json")
    assert manifest_file.is_file()

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for split in ["train", "val", "test"]:
        stems = manifest[f"{split}_vehicles"]
        img_dir = Path("artifacts/ealpr_yolo_dataset/images") / split
        lbl_dir = Path("artifacts/ealpr_yolo_dataset/labels") / split

        imgs = list(img_dir.glob("*.*"))
        lbls = list(lbl_dir.glob("*.txt"))

        assert len(imgs) == len(stems), f"{split} images count mismatch"
        assert len(lbls) == len(stems), f"{split} labels count mismatch"
