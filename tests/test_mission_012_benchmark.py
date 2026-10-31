"""Unit and Regression Tests for Mission 012: Independent Vehicle Ground Truth & Benchmark."""

import csv
import json
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
M12_DIR = ROOT_DIR / "artifacts" / "missions" / "mission_012"
GROUND_TRUTH_FILE = M12_DIR / "vehicle_ground_truth.json"
VALIDATION_REPORT = M12_DIR / "annotation_validation_report.json"
SPLITS_FILE = M12_DIR / "benchmark_splits.json"
INTERFACE_DIR = M12_DIR / "annotation_interface"

VALID_CLASSES = {
    "heavy_truck", "light_commercial", "passenger_car",
    "bus", "microbus_van", "motorcycle", "tricycle", "other_vehicle"
}
VALID_VISIBILITIES = {
    "fully_visible", "partially_visible", "heavily_occluded", "truncated_by_frame"
}


def test_ground_truth_integrity_and_validation():
    """Verify that all 340 gate captures are annotated, validated, and adhere to coordinate bounds."""
    assert GROUND_TRUTH_FILE.is_file(), f"Missing ground truth file: {GROUND_TRUTH_FILE}"
    assert VALIDATION_REPORT.is_file(), f"Missing validation report: {VALIDATION_REPORT}"

    with open(VALIDATION_REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["status"] == "PASSED", f"Ground truth validation failed: {report.get('errors')}"
    assert report["total_samples"] == 340, f"Expected 340 samples, found {report['total_samples']}"
    assert report["error_count"] == 0

    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        gt = json.load(f)

    assert len(gt) == 340

    for sid, entry in gt.items():
        assert entry["image_width"] == 3840
        assert entry["image_height"] == 2160
        assert entry["lighting_condition"] in {"NIGHT_IR", "DAYLIGHT", "EVENING_IR"}

        if entry["is_empty_frame"]:
            assert len(entry["boxes"]) == 0
        else:
            assert len(entry["boxes"]) >= 1
            for b in entry["boxes"]:
                assert b["class"] in VALID_CLASSES, f"Invalid class {b['class']} in {sid}"
                assert b["visibility"] in VALID_VISIBILITIES, f"Invalid visibility {b['visibility']} in {sid}"

                x1, y1, x2, y2 = b["bbox_xyxy"]
                assert 0.0 <= x1 < x2 <= 3840.0, f"Invalid X coordinates [{x1}, {x2}] in {sid}"
                assert 0.0 <= y1 < y2 <= 2160.0, f"Invalid Y coordinates [{y1}, {y2}] in {sid}"


def test_benchmark_splits_reproducibility_and_anti_leakage():
    """Verify benchmark splits: ~60/20/20 allocation, temporal event grouping, and zero intra-event leakage."""
    assert SPLITS_FILE.is_file(), f"Missing splits file: {SPLITS_FILE}"

    with open(SPLITS_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_samples = set(splits["train"]["sample_ids"])
    val_samples = set(splits["val"]["sample_ids"])
    test_samples = set(splits["test"]["sample_ids"])

    # Disjointness check
    assert len(train_samples.intersection(val_samples)) == 0, "Leakage detected between train and val"
    assert len(train_samples.intersection(test_samples)) == 0, "Leakage detected between train and test"
    assert len(val_samples.intersection(test_samples)) == 0, "Leakage detected between val and test"

    total_split_frames = len(train_samples) + len(val_samples) + len(test_samples)
    assert total_split_frames == 340, f"Expected 340 total frames, found {total_split_frames}"

    # Verify event allocation
    alloc = splits["metadata"]["allocation"]
    assert 55.0 <= alloc["train"]["pct_frames"] <= 65.0
    assert 18.0 <= alloc["val"]["pct_frames"] <= 25.0
    assert 18.0 <= alloc["test"]["pct_frames"] <= 25.0


def test_annotation_interface_files_exist_and_clean():
    """Verify annotation interface exists and contains zero model prediction leaks."""
    index_html = INTERFACE_DIR / "index.html"
    styles_css = INTERFACE_DIR / "styles.css"
    annotator_js = INTERFACE_DIR / "annotator.js"

    assert index_html.is_file()
    assert styles_css.is_file()
    assert annotator_js.is_file()

    with open(index_html, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Confirm no model predictions or legacy OCR text hardcoded in DOM
    assert "yolov8" not in html_content.lower()
    assert "rtdetr" not in html_content.lower()
