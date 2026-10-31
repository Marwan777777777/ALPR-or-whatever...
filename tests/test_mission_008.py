"""Test suite for Mission 008: Controlled NMS Correction and Improvement Diagnostics."""

import csv
import json
import hashlib
from pathlib import Path
import pytest

from vehicle_intelligence.core.config import DetectorConfig, OCRConfig, SystemConfig
from vehicle_intelligence.pipeline.alpr_pipeline import EndToEndALPRPipeline


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def mission_008_dir(repo_root):
    return repo_root / "artifacts" / "missions" / "mission_008"


def test_pipeline_default_agnostic_nms_is_true(repo_root):
    """Verify EndToEndALPRPipeline uses agnostic_nms=True by default and supports False explicitly."""
    p_default = EndToEndALPRPipeline(device="cpu")
    assert p_default.agnostic_nms is True, "Pipeline default agnostic_nms must be True (Mission 008 standard)"

    p_false = EndToEndALPRPipeline(device="cpu", agnostic_nms=False)
    assert p_false.agnostic_nms is False, "Pipeline must allow explicit agnostic_nms=False to reproduce historical baseline"


def test_config_ocr_agnostic_nms_is_true():
    """Verify OCRConfig defaults to agnostic_nms=True and DetectorConfig defaults to False."""
    ocr_cfg = OCRConfig()
    assert ocr_cfg.agnostic_nms is True

    det_cfg = DetectorConfig()
    assert det_cfg.agnostic_nms is False


def test_nms_comparison_target_5_crops_corrected(mission_008_dir):
    """Verify nms_comparison.json verifies all 5 corrected crops individually."""
    json_p = mission_008_dir / "nms_comparison.json"
    assert json_p.exists(), f"Missing {json_p}"

    data = json.loads(json_p.read_text(encoding="utf-8"))
    target_5 = data["target_5_corrected_crops_verification"]
    assert len(target_5) == 5

    expected_rows = [25, 45, 47, 94, 96]
    for item in target_5:
        assert item["row"] in expected_rows
        assert item["exact_agnostic_false"] is False, f"Row {item['row']} was supposed to fail under False NMS"
        assert item["exact_agnostic_true"] is True, f"Row {item['row']} was not corrected under True NMS"

    # Verify overall improvement
    summary_true = data["mission_008_intended_agnostic_true"]
    assert summary_true["exact_matches_numerator"] == 73
    assert summary_true["in_vocab_readable_denominator"] == 77
    assert summary_true["exact_match_accuracy_pct"] == 94.81


def test_detection_miss_analysis_csv_exists_and_valid(mission_008_dir):
    """Verify detection_miss_analysis.csv exists, has 18 rows and valid categories."""
    csv_p = mission_008_dir / "detection_miss_analysis.csv"
    assert csv_p.exists(), f"Missing {csv_p}"

    with open(csv_p, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 18, f"Expected 18 visible-plate misses, got {len(rows)}"

    valid_categories = {
        "Motion blur",
        "Headlight glare / blooming",
        "Plate occlusion",
        "Oblique angle",
        "Plate outside frame",
        "Small or unusual plate",
        "Low contrast / IR reflection",
        "Other",
    }

    for r in rows:
        assert r["likely_failure_category"] in valid_categories, f"Invalid category: {r['likely_failure_category']}"
        assert r["human_locatable"] in ("yes", "partial", "no")
        assert r["plausibly_fixable_by_detector_improvement"] in ("yes", "no")


def test_ocr_error_analysis_csv_exists_and_valid(mission_008_dir):
    """Verify ocr_error_analysis.csv exists and documents all remaining OCR failure modes."""
    csv_p = mission_008_dir / "ocr_error_analysis.csv"
    assert csv_p.exists(), f"Missing {csv_p}"

    with open(csv_p, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 8, f"Expected 8 documented OCR failure rows, got {len(rows)}"

    sample_ids = [r["sample_id"] for r in rows]
    # In-vocab remaining errors (4)
    assert "sample_041" in sample_ids
    assert "sample_079" in sample_ids
    assert "sample_083" in sample_ids
    assert "sample_084" in sample_ids
    # OOV errors (3)
    assert "sample_055" in sample_ids
    assert "sample_090" in sample_ids
    assert "sample_093" in sample_ids
    # Underexposed crop (1)
    assert "sample_015" in sample_ids


def test_experiment_results_json_structure(mission_008_dir):
    """Verify experiment_results.json contains all 3 experiments with valid metrics."""
    json_p = mission_008_dir / "experiment_results.json"
    assert json_p.exists(), f"Missing {json_p}"

    data = json.loads(json_p.read_text(encoding="utf-8"))
    assert "experiment_a_char_nms_iou" in data
    assert "experiment_b_crop_padding" in data
    assert "experiment_c_detector_resolution" in data

    exp_a = data["experiment_a_char_nms_iou"]
    assert exp_a["selected_best_iou_on_val"] == 0.45
    assert exp_a["gate_triage_comparison"]["iou_45"]["exact_matches"] == 75

    exp_c = data["experiment_c_detector_resolution"]
    assert len(exp_c["results"]) == 3
    # 1280 resolution recovers 6 missed plates
    assert exp_c["results"][2]["recovered_missed_plates_count"] == 6


def test_frozen_artifacts_integrity_preserved(repo_root):
    """Verify that all frozen triage files and completed annotations remain unchanged."""
    expected = {
        "artifacts/triage/labeling_interface.html": "a36c50dcf2ea2ec14f00f46c0760c085d089bae8bbb1bab155829dc2acf779f2",
        "artifacts/triage/label_sheet.csv": "97242c3475d89ed10e4f7220f758ac48f7c4a73600f04b1275e3d0f29013908e",
        "artifacts/triage/triage.csv": "4f941fc182972f518139e64984e7b8b0bb2b71e3a4f6196e88968555e36ff5cb",
        "artifacts/triage/triage_summary.json": "82d3fdcffba8aa86e1c0eed82c4bd0aa53a0a8f7dd0ea1b25647122b9914f889",
        "artifacts/triage/completed_annotations.csv": "12a09854d61699f0e98ac02661b47d3bc0f389794e40bf534bfb0d0d689e9d59",
        "artifacts/triage/completed_annotations_007e.csv": "2956e3880cbd068f4d2e63f1c5817a4ea3980167c8c28cbd8651b9294aa0a718",
    }
    for rel_p, exp_hash in expected.items():
        p = repo_root / rel_p
        assert p.exists()
        actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual_hash == exp_hash, f"Frozen file {rel_p} was altered!"
