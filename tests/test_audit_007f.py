"""Test suite for Mission 007F: Ground Truth Independence Audit & Metric Recomputation."""

import csv
import json
import hashlib
from pathlib import Path
import pytest


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def mission_007f_dir(repo_root):
    return repo_root / "artifacts" / "triage" / "mission_007f"


def test_frozen_artifacts_integrity(repo_root):
    """Verify that all 4 frozen triage artifacts and samples digest match their frozen values."""
    expected = {
        "artifacts/triage/labeling_interface.html": "a36c50dcf2ea2ec14f00f46c0760c085d089bae8bbb1bab155829dc2acf779f2",
        "artifacts/triage/label_sheet.csv": "97242c3475d89ed10e4f7220f758ac48f7c4a73600f04b1275e3d0f29013908e",
        "artifacts/triage/triage.csv": "4f941fc182972f518139e64984e7b8b0bb2b71e3a4f6196e88968555e36ff5cb",
        "artifacts/triage/triage_summary.json": "82d3fdcffba8aa86e1c0eed82c4bd0aa53a0a8f7dd0ea1b25647122b9914f889",
    }
    for rel_p, exp_hash in expected.items():
        p = repo_root / rel_p
        assert p.exists(), f"Missing frozen file {p}"
        actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual_hash == exp_hash, f"Hash mismatch for {rel_p}: expected {exp_hash}, got {actual_hash}"


def test_annotation_csv_hashes(repo_root):
    """Verify that original completed_annotations.csv and completed_annotations_007e.csv match their digests."""
    orig_p = repo_root / "artifacts" / "triage" / "completed_annotations.csv"
    final_p = repo_root / "artifacts" / "triage" / "completed_annotations_007e.csv"

    assert orig_p.exists()
    assert final_p.exists()

    orig_hash = hashlib.sha256(orig_p.read_bytes()).hexdigest()
    final_hash = hashlib.sha256(final_p.read_bytes()).hexdigest()

    assert orig_hash == "12a09854d61699f0e98ac02661b47d3bc0f389794e40bf534bfb0d0d689e9d59"
    assert final_hash == "2956e3880cbd068f4d2e63f1c5817a4ea3980167c8c28cbd8651b9294aa0a718"


def test_row_level_provenance_csv(mission_007f_dir):
    """Verify row_level_provenance.csv structure, columns, row count, and provenance categories."""
    prov_csv = mission_007f_dir / "row_level_provenance.csv"
    assert prov_csv.exists(), "row_level_provenance.csv does not exist"

    with open(prov_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    expected_fields = [
        "sample_id",
        "original_annotation",
        "final_annotation",
        "changed",
        "provenance_category",
        "evidence_available",
        "independence_status",
        "explanation",
    ]
    for field in expected_fields:
        assert field in fieldnames, f"Missing field {field} in row_level_provenance.csv"

    assert len(rows) == 100, f"Expected 100 rows, got {len(rows)}"

    cats = [r["provenance_category"] for r in rows]
    assert cats.count("copied_from_prior_annotation") == 89
    assert cats.count("visually_rechecked") == 11
    assert cats.count("new_blind_transcription") == 0

    # Verify the 11 modified rows specifically
    changed_rows = [int(r["sample_id"].replace("sample_", "")) for r in rows if r["changed"] == "true"]
    assert sorted(changed_rows) == [1, 14, 18, 23, 30, 32, 37, 38, 55, 90, 93]


def test_provenance_audit_json(mission_007f_dir):
    """Verify provenance_audit.json content, verdicts, and counts."""
    audit_json = mission_007f_dir / "provenance_audit.json"
    assert audit_json.exists(), "provenance_audit.json does not exist"

    data = json.loads(audit_json.read_text(encoding="utf-8"))
    assert data["provenance_breakdown"]["copied_from_prior_annotation"] == 89
    assert data["provenance_breakdown"]["visually_rechecked"] == 11
    assert data["provenance_breakdown"]["new_blind_transcription"] == 0
    assert data["independence_criteria_status"] == "BLOCKED"
    assert data["independence_verdict"]["claim_independent_ground_truth"] is False


def test_evaluation_configuration_audit_json(mission_007f_dir):
    """Verify evaluation_configuration_audit.json content, NMS resolution, and model hashes."""
    cfg_json = mission_007f_dir / "evaluation_configuration_audit.json"
    assert cfg_json.exists(), "evaluation_configuration_audit.json does not exist"

    data = json.loads(cfg_json.read_text(encoding="utf-8"))
    models = data["model_checkpoints"]
    assert models["detector"]["sha256"] == "1d9ab549c626ee1b708b3439ad63320515c6de105283a5c980b3c5952337b279"
    assert models["character_recognizer"]["sha256"] == "875b76e18b1df480829f84809e1f26d2c38d2c92c7f66d3bd85492db14488319"

    cfg_matrix = data["configuration_matrix"]
    assert cfg_matrix["val_tuning_selected_config"]["agnostic_nms"] is True
    assert cfg_matrix["production_pipeline_default_config"]["agnostic_nms"] is False
    assert cfg_matrix["mission_007e_scoring_config"]["agnostic_nms"] is False

    assert "crops_with_differing_predictions" in data["nms_discrepancy_resolution"]["empirical_gate_crop_impact"]
    assert data["nms_discrepancy_resolution"]["empirical_gate_crop_impact"]["crops_with_differing_predictions"] == 6


def test_metric_recalculation_json(mission_007f_dir):
    """Verify metric_recalculation.json metrics and checklist assertions."""
    metric_json = mission_007f_dir / "metric_recalculation.json"
    assert metric_json.exists(), "metric_recalculation.json does not exist"

    data = json.loads(metric_json.read_text(encoding="utf-8"))
    checklist = data["denominator_verification_checklist"]
    assert checklist["68_over_77_is_88_31_pct"]["verified"] is True
    assert checklist["detection_recall_uses_visible_plates_denominator"]["verified"] is True
    assert checklist["oov_plates_not_silently_counted_as_ordinary_ocr_failures_or_successes"]["verified"] is True
    assert checklist["no_crop_cases_included_in_end_to_end_but_excluded_from_conditional_ocr"]["verified"] is True
    assert checklist["all_frame_success_uses_all_100_frames"]["verified"] is True

    # Check specific metric items
    metrics_by_name = {m["metric_name"]: m for m in data["metrics_audit"]}
    assert metrics_by_name["Visible Plates"]["numerator"] == 99
    assert metrics_by_name["Readable Plates"]["numerator"] == 94
    assert metrics_by_name["Partially Readable Plates"]["numerator"] == 5
    assert metrics_by_name["No-Plate Frames"]["numerator"] == 1
    assert metrics_by_name["Detector Successes"]["numerator"] == 81
    assert metrics_by_name["Wrong Crops"]["numerator"] == 0
    assert metrics_by_name["Conditional OCR Exact Match"]["numerator"] == 68
    assert metrics_by_name["Conditional OCR Exact Match"]["denominator"] == 77
    assert metrics_by_name["Conditional OCR Exact Match"]["value_pct"] == 88.31
    assert metrics_by_name["All-Frame Exact Success"]["all_100_pct"] == 68.00


def test_blank_independent_interface_has_zero_prefilled_text(mission_007f_dir):
    """Verify blank_independent_interface.html has no pre-filled text or model predictions."""
    html_p = mission_007f_dir / "blank_independent_interface.html"
    assert html_p.exists()
    content = html_p.read_text(encoding="utf-8")

    # Assert no pre-filled Arabic strings
    assert 'value="سقج"' not in content
    assert 'value="سجق"' not in content
    assert 'value="٨٥٦٨"' not in content
    # Assert anonymous IDs are present
    assert "Sample #001" in content
    assert "Sample #100" in content
    # Assert export targets independent_annotations.csv
    assert "independent_annotations.csv" in content
