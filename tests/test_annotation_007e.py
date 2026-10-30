"""Tests for Mission 007E: Independent Gate Annotation and Ground Truth Finalization.

Covers:
- Strict validation rules enforcement (no_plate empty text, OOV character handling, invalid status)
- Distinction between wrong crops (crop_is_plate='no') and missed detections (crop_is_plate='no_crop')
- UTF-8 BOM round-trip and character encoding integrity
- Reconciled ground truth verification on completed_annotations_007e.csv
- Frozen artifact digest preservation (never modified)
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import pytest

from scripts.score_labeled_triage import (
    EALPR_LETTERS,
    EALPR_DIGITS,
    VALID_CHARACTERS,
    DOCUMENTED_OOV_LETTERS,
    score_labeled_csv,
    validate_row_schema,
    compute_file_sha256,
)

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_frozen_artifacts_integrity():
    """Verify that all frozen artifacts remain byte-exact and untouched."""
    expected_digests = {
        "artifacts/triage/labeling_interface.html": "a36c50dcf2ea2ec14f00f46c0760c085d089bae8bbb1bab155829dc2acf779f2",
        "artifacts/triage/label_sheet.csv": "97242c3475d89ed10e4f7220f758ac48f7c4a73600f04b1275e3d0f29013908e",
        "artifacts/triage/triage.csv": "4f941fc182972f518139e64984e7b8b0bb2b71e3a4f6196e88968555e36ff5cb",
        "artifacts/triage/triage_summary.json": "82d3fdcffba8aa86e1c0eed82c4bd0aa53a0a8f7dd0ea1b25647122b9914f889",
        "artifacts/triage/completed_annotations.csv": "12a09854d61699f0e98ac02661b47d3bc0f389794e40bf534bfb0d0d689e9d59",
    }
    for rel_path, expected_hash in expected_digests.items():
        p = ROOT_DIR / rel_path
        assert p.exists(), f"Frozen artifact missing: {p}"
        actual_hash = compute_file_sha256(p)
        assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}"


def test_completed_annotations_007e_passes_strict_validation():
    """Verify that completed_annotations_007e.csv passes strict scoring with zero errors."""
    csv_path = ROOT_DIR / "artifacts" / "triage" / "completed_annotations_007e.csv"
    assert csv_path.exists(), f"File missing: {csv_path}"

    report = score_labeled_csv(csv_path, strict=True, bootstrap_draws=100)
    assert report["sample_summary"]["total_frames"] == 100
    assert report["sample_summary"]["in_vocab_readable_crops"] == 77
    assert report["sample_summary"]["partially_readable"] == 5
    assert report["sample_summary"]["no_plate_frames"] == 1
    assert report["sample_summary"]["unreadable_plates"] == 0

    # Readable crops accuracy: 68 / 77 = 88.31%
    assert report["step_3a_exact_match_in_vocab"]["numerator"] == 68
    assert report["step_3a_exact_match_in_vocab"]["denominator"] == 77
    assert report["step_3a_exact_match_in_vocab"]["accuracy_pct"] == 88.31

    # All-in accuracy
    assert report["step_3c_all_in_accuracy"]["exact_matches"] == 68
    assert report["step_3c_all_in_accuracy"]["all_frames_denominator"] == 100
    assert report["step_3c_all_in_accuracy"]["all_in_accuracy_all_frames_pct"] == 68.0
    assert report["step_3c_all_in_accuracy"]["visible_plates_denominator"] == 99
    assert report["step_3c_all_in_accuracy"]["all_in_accuracy_visible_pct"] == 68.69

    # Model and legacy agreement against human label should now be 0
    assert report["step_3i_disagreements_model_legacy_agree_against_human"]["count"] == 0


def test_no_plate_empty_text_rule():
    """Verify that no_plate rows must not have invented plate text."""
    # Invalid: status=no_plate with text
    invalid_row = {
        "file": "test.jpg",
        "status": "no_plate",
        "true_letters": "???",
        "true_digits": "???",
        "crop_is_plate": "no_crop",
        "unsupported_chars": "no",
    }
    ok, errs = validate_row_schema(1, invalid_row, strict=True)
    assert not ok
    assert any("contains '?' but status is 'no_plate'" in e for e in errs)

    # Valid: status=no_plate with empty text
    valid_row = {
        "file": "test.jpg",
        "status": "no_plate",
        "true_letters": "",
        "true_digits": "",
        "crop_is_plate": "no_crop",
        "unsupported_chars": "no",
    }
    ok, errs = validate_row_schema(1, valid_row, strict=True)
    assert ok
    assert len(errs) == 0


def test_oov_character_requires_unsupported_flag():
    """Verify that OOV characters (e.g. 'ك') fail without unsupported_chars=yes."""
    row_no_flag = {
        "file": "test.jpg",
        "status": "readable",
        "true_letters": "كدم",
        "true_digits": "٨٥٧٣",
        "crop_is_plate": "yes",
        "unsupported_chars": "no",
    }
    ok, errs = validate_row_schema(1, row_no_flag, strict=True)
    assert not ok
    assert any("character 'ك' (U+0643) in 'true_letters' is outside" in e for e in errs)

    # When flagged as unsupported_chars=yes, it is accepted
    row_with_flag = dict(row_no_flag)
    row_with_flag["unsupported_chars"] = "yes"
    ok, errs = validate_row_schema(1, row_with_flag, strict=True)
    assert ok
    assert len(errs) == 0


def test_audit_revisions_reconciled():
    """Verify the 11 audited and revised rows in completed_annotations_007e.csv."""
    csv_path = ROOT_DIR / "artifacts" / "triage" / "completed_annotations_007e.csv"
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # Row 1: Seen, Qaf, Jeem
    assert rows[0]["true_letters"] == "سقج"
    assert rows[0]["true_digits"] == "٨٥٦٨"

    # Row 14: Motorcycle no plate
    assert rows[13]["status"] == "no_plate"
    assert rows[13]["true_letters"] == ""
    assert rows[13]["true_digits"] == ""
    assert rows[13]["crop_is_plate"] == "no_crop"
    assert rows[13]["plate_visible_to_eye"] == "no"

    # Row 18: Tricycle OOV ك
    assert rows[17]["true_letters"] == "كمد"
    assert rows[17]["unsupported_chars"] == "yes"

    # Row 23: Waw, Ain, Raa
    assert rows[22]["true_letters"] == "وعر"
    assert rows[22]["true_digits"] == "٥١٤٦"

    # Row 30: Tricycle OOV ك
    assert rows[29]["true_letters"] == "كمد"
    assert rows[29]["unsupported_chars"] == "yes"

    # Row 32: Tricycle OOV ك
    assert rows[31]["true_letters"] == "كمد"
    assert rows[31]["unsupported_chars"] == "yes"

    # Row 37: Seen, Qaf, Jeem
    assert rows[36]["true_letters"] == "سقج"
    assert rows[36]["true_digits"] == "٨٥٦٨"

    # Row 38: Tricycle OOV ك, status readable
    assert rows[37]["status"] == "readable"
    assert rows[37]["true_letters"] == "كمد"
    assert rows[37]["unsupported_chars"] == "yes"

    # Rows 55, 90, 93: OOV ك
    assert rows[54]["true_letters"] == "كدم"
    assert rows[54]["unsupported_chars"] == "yes"
    assert rows[89]["true_letters"] == "كدو"
    assert rows[89]["unsupported_chars"] == "yes"
    assert rows[92]["true_letters"] == "كطو"
    assert rows[92]["unsupported_chars"] == "yes"
