"""Unit and Regression Tests for Mission 012C: Final Split Reconciliation & Mission 013 Authorization."""

import csv
import hashlib
import json
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
M12_DIR = ROOT_DIR / "artifacts" / "missions" / "mission_012"
M12C_DIR = ROOT_DIR / "artifacts" / "missions" / "mission_012c"

CANONICAL_SPLIT_JSON = M12C_DIR / "canonical_event_split_report.json"
ASSOC_TEST_RECONCILIATION_CSV = M12C_DIR / "association_test_reconciliation.csv"
EXTERNAL_PROTOCOL_MD = M12C_DIR / "external_test_set_protocol.md"
MISSION_013_AUTH_MD = M12C_DIR / "mission_013_authorization.md"
MISSION_012C_REPORT_MD = M12C_DIR / "mission_012c_report.md"

ORIGINAL_SPLITS_FILE = M12_DIR / "benchmark_splits.json"
EXPECTED_SPLITS_SHA256 = "3138160cff5f6368d1d848cce48ede77b6ea2acbfc98c2dd0f566a5f24ada39e"


def test_canonical_event_split_integrity():
    """Verify canonical event split report: 175/58/58 events, 196/72/72 frames, and 0 leakage."""
    assert CANONICAL_SPLIT_JSON.is_file(), f"Missing {CANONICAL_SPLIT_JSON}"

    with open(CANONICAL_SPLIT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["milestone"] == "MISSION_012C"
    assert data["original_split_preserved"] is True
    assert data["splits_source_sha256"] == EXPECTED_SPLITS_SHA256

    summary = data["canonical_partition_summary"]
    assert summary["train"]["event_count"] == 175
    assert summary["train"]["frame_count"] == 196

    assert summary["validation"]["event_count"] == 58
    assert summary["validation"]["frame_count"] == 72

    assert summary["development_test_v1"]["event_count"] == 58
    assert summary["development_test_v1"]["frame_count"] == 72

    assert summary["dataset_total"]["total_events"] == 291
    assert summary["dataset_total"]["total_frames"] == 340

    leakage = data["leakage_and_overlap_audit"]
    assert leakage["zero_leakage_verified"] is True
    assert leakage["train_val_event_overlap"] == 0
    assert leakage["train_test_event_overlap"] == 0
    assert leakage["val_test_event_overlap"] == 0
    assert leakage["train_val_frame_overlap"] == 0
    assert leakage["train_test_frame_overlap"] == 0
    assert leakage["val_test_frame_overlap"] == 0
    assert leakage["unassigned_frames"] == 0
    assert leakage["duplicate_frame_assignments"] == 0


def test_association_test_accounting_completeness_69():
    """Verify test association accounting closure: exactly 69 observations, summing to 100% for all models."""
    assert ASSOC_TEST_RECONCILIATION_CSV.is_file(), f"Missing {ASSOC_TEST_RECONCILIATION_CSV}"

    with open(ASSOC_TEST_RECONCILIATION_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 69, f"Expected 69 test observations, found {len(rows)}"

    valid_outcomes = {
        "correctly_associated",
        "incorrectly_associated",
        "unassigned_no_vehicle",
        "unassigned_no_candidate",
        "ambiguous",
        "partially_contained_clipped",
    }

    # Verify YOLOv8n
    v8n_counts = {}
    v8m_counts = {}
    rt_counts = {}

    for r in rows:
        assert r["yolov8n_outcome"] in valid_outcomes
        assert r["yolov8m_outcome"] in valid_outcomes
        assert r["rtdetr_l_outcome"] in valid_outcomes

        v8n_counts[r["yolov8n_outcome"]] = v8n_counts.get(r["yolov8n_outcome"], 0) + 1
        v8m_counts[r["yolov8m_outcome"]] = v8m_counts.get(r["yolov8m_outcome"], 0) + 1
        rt_counts[r["rtdetr_l_outcome"]] = rt_counts.get(r["rtdetr_l_outcome"], 0) + 1

    assert sum(v8n_counts.values()) == 69
    assert sum(v8m_counts.values()) == 69
    assert sum(rt_counts.values()) == 69

    # Verify specific outcome reconciliations
    assert v8n_counts["correctly_associated"] == 31
    assert v8n_counts["incorrectly_associated"] == 2
    assert v8n_counts["unassigned_no_vehicle"] == 30
    assert v8n_counts["unassigned_no_candidate"] == 4
    assert v8n_counts["partially_contained_clipped"] == 2

    assert v8m_counts["correctly_associated"] == 50
    assert v8m_counts["incorrectly_associated"] == 5
    assert v8m_counts["unassigned_no_vehicle"] == 10
    assert v8m_counts["unassigned_no_candidate"] == 3
    assert v8m_counts["partially_contained_clipped"] == 1

    assert rt_counts["correctly_associated"] == 48
    assert rt_counts["incorrectly_associated"] == 5
    assert rt_counts["ambiguous"] == 6
    assert rt_counts["unassigned_no_vehicle"] == 6
    assert rt_counts["unassigned_no_candidate"] == 2
    assert rt_counts["partially_contained_clipped"] == 2

    # Verify geometric containment vs true association accuracy
    v8n_contained = sum(1 for r in rows if r["yolov8n_geometrically_contained"] == "True")
    v8m_contained = sum(1 for r in rows if r["yolov8m_geometrically_contained"] == "True")
    rt_contained = sum(1 for r in rows if r["rtdetr_l_geometrically_contained"] == "True")

    assert v8n_contained == 33
    assert v8m_contained == 55
    assert rt_contained == 59


def test_external_protocol_and_mission_013_authorization():
    """Verify external test set protocol and Mission 013 authorization constraints."""
    assert EXTERNAL_PROTOCOL_MD.is_file(), f"Missing {EXTERNAL_PROTOCOL_MD}"
    assert MISSION_013_AUTH_MD.is_file(), f"Missing {MISSION_013_AUTH_MD}"
    assert MISSION_012C_REPORT_MD.is_file(), f"Missing {MISSION_012C_REPORT_MD}"

    with open(EXTERNAL_PROTOCOL_MD, "r", encoding="utf-8") as f:
        proto_text = f.read()

    assert "development_test_v1" in proto_text
    assert "external_test_v2" in proto_text
    assert "500 independently sampled frames" in proto_text
    assert "200 deep-night IR frames" in proto_text
    assert "mIoU" in proto_text
    assert "Cohen's Kappa" in proto_text

    with open(MISSION_013_AUTH_MD, "r", encoding="utf-8") as f:
        auth_text = f.read()

    assert "development_test_v1" in auth_text
    assert "RT-DETR-L" in auth_text
    assert "YOLOv8m" in auth_text
    assert "Apache-2.0" in auth_text
    assert "AGPL-3.0" in auth_text
    assert "seed" in auth_text.lower()
    assert "overfitting" in auth_text.lower()


def test_historical_preservation_and_splits_hash():
    """Verify that original benchmark_splits.json is cryptographically untouched."""
    assert ORIGINAL_SPLITS_FILE.is_file()

    with open(ORIGINAL_SPLITS_FILE, "rb") as f:
        current_hash = hashlib.sha256(f.read()).hexdigest()

    assert current_hash == EXPECTED_SPLITS_SHA256, "Historical benchmark_splits.json has been tampered with!"
