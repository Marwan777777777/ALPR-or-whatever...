"""Unit and Regression Tests for Mission 012B: Benchmark Integrity Correction & Domain Training Readiness."""

import csv
import json
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
M12B_DIR = ROOT_DIR / "artifacts" / "missions" / "mission_012b"

SPLIT_BENCHMARK_CSV = M12B_DIR / "split_specific_benchmark.csv"
PER_LIGHTING_TEST_CSV = M12B_DIR / "per_lighting_test_metrics.csv"
INTEGRITY_JSON = M12B_DIR / "temporal_split_integrity.json"
ASSOC_RECONCILIATION_CSV = M12B_DIR / "association_reconciliation.csv"
ASSOC_TEST_METRICS_CSV = M12B_DIR / "association_test_metrics.csv"
REVIEW_QUEUE_CSV = M12B_DIR / "annotation_review_queue.csv"
LICENSING_MATRIX_MD = M12B_DIR / "licensing_matrix.md"
TRAINING_READINESS_MD = M12B_DIR / "mission_013_training_readiness.md"


def test_split_specific_benchmark_metrics_isolated():
    """Verify that benchmark metrics are cleanly isolated for train, val, test, and descriptive full."""
    assert SPLIT_BENCHMARK_CSV.is_file(), f"Missing {SPLIT_BENCHMARK_CSV}"

    with open(SPLIT_BENCHMARK_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 12 # 3 models * 4 splits

    models = set(r["model_id"] for r in rows)
    assert models == {"yolov8n", "yolov8m", "rtdetr-l"}

    splits = set(r["dataset_split"] for r in rows)
    assert splits == {
        "held_out_test_set",
        "validation_set",
        "training_set",
        "entire_dataset_in_sample_descriptive",
    }

    # Verify held-out test set frame count is exactly 72
    test_rows = [r for r in rows if r["dataset_split"] == "held_out_test_set"]
    assert len(test_rows) == 3
    for tr in test_rows:
        assert int(tr["frame_count"]) == 72
        assert 0.0 <= float(tr["precision_50"]) <= 1.0
        assert 0.0 <= float(tr["recall_50"]) <= 1.0
        assert 0.0 <= float(tr["f1_score_50"]) <= 1.0


def test_temporal_split_integrity_audit_passed():
    """Verify machine-checkable zero leakage audit across all 340 gate frames and 291 events."""
    assert INTEGRITY_JSON.is_file(), f"Missing {INTEGRITY_JSON}"

    with open(INTEGRITY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "PASSED"
    assert data["total_dataset_files"] == 340
    assert data["total_reconstructed_events"] == 291

    audit = data["leakage_audit"]
    assert audit["zero_leakage_verified"] is True
    assert audit["train_val_frame_overlap_count"] == 0
    assert audit["train_test_frame_overlap_count"] == 0
    assert audit["val_test_frame_overlap_count"] == 0
    assert audit["events_spanning_multiple_splits_count"] == 0
    assert audit["unassigned_frames_count"] == 0
    assert audit["duplicate_assignments_count"] == 0


def test_association_reconciliation_closure_322():
    """Verify that all 322 plate observations receive exactly one mutually exclusive outcome."""
    assert ASSOC_RECONCILIATION_CSV.is_file(), f"Missing {ASSOC_RECONCILIATION_CSV}"
    assert ASSOC_TEST_METRICS_CSV.is_file(), f"Missing {ASSOC_TEST_METRICS_CSV}"

    with open(ASSOC_RECONCILIATION_CSV, "r", encoding="utf-8") as f:
        reconcile_rows = list(csv.DictReader(f))

    assert len(reconcile_rows) == 322, f"Expected 322 plate observations, found {len(reconcile_rows)}"

    valid_outcomes = {
        "correctly_associated",
        "incorrectly_associated",
        "unassigned_no_vehicle",
        "unassigned_no_candidate",
        "ambiguous",
        "partially_contained_clipped",
    }

    for r in reconcile_rows:
        assert r["yolov8n_outcome"] in valid_outcomes
        assert r["yolov8m_outcome"] in valid_outcomes
        assert r["rtdetr-l_outcome"] in valid_outcomes

    with open(ASSOC_TEST_METRICS_CSV, "r", encoding="utf-8") as f:
        metric_rows = list(csv.DictReader(f))

    assert len(metric_rows) == 6 # 3 models * 2 splits (test and descriptive)
    for mr in metric_rows:
        assert mr["accounting_sum_verified"] == "True"


def test_annotation_review_queue_and_licensing():
    """Verify that review queue covers all 17 adjudications and licensing matrix documents AGPL constraints."""
    assert REVIEW_QUEUE_CSV.is_file(), f"Missing {REVIEW_QUEUE_CSV}"
    assert LICENSING_MATRIX_MD.is_file(), f"Missing {LICENSING_MATRIX_MD}"
    assert TRAINING_READINESS_MD.is_file(), f"Missing {TRAINING_READINESS_MD}"

    with open(REVIEW_QUEUE_CSV, "r", encoding="utf-8") as f:
        queue_rows = list(csv.DictReader(f))

    adj_count = sum(1 for r in queue_rows if "ADJUDICATION_FLAGGED" in r["review_categories"])
    assert adj_count == 17, f"Expected 17 adjudication flagged items, found {adj_count}"

    with open(LICENSING_MATRIX_MD, "r", encoding="utf-8") as f:
        licensing_text = f.read()

    assert "AGPL-3.0" in licensing_text
    assert "Section 13" in licensing_text
    assert "Apache-2.0" in licensing_text
    assert "legal advice" in licensing_text.lower()
