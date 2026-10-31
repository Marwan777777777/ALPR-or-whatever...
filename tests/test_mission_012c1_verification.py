"""Unit and Regression Tests for Mission 012C.1: Independent Event Membership Verification & Lighting Reconciliation."""

import csv
import json
from pathlib import Path
import pytest

from scripts.generate_benchmark_splits import parse_timestamp, get_lighting_condition, cluster_temporal_events, GATE_DIR

ROOT_DIR = Path(__file__).resolve().parent.parent
M12_SPLITS_FILE = ROOT_DIR / "artifacts" / "missions" / "mission_012" / "benchmark_splits.json"
M12C_DIR = ROOT_DIR / "artifacts" / "missions" / "mission_012c"
INDEPENDENT_VERIFICATION_JSON = M12C_DIR / "independent_event_membership_verification.json"
LIGHTING_RECONCILIATION_CSV = M12C_DIR / "lighting_distribution_reconciliation.csv"


def test_filename_uniqueness_and_completeness():
    """Fail if a filename is duplicated across splits or missing from the 340 source captures."""
    disk_files = sorted(list(GATE_DIR.glob("*.jpg")))
    disk_filenames = [f.name for f in disk_files]
    assert len(disk_filenames) == 340, f"Expected 340 source captures, found {len(disk_filenames)}"

    with open(M12_SPLITS_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_files = splits["train"]["filenames"]
    val_files = splits["val"]["filenames"]
    test_files = splits["test"]["filenames"]

    all_split_files = train_files + val_files + test_files
    assert len(all_split_files) == 340

    # Fail if duplicate filename in splits
    assert len(all_split_files) == len(set(all_split_files)), "Duplicate filename detected across splits!"

    # Fail if missing filename
    assert set(all_split_files) == set(disk_filenames), "Mismatch between source filenames and split filenames!"

    # Fail if inter-split frame intersection
    assert len(set(train_files) & set(val_files)) == 0, "Train and Val share frames!"
    assert len(set(train_files) & set(test_files)) == 0, "Train and Test share frames!"
    assert len(set(val_files) & set(test_files)) == 0, "Val and Test share frames!"


def test_event_disjointness_and_no_spanning_events():
    """Fail if an event ID is shared across splits or if a reconstructed temporal event spans multiple splits."""
    disk_files = sorted(list(GATE_DIR.glob("*.jpg")))
    reconstructed_events = cluster_temporal_events(disk_files, gap_threshold_sec=20.0)
    assert len(reconstructed_events) == 291

    with open(M12_SPLITS_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_events = set(splits["train"]["event_ids"])
    val_events = set(splits["val"]["event_ids"])
    test_events = set(splits["test"]["event_ids"])

    # Fail if event ID overlap
    assert len(train_events & val_events) == 0, "Train and Val share event IDs!"
    assert len(train_events & test_events) == 0, "Train and Test share event IDs!"
    assert len(val_events & test_events) == 0, "Val and Test share event IDs!"

    # Map each frame to its assigned split
    file_to_split = {}
    for fn in splits["train"]["filenames"]:
        file_to_split[fn] = "train"
    for fn in splits["val"]["filenames"]:
        file_to_split[fn] = "val"
    for fn in splits["test"]["filenames"]:
        file_to_split[fn] = "test"

    # Fail if any temporal event spans multiple splits
    for ev in reconstructed_events:
        splits_present = set(file_to_split.get(f["filename"]) for f in ev["frames"])
        assert len(splits_present) == 1, f"Event {ev['event_id']} spans multiple splits: {splits_present}"


def test_stored_vs_reconstructed_event_membership():
    """Fail if stored event-to-frame membership differs from independently reconstructed membership."""
    disk_files = sorted(list(GATE_DIR.glob("*.jpg")))
    reconstructed_events = cluster_temporal_events(disk_files, gap_threshold_sec=20.0)
    recon_event_map = {e["event_id"]: [f["filename"] for f in e["frames"]] for e in reconstructed_events}

    with open(M12_SPLITS_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    for split_name in ["train", "val", "test"]:
        stored_event_ids = splits[split_name]["event_ids"]
        stored_filenames = splits[split_name]["filenames"]

        reconstructed_split_filenames = []
        for ev_id in stored_event_ids:
            assert ev_id in recon_event_map, f"Stored event ID {ev_id} not found in reconstructed events!"
            reconstructed_split_filenames.extend(recon_event_map[ev_id])

        assert reconstructed_split_filenames == stored_filenames, (
            f"Exact event membership mismatch in {split_name} split! "
            f"Stored {len(stored_filenames)} frames, reconstructed {len(reconstructed_split_filenames)} frames."
        )


def test_source_derived_lighting_breakdown():
    """Fail if lighting breakdown differs from source-derived counts directly computed from filenames."""
    disk_files = sorted(list(GATE_DIR.glob("*.jpg")))

    with open(M12_SPLITS_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    file_to_split = {}
    for s_name in ["train", "val", "test"]:
        for fn in splits[s_name]["filenames"]:
            file_to_split[fn] = s_name

    computed_counts = {
        "train": {"NIGHT_IR": 0, "DAYLIGHT": 0, "EVENING_IR": 0},
        "val": {"NIGHT_IR": 0, "DAYLIGHT": 0, "EVENING_IR": 0},
        "test": {"NIGHT_IR": 0, "DAYLIGHT": 0, "EVENING_IR": 0},
    }

    for f in disk_files:
        dt, _ = parse_timestamp(f.name)
        cond = get_lighting_condition(dt)
        split = file_to_split[f.name]
        computed_counts[split][cond] += 1

    # Verify exact source-derived counts
    assert computed_counts["train"] == {"NIGHT_IR": 6, "DAYLIGHT": 122, "EVENING_IR": 68}
    assert computed_counts["val"] == {"NIGHT_IR": 2, "DAYLIGHT": 43, "EVENING_IR": 27}
    assert computed_counts["test"] == {"NIGHT_IR": 2, "DAYLIGHT": 44, "EVENING_IR": 26}

    # Verify reconciliation CSV matches exactly
    assert LIGHTING_RECONCILIATION_CSV.is_file(), f"Missing {LIGHTING_RECONCILIATION_CSV}"
    with open(LIGHTING_RECONCILIATION_CSV, "r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))

    assert len(csv_rows) == 340
    csv_counts = {
        "train": {"NIGHT_IR": 0, "DAYLIGHT": 0, "EVENING_IR": 0},
        "val": {"NIGHT_IR": 0, "DAYLIGHT": 0, "EVENING_IR": 0},
        "test": {"NIGHT_IR": 0, "DAYLIGHT": 0, "EVENING_IR": 0},
    }
    for r in csv_rows:
        csv_counts[r["dataset_split"]][r["lighting_classification"]] += 1

    assert csv_counts == computed_counts


def test_independent_verification_artifact_and_zero_leakage_condition():
    """Fail if independent verification artifact is missing or if zero_leakage_verified is True while any check fails."""
    assert INDEPENDENT_VERIFICATION_JSON.is_file(), f"Missing {INDEPENDENT_VERIFICATION_JSON}"

    with open(INDEPENDENT_VERIFICATION_JSON, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["audit_milestone"] == "MISSION_012C.1"
    assert audit["source_file_count_on_disk"] == 340
    assert audit["total_reconstructed_events"] == 291
    assert audit["comprehensive_zero_leakage_verified"] is True

    frame_audit = audit["frame_disjointness_audit"]
    assert frame_audit["duplicate_frame_assignments"] == 0
    assert frame_audit["unassigned_source_frames"] == 0
    assert frame_audit["extraneous_split_frames"] == 0
    assert frame_audit["train_val_frame_overlap_count"] == 0
    assert frame_audit["train_test_frame_overlap_count"] == 0
    assert frame_audit["val_test_frame_overlap_count"] == 0

    event_audit = audit["event_disjointness_audit"]
    assert event_audit["train_val_event_overlap_count"] == 0
    assert event_audit["train_test_event_overlap_count"] == 0
    assert event_audit["val_test_event_overlap_count"] == 0
    assert event_audit["events_spanning_multiple_splits_count"] == 0

    # Verify that each split has exact frame sequence match
    membership = audit["membership_comparison_by_split"]
    for s_name in ["train", "val", "test"]:
        assert membership[s_name]["exact_frame_sequence_match"] is True
        assert len(membership[s_name]["missing_frames_in_split"]) == 0
        assert len(membership[s_name]["extraneous_frames_in_split"]) == 0
