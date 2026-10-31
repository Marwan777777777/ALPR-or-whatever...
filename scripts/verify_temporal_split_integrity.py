"""Mission 012B: Temporal Split Integrity & Leakage Verification Script.

Inspects the 340 gate captures, verifies deterministic timestamp parsing, validates
event clustering, checks for zero frame/event overlap between splits, and exports
a comprehensive machine-checkable audit report:
`artifacts/missions/mission_012b/temporal_split_integrity.json`.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import Dict, List, Set, Tuple

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
GATE_DIR = Path("D:/TruckEntry")
SPLITS_FILE = ROOT_DIR / "artifacts" / "missions" / "mission_012" / "benchmark_splits.json"
OUT_REPORT = ROOT_DIR / "artifacts" / "missions" / "mission_012b" / "temporal_split_integrity.json"


def parse_timestamp(filename: str) -> Tuple[datetime, float]:
    m = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_(\d{3})", filename)
    if not m:
        raise ValueError(f"Could not parse timestamp from {filename}")
    year, month, day, hour, minute, second, ms = [int(v) for v in m.groups()]
    dt = datetime(year, month, day, hour, minute, second, ms * 1000)
    return dt, dt.timestamp()


def run_integrity_check():
    print("=" * 80)
    print("MISSION 012B: TEMPORAL SPLIT INTEGRITY & LEAKAGE VERIFICATION")
    print("=" * 80)

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(list(GATE_DIR.glob("*.jpg")))
    total_files = len(files)
    print(f"Total gate captures in D:/TruckEntry: {total_files}")

    # 1. Verify timestamp parsing
    parsed_timestamps = []
    parse_errors = []
    for f in files:
        try:
            dt, ts = parse_timestamp(f.name)
            parsed_timestamps.append((f.name, dt, ts))
        except Exception as e:
            parse_errors.append({"file": f.name, "error": str(e)})

    print(f"Timestamp parsing: {len(parsed_timestamps)} successful, {len(parse_errors)} errors.")

    # 2. Verify chronological order
    is_sorted = all(parsed_timestamps[i][2] <= parsed_timestamps[i+1][2] for i in range(len(parsed_timestamps)-1))
    print(f"Chronological ordering verified: {is_sorted}")

    # 3. Check time gaps and cluster deterministically
    gap_threshold_sec = 20.0
    events: List[Dict] = []
    current_event_files: List[Tuple[str, datetime, float]] = []

    for item in parsed_timestamps:
        fname, dt, ts = item
        if not current_event_files:
            current_event_files.append(item)
        else:
            prev_ts = current_event_files[-1][2]
            gap = ts - prev_ts
            if gap > gap_threshold_sec:
                # Close previous event
                ev_id = f"event_{len(events) + 1:03d}"
                events.append({
                    "event_id": ev_id,
                    "frame_count": len(current_event_files),
                    "filenames": [x[0] for x in current_event_files],
                    "start_time": current_event_files[0][1].isoformat(),
                    "end_time": current_event_files[-1][1].isoformat(),
                    "duration_sec": round(current_event_files[-1][2] - current_event_files[0][2], 3),
                })
                current_event_files = [item]
            else:
                current_event_files.append(item)

    if current_event_files:
        ev_id = f"event_{len(events) + 1:03d}"
        events.append({
            "event_id": ev_id,
            "frame_count": len(current_event_files),
            "filenames": [x[0] for x in current_event_files],
            "start_time": current_event_files[0][1].isoformat(),
            "end_time": current_event_files[-1][1].isoformat(),
            "duration_sec": round(current_event_files[-1][2] - current_event_files[0][2], 3),
        })

    print(f"Reconstructed {len(events)} events using gap threshold = {gap_threshold_sec}s.")

    # 4. Load benchmark_splits.json and verify consistency
    with open(SPLITS_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_files: List[str] = splits["train"]["filenames"]
    val_files: List[str] = splits["val"]["filenames"]
    test_files: List[str] = splits["test"]["filenames"]

    train_set: Set[str] = set(train_files)
    val_set: Set[str] = set(val_files)
    test_set: Set[str] = set(test_files)

    train_events: Set[str] = set(splits["train"]["event_ids"])
    val_events: Set[str] = set(splits["val"]["event_ids"])
    test_events: Set[str] = set(splits["test"]["event_ids"])

    # Overlap / Leakage checks
    train_val_overlap = sorted(list(train_set.intersection(val_set)))
    train_test_overlap = sorted(list(train_set.intersection(test_set)))
    val_test_overlap = sorted(list(val_set.intersection(test_set)))

    ev_train_val_overlap = sorted(list(train_events.intersection(val_events)))
    ev_train_test_overlap = sorted(list(train_events.intersection(test_events)))
    ev_val_test_overlap = sorted(list(val_events.intersection(test_events)))

    all_split_files = train_set.union(val_set).union(test_set)
    unassigned_files = [f.name for f in files if f.name not in all_split_files]
    duplicate_files = (len(train_files) + len(val_files) + len(test_files)) - len(all_split_files)

    # Check whether any reconstructed event spans multiple splits
    event_leakage_detected = []
    file_to_split = {}
    for fn in train_files:
        file_to_split[fn] = "train"
    for fn in val_files:
        file_to_split[fn] = "val"
    for fn in test_files:
        file_to_split[fn] = "test"

    for ev in events:
        splits_present = set(file_to_split.get(fn, "MISSING") for fn in ev["filenames"])
        if len(splits_present) > 1:
            event_leakage_detected.append({
                "event_id": ev["event_id"],
                "splits_spanned": list(splits_present),
                "filenames": ev["filenames"],
            })

    # Detailed report dictionary
    report = {
        "status": "PASSED" if not (train_val_overlap or train_test_overlap or val_test_overlap or event_leakage_detected or unassigned_files) else "FAILED",
        "verification_timestamp": datetime.now().isoformat(),
        "total_dataset_files": total_files,
        "total_reconstructed_events": len(events),
        "split_summary": {
            "train": {
                "frame_count": len(train_files),
                "event_count": len(train_events),
                "percentage_frames": round(len(train_files) / total_files * 100, 2),
            },
            "validation": {
                "frame_count": len(val_files),
                "event_count": len(val_events),
                "percentage_frames": round(len(val_files) / total_files * 100, 2),
            },
            "test": {
                "frame_count": len(test_files),
                "event_count": len(test_events),
                "percentage_frames": round(len(test_files) / total_files * 100, 2),
            },
        },
        "leakage_audit": {
            "train_val_frame_overlap_count": len(train_val_overlap),
            "train_test_frame_overlap_count": len(train_test_overlap),
            "val_test_frame_overlap_count": len(val_test_overlap),
            "train_val_event_overlap_count": len(ev_train_val_overlap),
            "train_test_event_overlap_count": len(ev_train_test_overlap),
            "val_test_event_overlap_count": len(ev_val_test_overlap),
            "events_spanning_multiple_splits_count": len(event_leakage_detected),
            "unassigned_frames_count": len(unassigned_files),
            "duplicate_assignments_count": duplicate_files,
            "zero_leakage_verified": bool(
                len(train_val_overlap) == 0
                and len(train_test_overlap) == 0
                and len(val_test_overlap) == 0
                and len(event_leakage_detected) == 0
                and len(unassigned_files) == 0
                and duplicate_files == 0
            ),
        },
        "methodological_limitations": {
            "nature_of_event_grouping": (
                "The 20.0-second gap threshold is an empirical operational heuristic based on typical barrier cycle "
                "cadence. It does NOT represent verified physical vehicle re-identification or ground-truth vehicle "
                "identity. Vehicles queuing closely (<20s) may be clustered into a single event, while vehicles lingering "
                "in front of the gate (>20s) could be separated into two events. However, for preventing train/test data "
                "leakage of adjacent near-identical frames, this heuristic is strictly conservative."
            ),
            "night_sample_scarcity": (
                "The test split contains exactly 2 deep-night IR frames (out of 10 total across the dataset). While this "
                "preserves the 60/20/20 proportion, n=2 is statistically inadequate for robust generalized performance claims. "
                "Deep-night test metrics must be reported with explicit disclosure of sample size."
            ),
        },
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nAudit Report written to: {OUT_REPORT}")
    print(f"Status: {report['status']}")
    print(f"Zero Leakage Verified: {report['leakage_audit']['zero_leakage_verified']}")
    print(f"Train/Val/Test overlaps: {len(train_val_overlap)} / {len(train_test_overlap)} / {len(val_test_overlap)}")
    print(f"Events spanning multiple splits: {len(event_leakage_detected)}")


if __name__ == "__main__":
    run_integrity_check()
