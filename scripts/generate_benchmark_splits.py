"""Mission 012: Reproducible Benchmark Splits Generator.

Partitions all 340 gate captures into Train (60%), Validation (20%), and Test (20%)
sets using temporal event-group clustering to prevent data leakage between adjacent frames.
Never uses filename plate strings to establish identity.
Guarantees balanced representation across Night IR, Daylight, and Evening IR regimes.
"""

from __future__ import annotations

from datetime import datetime
import glob
import json
import os
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
GATE_DIR = Path("D:/TruckEntry")
OUT_FILE = ROOT_DIR / "artifacts" / "missions" / "mission_012" / "benchmark_splits.json"


def parse_timestamp(filename: str) -> Tuple[datetime, float]:
    m = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_(\d{3})", filename)
    if not m:
        raise ValueError(f"Could not parse timestamp from {filename}")
    year, month, day, hour, minute, second, ms = [int(v) for v in m.groups()]
    dt = datetime(year, month, day, hour, minute, second, ms * 1000)
    return dt, dt.timestamp()


def get_lighting_condition(dt: datetime) -> str:
    hour = dt.hour + dt.minute / 60.0
    if hour < 6.5:
        return "NIGHT_IR"
    elif hour < 17.5:
        return "DAYLIGHT"
    else:
        return "EVENING_IR"


def cluster_temporal_events(files: List[Path], gap_threshold_sec: float = 20.0) -> List[Dict]:
    """Clusters chronological frames into transit events based on time gap threshold."""
    events = []
    current_frames = []

    for idx, f in enumerate(files):
        fname = f.name
        dt, ts = parse_timestamp(fname)
        lighting = get_lighting_condition(dt)
        sample_id = f"sample_{idx + 1:03d}"
        anon_id = f"SAMPLE-{idx + 1:03d}"

        frame_info = {
            "index": idx,
            "sample_id": sample_id,
            "anonymous_id": anon_id,
            "filename": fname,
            "timestamp": dt.strftime("%H:%M:%S"),
            "full_timestamp": dt.isoformat(),
            "epoch": ts,
            "lighting_condition": lighting,
        }

        if not current_frames:
            current_frames.append(frame_info)
        else:
            gap = ts - current_frames[-1]["epoch"]
            if gap > gap_threshold_sec:
                # Close previous event
                event_id = f"event_{len(events) + 1:03d}"
                events.append({
                    "event_id": event_id,
                    "frame_count": len(current_frames),
                    "lighting_condition": current_frames[0]["lighting_condition"],
                    "start_time": current_frames[0]["timestamp"],
                    "end_time": current_frames[-1]["timestamp"],
                    "start_epoch": current_frames[0]["epoch"],
                    "end_epoch": current_frames[-1]["epoch"],
                    "frames": current_frames,
                })
                current_frames = [frame_info]
            else:
                current_frames.append(frame_info)

    if current_frames:
        event_id = f"event_{len(events) + 1:03d}"
        events.append({
            "event_id": event_id,
            "frame_count": len(current_frames),
            "lighting_condition": current_frames[0]["lighting_condition"],
            "start_time": current_frames[0]["timestamp"],
            "end_time": current_frames[-1]["timestamp"],
            "start_epoch": current_frames[0]["epoch"],
            "end_epoch": current_frames[-1]["epoch"],
            "frames": current_frames,
        })

    return events


def build_stratified_splits(events: List[Dict]) -> Dict:
    """Stratifies events across lighting regimes into ~60% train, ~20% val, ~20% test."""
    by_lighting: Dict[str, List[Dict]] = {
        "NIGHT_IR": [],
        "DAYLIGHT": [],
        "EVENING_IR": [],
    }

    for e in events:
        by_lighting[e["lighting_condition"]].append(e)

    splits: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

    for regime, regime_events in by_lighting.items():
        n = len(regime_events)
        # 60% train, 20% val, 20% test
        n_train = int(round(n * 0.60))
        n_val = int(round(n * 0.20))
        n_test = n - n_train - n_val

        # Chronological slice within regime
        train_events = regime_events[:n_train]
        val_events = regime_events[n_train:n_train + n_val]
        test_events = regime_events[n_train + n_val:]

        splits["train"].extend(train_events)
        splits["val"].extend(val_events)
        splits["test"].extend(test_events)

    # Format output
    result = {
        "metadata": {
            "total_frames": sum(len(e["frames"]) for e in events),
            "total_events": len(events),
            "clustering_gap_threshold_seconds": 20.0,
            "strategy": "Stratified chronological event-block partitioning (zero intra-event leakage)",
            "allocation": {
                "train": {
                    "event_count": len(splits["train"]),
                    "frame_count": sum(e["frame_count"] for e in splits["train"]),
                    "pct_frames": round(sum(e["frame_count"] for e in splits["train"]) / 340.0 * 100, 2),
                },
                "val": {
                    "event_count": len(splits["val"]),
                    "frame_count": sum(e["frame_count"] for e in splits["val"]),
                    "pct_frames": round(sum(e["frame_count"] for e in splits["val"]) / 340.0 * 100, 2),
                },
                "test": {
                    "event_count": len(splits["test"]),
                    "frame_count": sum(e["frame_count"] for e in splits["test"]),
                    "pct_frames": round(sum(e["frame_count"] for e in splits["test"]) / 340.0 * 100, 2),
                },
            },
            "lighting_distribution_by_split": {
                "train": {
                    "NIGHT_IR": sum(e["frame_count"] for e in splits["train"] if e["lighting_condition"] == "NIGHT_IR"),
                    "DAYLIGHT": sum(e["frame_count"] for e in splits["train"] if e["lighting_condition"] == "DAYLIGHT"),
                    "EVENING_IR": sum(e["frame_count"] for e in splits["train"] if e["lighting_condition"] == "EVENING_IR"),
                },
                "val": {
                    "NIGHT_IR": sum(e["frame_count"] for e in splits["val"] if e["lighting_condition"] == "NIGHT_IR"),
                    "DAYLIGHT": sum(e["frame_count"] for e in splits["val"] if e["lighting_condition"] == "DAYLIGHT"),
                    "EVENING_IR": sum(e["frame_count"] for e in splits["val"] if e["lighting_condition"] == "EVENING_IR"),
                },
                "test": {
                    "NIGHT_IR": sum(e["frame_count"] for e in splits["test"] if e["lighting_condition"] == "NIGHT_IR"),
                    "DAYLIGHT": sum(e["frame_count"] for e in splits["test"] if e["lighting_condition"] == "DAYLIGHT"),
                    "EVENING_IR": sum(e["frame_count"] for e in splits["test"] if e["lighting_condition"] == "EVENING_IR"),
                },
            },
        },
        "train": {
            "event_ids": [e["event_id"] for e in splits["train"]],
            "sample_ids": [f["sample_id"] for e in splits["train"] for f in e["frames"]],
            "anonymous_ids": [f["anonymous_id"] for e in splits["train"] for f in e["frames"]],
            "filenames": [f["filename"] for e in splits["train"] for f in e["frames"]],
        },
        "val": {
            "event_ids": [e["event_id"] for e in splits["val"]],
            "sample_ids": [f["sample_id"] for e in splits["val"] for f in e["frames"]],
            "anonymous_ids": [f["anonymous_id"] for e in splits["val"] for f in e["frames"]],
            "filenames": [f["filename"] for e in splits["val"] for f in e["frames"]],
        },
        "test": {
            "event_ids": [e["event_id"] for e in splits["test"]],
            "sample_ids": [f["sample_id"] for e in splits["test"] for f in e["frames"]],
            "anonymous_ids": [f["anonymous_id"] for e in splits["test"] for f in e["frames"]],
            "filenames": [f["filename"] for e in splits["test"] for f in e["frames"]],
        },
    }
    return result


def main():
    files = sorted(list(GATE_DIR.glob("*.jpg")))
    print(f"Loaded {len(files)} gate captures from {GATE_DIR}")
    events = cluster_temporal_events(files)
    print(f"Formed {len(events)} temporal transit events.")

    splits_data = build_stratified_splits(events)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2, ensure_ascii=False)

    print(f"Benchmark splits saved to: {OUT_FILE}")
    alloc = splits_data["metadata"]["allocation"]
    print(f"  Train: {alloc['train']['event_count']} events, {alloc['train']['frame_count']} frames ({alloc['train']['pct_frames']}%)")
    print(f"  Val:   {alloc['val']['event_count']} events, {alloc['val']['frame_count']} frames ({alloc['val']['pct_frames']}%)")
    print(f"  Test:  {alloc['test']['event_count']} events, {alloc['test']['frame_count']} frames ({alloc['test']['pct_frames']}%)")


if __name__ == "__main__":
    main()
