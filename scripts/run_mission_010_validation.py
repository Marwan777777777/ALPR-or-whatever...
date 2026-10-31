"""Mission 010 Gate Validation and Analysis Script.

Processes gate footage chronologically, evaluates vehicle detection, vehicle tracking,
plate-to-vehicle association, and transit event generation without using filename plate text
as ground truth.
"""

from __future__ import annotations

import csv
from datetime import datetime
import glob
import os
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

sys.stdout.reconfigure(encoding="utf-8")

from vehicle_intelligence.core.domain import (
    AssociationStatus,
    BoundingBox,
    PlateDetection,
    PlateTextResult,
    VehicleClass,
    VehicleDetection,
    VehicleTransitEvent,
)
from vehicle_intelligence.detectors.pytorch_yolo import PyTorchYoloVehicleDetector
from vehicle_intelligence.tracking.tracker import CentroidIoUTracker
from vehicle_intelligence.association.plate_vehicle_associator import PlateVehicleAssociator
from vehicle_intelligence.tracking.aggregator import MultiFrameEvidenceAggregator
from vehicle_intelligence.pipeline.alpr_pipeline import EndToEndALPRPipeline


def parse_timestamp_from_filename(filename: str) -> Tuple[datetime, float]:
    """Extracts timestamp from filename prefix YYYYMMDD_HHMMSS_mmm without using plate metadata."""
    m = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_(\d{3})", filename)
    if not m:
        raise ValueError(f"Could not parse timestamp from {filename}")
    year, month, day, hour, minute, second, ms = [int(v) for v in m.groups()]
    dt = datetime(year, month, day, hour, minute, second, ms * 1000)
    return dt, dt.timestamp()


def run_mission_010_validation():
    print("=" * 80)
    print("MISSION 010: VEHICLE DETECTION, TRACKING, AND ASSOCIATION VALIDATION")
    print("=" * 80)

    out_dir = Path("artifacts/missions/mission_010")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover gate images
    gate_files = sorted(glob.glob("D:/TruckEntry/*.jpg"))
    if not gate_files:
        print("Warning: D:/TruckEntry not found. Checking local triage samples...")
        gate_files = sorted(glob.glob("artifacts/triage/samples/*.jpg"))

    print(f"Discovered {len(gate_files)} gate images.")

    # 2. Parse timestamps and sort chronologically
    records = []
    for f in gate_files:
        fname = os.path.basename(f)
        try:
            dt, ts = parse_timestamp_from_filename(fname)
            records.append({
                "path": f,
                "filename": fname,
                "datetime": dt,
                "timestamp": ts,
            })
        except Exception as e:
            # Fallback if filename lacks standard timestamp format
            mtime = os.path.getmtime(f)
            records.append({
                "path": f,
                "filename": fname,
                "datetime": datetime.fromtimestamp(mtime),
                "timestamp": mtime,
            })

    records.sort(key=lambda r: r["timestamp"])
    print(f"Chronologically sorted {len(records)} frames.")
    print(f"Time range: {records[0]['datetime']} to {records[-1]['datetime']}")

    # 3. Initialize pipeline components
    print("Initializing ALPR and Vehicle Pipeline components...")
    v_detector = PyTorchYoloVehicleDetector(
        weights_path="yolov8n.pt",
        confidence_threshold=0.20,
        iou_threshold=0.45,
        device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 or os.path.exists("yolov8n.pt") else "cpu",
    )
    associator = PlateVehicleAssociator(
        min_containment_iop=0.45,
        confident_containment_iop=0.75,
        min_association_score=0.35,
        confident_association_score=0.60,
        ambiguity_margin=0.15,
    )
    # Tracker with max_lost_frames=5 (at 10s gap, inactive tracks will expire naturally)
    tracker = CentroidIoUTracker(max_lost_frames=5, iou_threshold=0.20)
    aggregator = MultiFrameEvidenceAggregator(min_confidence_threshold=0.70)

    # ALPR pipeline for plate detection and character OCR
    alpr = EndToEndALPRPipeline(
        detector_weights="artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt",
        char_weights="artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt",
        det_conf=0.25,
        char_conf=0.20,
        agnostic_nms=True,
        char_iou=0.45,
    )

    # 4. Process all frames
    print("Processing frames chronologically...")
    detection_rows = []
    track_rows = []
    assoc_rows = []
    all_finalized_events = []

    last_ts = None
    time_gaps = []

    for idx, rec in enumerate(records):
        fpath = rec["path"]
        fname = rec["filename"]
        ts = rec["timestamp"]

        delta_t = (ts - last_ts) if last_ts is not None else 0.0
        time_gaps.append(delta_t)
        last_ts = ts

        # If gap is very large (> 15 seconds), vehicle tracker should reset lost tracks
        if delta_t > 15.0:
            tracker.update([], frame_id=idx, timestamp=ts)

        img = cv2.imread(fpath)
        if img is None:
            continue
        h, w = img.shape[:2]

        # 4a. Run vehicle detector
        v_dets = v_detector.detect(img)

        # Record vehicle detection results
        v_classes = [d.vehicle_class.value for d in v_dets]
        v_confs = [round(d.confidence, 4) for d in v_dets]
        v_boxes = [[int(x) for x in d.box.to_xyxy()] for d in v_dets]

        detection_rows.append({
            "frame_id": idx + 1,
            "filename": fname,
            "timestamp": ts,
            "datetime": rec["datetime"].isoformat(),
            "time_gap_seconds": round(delta_t, 3),
            "image_width": w,
            "image_height": h,
            "num_vehicles_detected": len(v_dets),
            "vehicle_classes": ";".join(v_classes) if v_classes else "NONE",
            "confidences": ";".join(str(c) for c in v_confs) if v_confs else "NONE",
            "boxes": ";".join(str(b) for b in v_boxes) if v_boxes else "NONE",
        })

        # 4b. Run vehicle tracker
        tracked_vehicles = tracker.update(v_dets, frame_id=idx + 1, timestamp=ts)

        # 4c. Run ALPR pipeline (Plate detection + Character OCR)
        alpr_results = alpr.read_plate_image(image=img, source_id=fname, timestamp=ts)

        plate_dets = [
            PlateDetection(box=r.bounding_box, confidence=r.detector_confidence)
            for r in alpr_results
        ]

        # 4d. Associate plates with vehicles
        associations = associator.associate(plate_dets, tracked_vehicles, frame_id=idx + 1, timestamp=ts)

        for p_idx, (r, p_det, assoc) in enumerate(zip(alpr_results, plate_dets, associations)):
            target_track_id = assoc.vehicle_track_id

            assoc_rows.append({
                "frame_id": idx + 1,
                "filename": fname,
                "plate_index": p_idx + 1,
                "plate_box": str([int(x) for x in p_det.box.to_xyxy()]),
                "plate_detector_conf": round(p_det.confidence, 4),
                "plate_ocr_text": r.plate_text,
                "plate_ocr_conf": round(r.plate_confidence, 4),
                "vehicle_track_id": target_track_id if target_track_id is not None else "UNASSIGNED",
                "association_status": assoc.status.value,
                "association_confidence": round(assoc.association_confidence, 4),
                "containment_iop": round(assoc.containment_ratio, 4),
                "competing_vehicle_ids": ";".join(str(cid) for cid in assoc.competing_vehicle_ids) if assoc.competing_vehicle_ids else "NONE",
                "needs_review": assoc.needs_review,
                "review_reasons": ";".join(assoc.review_reasons) if assoc.review_reasons else "NONE",
            })

            # Record in aggregator if vehicle track exists
            if target_track_id is not None:
                matched_v = next((v for v in tracked_vehicles if v.track_id == target_track_id), None)
                if matched_v:
                    ocr_res = PlateTextResult(
                        raw_text=r.plate_text,
                        arabic_letters=r.letters,
                        arabic_digits=r.digits,
                        confidence=r.plate_confidence,
                        needs_review=r.needs_review,
                        review_reasons=r.review_reasons,
                    )
                    aggregator.record_observation(
                        track_id=target_track_id,
                        timestamp=ts,
                        vehicle_detection=matched_v,
                        plate_detection=p_det,
                        plate_reading=ocr_res,
                        plate_association=assoc,
                        frame_id=idx + 1,
                        crop_path=r.crop_path,
                        source_id=fname,
                    )

        # Finalize any completed tracks
        if delta_t > 15.0:
            events = aggregator.finalize_all_transit_events()
            all_finalized_events.extend(events)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(records):
            print(f"Processed {idx + 1}/{len(records)} frames...")

    # Finalize any remaining tracks
    remaining_events = aggregator.finalize_all_transit_events()
    all_finalized_events.extend(remaining_events)

    print(f"Processing complete. Finalized {len(all_finalized_events)} vehicle transit events.")

    # 5. Export vehicle_detection_results.csv
    det_csv = out_dir / "vehicle_detection_results.csv"
    with open(det_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detection_rows[0].keys())
        writer.writeheader()
        writer.writerows(detection_rows)
    print(f"Exported: {det_csv}")

    # 6. Export plate_vehicle_association_results.csv
    assoc_csv = out_dir / "plate_vehicle_association_results.csv"
    with open(assoc_csv, "w", newline="", encoding="utf-8") as f:
        if assoc_rows:
            writer = csv.DictWriter(f, fieldnames=assoc_rows[0].keys())
            writer.writeheader()
            writer.writerows(assoc_rows)
        else:
            f.write("frame_id,filename,plate_index,plate_box,plate_detector_conf,plate_ocr_text,plate_ocr_conf,vehicle_track_id,association_status,association_confidence,containment_iop,competing_vehicle_ids,needs_review,review_reasons\n")
    print(f"Exported: {assoc_csv}")

    # 7. Export vehicle_track_analysis.csv
    # Analyze track continuity and gaps
    track_rows = []
    for ev in all_finalized_events:
        span = ev.last_seen_timestamp - ev.first_seen_timestamp
        is_genuine = (ev.total_frames_tracked > 1 and span < 30.0)
        gap_note = "Single snapshot trigger" if ev.total_frames_tracked == 1 else (
            f"Multi-frame continuous sequence (span={span:.1f}s)" if is_genuine else f"Intermittent triggers across {span:.1f}s"
        )
        track_rows.append({
            "vehicle_track_id": ev.vehicle_track_id,
            "vehicle_class": ev.vehicle_class.value,
            "vehicle_confidence": round(ev.vehicle_confidence, 4),
            "first_seen_timestamp": ev.first_seen_timestamp,
            "last_seen_timestamp": ev.last_seen_timestamp,
            "time_span_seconds": round(span, 2),
            "total_frames_tracked": ev.total_frames_tracked,
            "is_genuine_continuous_trajectory": is_genuine,
            "observation_gap_notes": gap_note,
            "associated_plate_text": ev.plate_text,
            "plate_association_status": ev.association_status.value,
            "needs_review": ev.needs_review,
            "review_reasons": ";".join(ev.review_reasons) if ev.review_reasons else "NONE",
        })

    track_csv = out_dir / "vehicle_track_analysis.csv"
    with open(track_csv, "w", newline="", encoding="utf-8") as f:
        if track_rows:
            writer = csv.DictWriter(f, fieldnames=track_rows[0].keys())
            writer.writeheader()
            writer.writerows(track_rows)
        else:
            f.write("vehicle_track_id,vehicle_class,vehicle_confidence,first_seen_timestamp,last_seen_timestamp,time_span_seconds,total_frames_tracked,is_genuine_continuous_trajectory,observation_gap_notes,associated_plate_text,plate_association_status,needs_review,review_reasons\n")
    print(f"Exported: {track_csv}")

    # 8. Export event_examples.csv
    event_csv = out_dir / "event_examples.csv"
    event_rows = []
    for ev in all_finalized_events[:50]:  # Top 50 representative events
        event_rows.append({
            "event_type": ev.event_type,
            "vehicle_track_id": ev.vehicle_track_id,
            "vehicle_class": ev.vehicle_class.value,
            "vehicle_confidence": round(ev.vehicle_confidence, 4),
            "first_seen_timestamp": ev.first_seen_timestamp,
            "last_seen_timestamp": ev.last_seen_timestamp,
            "total_frames_tracked": ev.total_frames_tracked,
            "plate_text": ev.plate_text,
            "letters": ev.letters,
            "digits": ev.digits,
            "plate_confidence": round(ev.plate_confidence, 4),
            "plate_association_confidence": round(ev.plate_association_confidence, 4),
            "association_status": ev.association_status.value,
            "best_vehicle_box": str([int(x) for x in ev.best_vehicle_box.to_xyxy()]),
            "best_plate_box": str([int(x) for x in ev.best_plate_box.to_xyxy()]) if ev.best_plate_box else "NONE",
            "has_conflicting_reads": ev.has_conflicting_reads,
            "has_conflicting_association": ev.has_conflicting_association,
            "competing_vehicle_ids": ";".join(str(c) for c in ev.competing_vehicle_ids) if ev.competing_vehicle_ids else "NONE",
            "needs_review": ev.needs_review,
            "review_reasons": ";".join(ev.review_reasons) if ev.review_reasons else "NONE",
        })

    with open(event_csv, "w", newline="", encoding="utf-8") as f:
        if event_rows:
            writer = csv.DictWriter(f, fieldnames=event_rows[0].keys())
            writer.writeheader()
            writer.writerows(event_rows)
        else:
            f.write("event_type,vehicle_track_id,vehicle_class,vehicle_confidence,first_seen_timestamp,last_seen_timestamp,total_frames_tracked,plate_text,letters,digits,plate_confidence,plate_association_confidence,association_status,best_vehicle_box,best_plate_box,has_conflicting_reads,has_conflicting_association,competing_vehicle_ids,needs_review,review_reasons\n")
    print(f"Exported: {event_csv}")

    print("=" * 80)
    print("VALIDATION SUMMARY")
    print(f"Total gate frames evaluated: {len(records)}")
    print(f"Total vehicle detections recorded: {len(detection_rows)}")
    print(f"Total plate associations recorded: {len(assoc_rows)}")
    print(f"Total vehicle transit events consolidated: {len(all_finalized_events)}")
    print("=" * 80)


if __name__ == "__main__":
    run_mission_010_validation()
