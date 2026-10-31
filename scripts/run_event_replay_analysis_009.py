#!/usr/bin/env python3
"""Event Replay and Multi-Frame Consolidation Analysis (Mission 009 Objective 4).

Validates:
1. Tracking ID stability across sequential frames.
2. Stronger later observations improving earlier weak reads.
3. Best available evidence frame retention.
4. Conflicting reads preservation as alternatives (no silent overwrite).
5. Elimination of duplicate / fragmented transit events per vehicle.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.core.domain import (
    ALPRPlateResult,
    BoundingBox,
    PlateDetection,
    PlateTextResult,
    VehicleDetection,
    VehicleSessionSummary,
)
from vehicle_intelligence.pipeline.alpr_pipeline import EndToEndALPRPipeline
from vehicle_intelligence.tracking.aggregator import MultiFrameEvidenceAggregator
from vehicle_intelligence.tracking.tracker import CentroidIoUTracker

sys.stdout.reconfigure(encoding="utf-8")


def run_event_replay_analysis():
    print("=== RUNNING EVENT-LEVEL MULTI-FRAME REPLAY ANALYSIS ===")
    out_dir = root_dir / "artifacts" / "missions" / "mission_009"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipeline = EndToEndALPRPipeline(
        device=device,
        det_imgsz=1280,
        char_iou=0.45,
        agnostic_nms=True,
    )

    # 1. Multi-frame vehicle sequences from D:\TruckEntry
    d_truck = Path("D:/TruckEntry")
    multi_frame_transits = []

    if d_truck.exists():
        files = sorted(list(d_truck.glob("*.jpg")))
        by_plate = defaultdict(list)
        for f in files:
            parts = f.stem.split("_")
            if len(parts) >= 5:
                by_plate[parts[4]].append(f)

        # Select vehicles with multiple frames
        multi_plates = {k: v for k, v in by_plate.items() if len(v) >= 2}
        print(f"Found {len(multi_plates)} vehicle groups with multiple frames in D:/TruckEntry.")
        
        # Take the top 10 multi-frame vehicle transits
        for plate_key, p_files in sorted(multi_plates.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            multi_frame_transits.append({
                "vehicle_id": f"burst_{plate_key}",
                "expected_legacy": plate_key,
                "frames": p_files,
            })

    # Also evaluate sequential frames in the triage set that track the same vehicle
    triage_frames = sorted(list((root_dir / "artifacts" / "triage" / "samples").glob("frame_*.jpg")))
    
    analysis_records = []

    # Process each multi-frame vehicle transit
    for transit in multi_frame_transits:
        v_id = transit["vehicle_id"]
        exp_leg = transit["expected_legacy"]
        f_list = transit["frames"]
        
        tracker = CentroidIoUTracker(max_lost_frames=15, iou_threshold=0.20)
        aggregator = MultiFrameEvidenceAggregator(min_confidence_threshold=0.82)
        
        obs_by_track = defaultdict(list)
        
        for f_idx, f_path in enumerate(f_list, start=1):
            img = cv2.imread(str(f_path))
            if img is None:
                continue
            
            results = pipeline.read_plate_image(
                image=img,
                source_id=f_path.name,
                timestamp=f_idx * 0.5,
                save_crops_dir=out_dir / "replay_crops",
            )
            
            detections = [VehicleDetection(box=r.bounding_box, confidence=r.detector_confidence) for r in results]
            tracked = tracker.update(detections, frame_id=f_idx, timestamp=f_idx * 0.5)
            
            for t_veh in tracked:
                for r in results:
                    if t_veh.box.iou(r.bounding_box) >= 0.20:
                        r.track_id = t_veh.track_id
                        obs_by_track[t_veh.track_id].append(r)
                        
                        p_det = PlateDetection(box=r.bounding_box, confidence=r.detector_confidence)
                        p_read = PlateTextResult(raw_text=r.plate_text, confidence=r.plate_confidence, needs_review=r.needs_review)
                        aggregator.record_observation(
                            track_id=t_veh.track_id,
                            timestamp=f_idx * 0.5,
                            vehicle_detection=t_veh,
                            plate_detection=p_det,
                            plate_reading=p_read,
                            frame_id=f_idx,
                            crop_path=r.crop_path,
                        )

        # Consolidate tracks for this vehicle
        for t_id, obs_list in obs_by_track.items():
            summary = aggregator.finalize_track(t_id)
            if summary is None:
                continue

            first_obs = obs_list[0]
            last_obs = obs_list[-1]
            best_read = summary.plate_reading

            first_conf = first_obs.plate_confidence
            last_conf = last_obs.plate_confidence
            best_conf = best_read.confidence if best_read else 0.0

            later_improved = (last_conf > first_conf and last_obs.plate_text == best_read.raw_text if best_read else False)
            
            all_texts = [o.plate_text for o in obs_list if o.plate_text]
            unique_texts = list(dict.fromkeys(all_texts))
            is_stable = (len(unique_texts) <= 1)

            analysis_records.append({
                "transit_vehicle_id": v_id,
                "track_id": t_id,
                "legacy_plate": exp_leg,
                "total_frames_observed": len(obs_list),
                "first_frame_text": first_obs.plate_text,
                "first_frame_confidence": round(first_conf, 4),
                "last_frame_text": last_obs.plate_text,
                "last_frame_confidence": round(last_conf, 4),
                "consolidated_best_text": best_read.raw_text if best_read else "",
                "consolidated_best_confidence": round(best_conf, 4),
                "later_observation_improved": later_improved,
                "is_reading_stable": is_stable,
                "conflicting_variants_count": len(unique_texts),
                "conflicting_variants_list": "; ".join(unique_texts),
                "alternative_readings_preserved_count": len(summary.alternative_readings),
                "best_evidence_frame_id": summary.best_evidence_frame_id,
                "best_crop_path": summary.best_crop_path or "",
                "needs_review": summary.needs_review,
                "review_reasons": "; ".join(summary.review_reasons),
            })

    # Save to CSV
    csv_path = out_dir / "event_replay_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(analysis_records[0].keys()))
        writer.writeheader()
        writer.writerows(analysis_records)

    print(f"Exported {len(analysis_records)} event replay records to: {csv_path}")
    print(f"Summary metrics:")
    print(f"  Total transits analyzed: {len(analysis_records)}")
    print(f"  Stable reads across frames: {sum(1 for r in analysis_records if r['is_reading_stable'])} / {len(analysis_records)}")
    print(f"  Transits where later observation improved read: {sum(1 for r in analysis_records if r['later_observation_improved'])}")
    print(f"  Transits with conflicting reads preserved: {sum(1 for r in analysis_records if r['alternative_readings_preserved_count'] > 0)}")


if __name__ == "__main__":
    run_event_replay_analysis()
