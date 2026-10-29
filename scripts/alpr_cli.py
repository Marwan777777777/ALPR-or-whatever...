#!/usr/bin/env python3
"""ALPR CLI: End-to-end command-line license plate recognition pipeline (Mission 6 Part D).
Supports:
1. Single Image: writes JSON result, annotated image, and plate crop.
2. Image Folder: processes all images, writes crops/JSONs, and outputs a summary CSV.
3. Video / RTSP Stream: samples frames at configurable FPS.
4. Gate Replay Mode: tracks vehicles, selects best frame per transit, emits 1 event per vehicle in JSON Lines.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.core.domain import ALPRPlateResult, VehicleDetection
from vehicle_intelligence.pipeline.alpr_pipeline import EndToEndALPRPipeline
from vehicle_intelligence.tracking.tracker import CentroidIoUTracker


def process_single_image(
    pipeline: EndToEndALPRPipeline,
    img_path: Path,
    output_dir: Path,
) -> List[ALPRPlateResult]:
    im = cv2.imread(str(img_path))
    if im is None:
        raise ValueError(f"Could not load image: {img_path}")

    crops_dir = output_dir / "crops"
    ann_dir = output_dir / "annotated"
    crops_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    results = pipeline.read_plate_image(
        image=im,
        source_id=img_path.name,
        timestamp=time.time(),
        save_crops_dir=crops_dir,
        save_annotated_dir=ann_dir,
    )

    # Save per-image JSON
    json_path = output_dir / f"{img_path.stem}_result.json"
    json_path.write_text(
        json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return results


def process_image_folder(
    pipeline: EndToEndALPRPipeline,
    folder_path: Path,
    output_dir: Path,
) -> List[ALPRPlateResult]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted([f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
    print(f"Found {len(img_files)} images in {folder_path}...")

    all_results: List[ALPRPlateResult] = []
    csv_rows = []

    for idx, f in enumerate(img_files, start=1):
        try:
            res = process_single_image(pipeline, f, output_dir)
            all_results.extend(res)
            if res:
                for r in res:
                    csv_rows.append({
                        "source_id": r.source_id,
                        "plate_text": r.plate_text,
                        "digits": r.digits,
                        "letters": r.letters,
                        "plate_confidence": r.plate_confidence,
                        "detector_confidence": r.detector_confidence,
                        "needs_review": r.needs_review,
                        "review_reasons": "; ".join(r.review_reasons),
                        "crop_path": r.crop_path or "",
                    })
            else:
                csv_rows.append({
                    "source_id": f.name,
                    "plate_text": "NO_PLATE_DETECTED",
                    "digits": "",
                    "letters": "",
                    "plate_confidence": 0.0,
                    "detector_confidence": 0.0,
                    "needs_review": True,
                    "review_reasons": "NO_PLATE_FOUND",
                    "crop_path": "",
                })
        except Exception as e:
            print(f"Error processing {f.name}: {e}")

    # Write summary CSV
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "source_id", "plate_text", "digits", "letters",
            "plate_confidence", "detector_confidence",
            "needs_review", "review_reasons", "crop_path",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Folder processing complete. Summary CSV written to: {csv_path}")
    return all_results


def process_video_stream(
    pipeline: EndToEndALPRPipeline,
    video_source: str,
    output_dir: Path,
    sample_fps: float = 2.0,
    gate_replay: bool = False,
) -> None:
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or np.isnan(video_fps):
        video_fps = 25.0
    frame_interval = max(1, int(round(video_fps / sample_fps)))

    print(f"Streaming video at source FPS={video_fps:.1f}, sampling every {frame_interval} frames (~{sample_fps} FPS)...")

    frame_idx = 0
    sampled_idx = 0
    tracker = CentroidIoUTracker()
    transit_observations = defaultdict(list)

    crops_dir = output_dir / "crops"
    ann_dir = output_dir / "annotated"
    crops_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "transit_events.jsonl" if gate_replay else output_dir / "frame_readings.jsonl"
    jsonl_file = open(jsonl_path, "w", encoding="utf-8")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                sampled_idx += 1
                src_id = f"frame_{sampled_idx:05d}"
                ts = frame_idx / video_fps

                results = pipeline.read_plate_image(
                    image=frame,
                    source_id=src_id,
                    timestamp=ts,
                    save_crops_dir=crops_dir,
                    save_annotated_dir=ann_dir,
                )

                if gate_replay:
                    # Update tracker using plate bounding boxes as proxy for vehicle transit track
                    detections = [
                        VehicleDetection(box=r.bounding_box, confidence=r.detector_confidence)
                        for r in results
                    ]
                    tracked = tracker.update(detections, frame_id=sampled_idx, timestamp=ts)

                    # Correlate track IDs
                    for t_veh in tracked:
                        for r in results:
                            if t_veh.box.iou(r.bounding_box) >= 0.50:
                                r.track_id = t_veh.track_id
                                transit_observations[t_veh.track_id].append(r)
                else:
                    for r in results:
                        jsonl_file.write(r.model_dump_json() + "\n")

            frame_idx += 1

        if gate_replay:
            print(f"Gate Replay: Processing {len(transit_observations)} vehicle transits...")
            for track_id, obs_list in transit_observations.items():
                # Pick the best frame per transit by composite plate confidence
                best_obs = max(obs_list, key=lambda x: x.plate_confidence)
                event_data = {
                    "event_type": "VEHICLE_TRANSIT",
                    "track_id": track_id,
                    "total_observations": len(obs_list),
                    "best_reading": best_obs.model_dump(),
                }
                jsonl_file.write(json.dumps(event_data, ensure_ascii=False) + "\n")

    finally:
        cap.release()
        jsonl_file.close()

    print(f"Video processing finished. Events logged to: {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description="Egyptian ALPR Command Line Interface.")
    parser.add_argument("--input", type=str, required=True, help="Path to single image, image folder, video file, or RTSP URL")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cli_output"), help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate for video streams")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device (cuda:0 or cpu)")
    parser.add_argument("--conf-threshold", type=float, default=0.82, help="Plate confidence threshold for review")
    parser.add_argument("--gate-replay", action="store_true", help="Enable simulated gate transit tracking mode")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = EndToEndALPRPipeline(
        device=args.device,
        confidence_threshold=args.conf_threshold,
    )

    in_path = Path(args.input)
    if in_path.is_file() and in_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        print(f"Processing single image: {in_path}")
        results = process_single_image(pipeline, in_path, args.output_dir)
        print(f"Detected {len(results)} plate(s):")
        for r in results:
            status = "NEEDS_REVIEW" if r.needs_review else "ACCEPTED"
            print(f"  Plate: '{r.plate_text}' [Conf: {r.plate_confidence:.2f}, Status: {status}]")
    elif in_path.is_dir():
        print(f"Processing image folder: {in_path}")
        process_image_folder(pipeline, in_path, args.output_dir)
    elif str(args.input).startswith("rtsp://") or (in_path.is_file() and in_path.suffix.lower() in {".mp4", ".avi", ".mov"}):
        print(f"Processing video stream: {args.input}")
        process_video_stream(pipeline, str(args.input), args.output_dir, sample_fps=args.fps, gate_replay=args.gate_replay)
    else:
        raise ValueError(f"Unsupported input type or path: {args.input}")


if __name__ == "__main__":
    main()
