#!/usr/bin/env python3
"""CLI utility to process an image or video with the vehicle intelligence pipeline."""

import argparse
import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals for Arabic characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.core.logging import setup_logger
from vehicle_intelligence.pipeline.engine import VehicleIntelligenceEngine
from vehicle_intelligence.pipeline.media_loader import MediaLoader


def main():
    parser = argparse.ArgumentParser(description="Process image or video with vehicle intelligence engine.")
    parser.add_argument("source", type=str, help="Path to input image or video file.")
    parser.add_argument("--max-frames", type=int, default=30, help="Max video frames to process (default: 30).")
    parser.add_argument("--fps", type=float, default=10.0, help="Target processing FPS for video.")
    args = parser.parse_args()

    logger = setup_logger("process_media")
    source_path = Path(args.source)

    if not source_path.exists():
        logger.error(f"Input file not found: {source_path}")
        sys.exit(1)

    engine = VehicleIntelligenceEngine()
    ext = source_path.suffix.lower()

    if ext in MediaLoader.IMAGE_EXTENSIONS:
        logger.info(f"Processing static image: {source_path}")
        image = MediaLoader.load_image(source_path)
        result = engine.process_frame(image, frame_id=0)

        print("\n--- Structured Processing Result ---")
        print(result.model_dump_json(indent=2))

        # Check vehicle track session
        for vehicle in result.vehicle_detections:
            if vehicle.track_id:
                summary = engine.finalize_vehicle_session(vehicle.track_id)
                if summary:
                    print("\n--- Finalized Vehicle Session Summary ---")
                    print(summary.model_dump_json(indent=2))

    elif ext in MediaLoader.VIDEO_EXTENSIONS:
        logger.info(f"Processing video stream: {source_path} (max_frames={args.max_frames}, fps={args.fps})")
        stream = MediaLoader.stream_video(source_path, target_fps=args.fps, max_frames=args.max_frames)

        frame_count = 0
        active_track_ids = set()
        for frame_id, ts, frame in stream:
            res = engine.process_frame(frame, frame_id=frame_id, timestamp=ts)
            frame_count += 1
            for v in res.vehicle_detections:
                if v.track_id:
                    active_track_ids.add(v.track_id)
            logger.info(
                f"Frame {frame_id:04d} @ {ts:.2f}s | "
                f"Vehicles: {len(res.vehicle_detections)} | "
                f"Plates: {len(res.plate_detections)} | "
                f"Latency: {res.latency_ms:.2f}ms"
            )

        print(f"\nCompleted processing {frame_count} frames.")
        for tid in active_track_ids:
            summary = engine.finalize_vehicle_session(tid)
            if summary:
                print(f"\n--- Vehicle Session Summary (Track ID {tid}) ---")
                print(summary.model_dump_json(indent=2))
    else:
        logger.error(f"Unsupported media format extension: {ext}")
        sys.exit(1)


if __name__ == "__main__":
    main()
