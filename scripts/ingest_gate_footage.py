#!/usr/bin/env python3
"""Gate Footage Ingestion Utility (Mission 6 Part E).
Extracts frames at configurable FPS, drops near-duplicates via dHash,
writes frames to data/gate_raw/ (gitignored), and generates YOLO-format pre-labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.data.splitter import compute_dhash, hamming_distance


def ingest_video(
    video_path: Path,
    output_dir: Path,
    sample_fps: float = 2.0,
    dedup_threshold: int = 3,
    detector_weights: Path = Path("artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt"),
    conf_thresh: float = 0.25,
) -> dict:
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading plate detector from {detector_weights} for pre-labeling...")
    detector = YOLO(str(detector_weights))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0 or np.isnan(source_fps):
        source_fps = 25.0

    interval = max(1, int(round(source_fps / sample_fps)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Ingesting '{video_path.name}': {total_frames} frames, source FPS={source_fps:.1f}, sampling every {interval} frames...")

    frame_idx = 0
    saved_frames = 0
    dropped_duplicates = 0
    total_plates_prelabeled = 0
    last_saved_hash = None

    video_prefix = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            h, w = frame.shape[:2]
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_dhash = compute_dhash(pil_img)

            # Check near-duplicate with previous saved frame
            if last_saved_hash is not None and hamming_distance(last_saved_hash, frame_dhash) <= dedup_threshold:
                dropped_duplicates += 1
                frame_idx += 1
                continue

            last_saved_hash = frame_dhash
            saved_frames += 1

            frame_stem = f"{video_prefix}_frame_{saved_frames:05d}"
            img_file = images_dir / f"{frame_stem}.jpg"
            lbl_file = labels_dir / f"{frame_stem}.txt"

            cv2.imwrite(str(img_file), frame)

            # Run detector to generate YOLO pre-labels
            res = detector.predict(source=frame, conf=conf_thresh, imgsz=640, device=0, verbose=False)[0]
            label_lines = []
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    bw = (bx2 - bx1) / w
                    bh = (by2 - by1) / h
                    bxc = (bx1 + bx2) / (2.0 * w)
                    byc = (by1 + by2) / (2.0 * h)
                    # Class 0: license_plate
                    label_lines.append(f"0 {bxc:.6f} {byc:.6f} {bw:.6f} {bh:.6f}")
                    total_plates_prelabeled += 1

            lbl_file.write_text("\n".join(label_lines), encoding="utf-8")

        frame_idx += 1

    cap.release()

    manifest_data = {
        "video_source": str(video_path),
        "source_fps": source_fps,
        "sample_fps": sample_fps,
        "total_source_frames": frame_idx,
        "saved_frames": saved_frames,
        "dropped_duplicates": dropped_duplicates,
        "total_plates_prelabeled": total_plates_prelabeled,
        "images_directory": str(images_dir),
        "labels_directory": str(labels_dir),
    }

    manifest_path = output_dir / f"{video_prefix}_ingest_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    print(f"\nIngestion Complete:")
    print(f"  Saved frames:      {saved_frames}")
    print(f"  Dropped duplicates:{dropped_duplicates}")
    print(f"  Plates pre-labeled:{total_plates_prelabeled}")
    print(f"  Manifest written to: {manifest_path}")

    return manifest_data


def main():
    parser = argparse.ArgumentParser(description="Ingest gate camera footage, deduplicate, and pre-label.")
    parser.add_argument("--video", type=Path, required=True, help="Path to video file (.mp4, .avi, etc.)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/gate_raw"), help="Destination directory (gitignored)")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate")
    parser.add_argument("--dedup-dist", type=int, default=3, help="Hamming distance threshold for near-duplicates")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold")
    args = parser.parse_args()

    ingest_video(
        video_path=args.video,
        output_dir=args.output_dir,
        sample_fps=args.fps,
        dedup_threshold=args.dedup_dist,
        conf_thresh=args.conf,
    )


if __name__ == "__main__":
    main()
