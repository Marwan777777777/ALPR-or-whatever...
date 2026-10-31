#!/usr/bin/env python3
"""Gate Replay Benchmark and Pipeline Integration Validation (Mission 009).

Measures the complete processing path across:
1. Frame loading and decoding
2. Plate detection
3. Plate crop extraction
4. Character detection and OCR
5. Tracking and plate-to-vehicle association
6. Event creation and evidence output

Compares:
- Configuration A: 640 plate detector, character NMS IoU 0.70
- Configuration B: 640 plate detector, character NMS IoU 0.45
- Configuration C: 1280 plate detector, character NMS IoU 0.45
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    VehicleClass,
    VehicleDetection,
    VehicleSessionSummary,
)
from vehicle_intelligence.pipeline.alpr_pipeline import EndToEndALPRPipeline
from vehicle_intelligence.tracking.aggregator import MultiFrameEvidenceAggregator
from vehicle_intelligence.tracking.tracker import CentroidIoUTracker

sys.stdout.reconfigure(encoding="utf-8")


def run_single_configuration_benchmark(
    config_name: str,
    pipeline: EndToEndALPRPipeline,
    frame_paths: List[Path],
    output_dir: Path,
) -> Dict[str, Any]:
    """Runs a full end-to-end replay benchmark across the specified frame sequence."""
    tracker = CentroidIoUTracker(max_lost_frames=10, iou_threshold=0.20)
    aggregator = MultiFrameEvidenceAggregator(min_confidence_threshold=0.82)

    crops_dir = output_dir / "crops"
    ann_dir = output_dir / "annotated"
    crops_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    per_frame_latencies = []
    load_latencies = []
    det_latencies = []
    crop_latencies = []
    ocr_latencies = []
    track_latencies = []
    event_latencies = []

    total_plate_detections = 0
    total_ocr_results = 0
    total_low_confidence = 0

    transit_observations = defaultdict(list)
    t_start_total = time.perf_counter()

    for idx, f_path in enumerate(frame_paths, start=1):
        t0 = time.perf_counter()

        # Stage 1: Frame loading and decoding
        t_l0 = time.perf_counter()
        img = cv2.imread(str(f_path))
        if img is None:
            continue
        t_l1 = time.perf_counter()
        load_latencies.append((t_l1 - t_l0) * 1000.0)

        h, w = img.shape[:2]
        ts = idx * 0.1  # Simulated 10 FPS timestamp

        # Stage 2: Plate detection
        t_d0 = time.perf_counter()
        det_res = pipeline.detector.predict(
            source=img,
            conf=pipeline.det_conf,
            iou=pipeline.det_iou,
            imgsz=pipeline.det_imgsz,
            agnostic_nms=pipeline.det_agnostic_nms,
            device=pipeline.device,
            verbose=False,
        )[0]
        t_d1 = time.perf_counter()
        det_latencies.append((t_d1 - t_d0) * 1000.0)

        frame_results: List[ALPRPlateResult] = []

        # Stage 3: Crop extraction & Stage 4: Character OCR
        t_c_total = 0.0
        t_ocr_total = 0.0

        if det_res.boxes is not None and len(det_res.boxes) > 0:
            total_plate_detections += len(det_res.boxes)

            for b_idx, b in enumerate(det_res.boxes):
                bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
                det_score = float(b.conf[0])

                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w, bx2), min(h, by2)
                if bx2 <= bx1 or by2 <= by1:
                    continue

                # Stage 3: Plate crop extraction
                t_cr0 = time.perf_counter()
                plate_crop = img[by1:by2, bx1:bx2]
                plate_box = BoundingBox(x1=bx1, y1=by1, x2=bx2, y2=by2)
                t_cr1 = time.perf_counter()
                t_c_total += (t_cr1 - t_cr0) * 1000.0

                # Stage 4: Character detection and OCR
                t_o0 = time.perf_counter()
                c_res = pipeline.char_detector.predict(
                    source=plate_crop,
                    conf=pipeline.char_conf,
                    iou=pipeline.char_iou,
                    agnostic_nms=pipeline.agnostic_nms,
                    imgsz=pipeline.char_imgsz,
                    device=pipeline.device,
                    verbose=False,
                )[0]

                from vehicle_intelligence.ocr.reading_order import CharacterBox, EgyptianPlateReadingOrder
                char_boxes = []
                char_confs = []
                if c_res.boxes is not None and len(c_res.boxes) > 0:
                    for cb in c_res.boxes:
                        c_conf = float(cb.conf[0])
                        if c_conf >= pipeline.char_conf:
                            cls_id = int(cb.cls[0])
                            xywhn = cb.xywhn[0].tolist()
                            char_boxes.append(
                                CharacterBox(
                                    class_id=cls_id,
                                    x_center=xywhn[0],
                                    y_center=xywhn[1],
                                    width=xywhn[2],
                                    height=xywhn[3],
                                    confidence=c_conf,
                                )
                            )
                            char_confs.append(round(c_conf, 4))

                plate_str, letters_str, digits_str = EgyptianPlateReadingOrder.reconstruct_plate(char_boxes)
                plate_conf = pipeline.compute_plate_confidence(det_score, char_confs)
                t_o1 = time.perf_counter()
                t_ocr_total += (t_o1 - t_o0) * 1000.0

                total_ocr_results += 1
                needs_review = (plate_conf < pipeline.confidence_threshold or det_score < 0.50 or len(char_boxes) < 2)
                if needs_review:
                    total_low_confidence += 1

                # Save sample crop
                c_file = crops_dir / f"{f_path.stem}_plate_{b_idx+1}.jpg"
                cv2.imwrite(str(c_file), plate_crop)

                res_item = ALPRPlateResult(
                    source_id=f_path.name,
                    timestamp=ts,
                    plate_text=plate_str,
                    letters=letters_str,
                    digits=digits_str,
                    per_character_confidences=char_confs,
                    plate_confidence=plate_conf,
                    detector_confidence=round(det_score, 4),
                    needs_review=needs_review,
                    review_reasons=["LOW_CONF"] if needs_review else [],
                    bounding_box=plate_box,
                    crop_path=str(c_file),
                )
                frame_results.append(res_item)

        crop_latencies.append(t_c_total)
        ocr_latencies.append(t_ocr_total)

        # Stage 5: Tracking and plate-to-vehicle association
        t_tr0 = time.perf_counter()
        detections = [VehicleDetection(box=r.bounding_box, confidence=r.detector_confidence) for r in frame_results]
        tracked = tracker.update(detections, frame_id=idx, timestamp=ts)

        for t_veh in tracked:
            for r in frame_results:
                if t_veh.box.iou(r.bounding_box) >= 0.20:
                    r.track_id = t_veh.track_id
                    transit_observations[t_veh.track_id].append(r)

                    # Also update evidence aggregator
                    p_det = PlateDetection(box=r.bounding_box, confidence=r.detector_confidence)
                    p_read = PlateTextResult(raw_text=r.plate_text, confidence=r.plate_confidence, needs_review=r.needs_review)
                    aggregator.record_observation(
                        track_id=t_veh.track_id,
                        timestamp=ts,
                        vehicle_detection=t_veh,
                        plate_detection=p_det,
                        plate_reading=p_read,
                        frame_id=idx,
                        crop_path=r.crop_path,
                    )
        t_tr1 = time.perf_counter()
        track_latencies.append((t_tr1 - t_tr0) * 1000.0)

        # Stage 6: Per-frame evidence output latency
        t_ev0 = time.perf_counter()
        # Save annotated image for verification
        ann_file = ann_dir / f"{f_path.stem}_annotated.jpg"
        for r in frame_results:
            cv2.rectangle(img, (int(r.bounding_box.x1), int(r.bounding_box.y1)), (int(r.bounding_box.x2), int(r.bounding_box.y2)), (0, 255, 0), 2)
        cv2.imwrite(str(ann_file), img)
        t_ev1 = time.perf_counter()
        event_latencies.append((t_ev1 - t_ev0) * 1000.0)

        t_end_frame = time.perf_counter()
        per_frame_latencies.append((t_end_frame - t0) * 1000.0)

    total_time_seconds = time.perf_counter() - t_start_total

    # Finalize all vehicle transits
    emitted_transit_events = 0
    duplicate_or_fragmented_events = 0
    finalized_summaries = []

    for track_id in list(transit_observations.keys()):
        summary = aggregator.finalize_track(track_id)
        if summary is not None:
            emitted_transit_events += 1
            finalized_summaries.append(summary)

    # Check for duplicate/fragmented events (same plate string emitted under different track IDs within short duration)
    plate_tracks = defaultdict(list)
    for s in finalized_summaries:
        if s.plate_reading and s.plate_reading.raw_text:
            plate_tracks[s.plate_reading.raw_text].append(s)

    for plate_txt, track_list in plate_tracks.items():
        if len(track_list) > 1:
            duplicate_or_fragmented_events += (len(track_list) - 1)

    peak_gpu_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_gpu_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

    fps = round(len(frame_paths) / max(0.001, total_time_seconds), 2)

    return {
        "config_name": config_name,
        "detector_resolution": pipeline.det_imgsz,
        "detector_iou": pipeline.det_iou,
        "detector_agnostic_nms": pipeline.det_agnostic_nms,
        "character_nms_iou": pipeline.char_iou,
        "character_agnostic_nms": pipeline.agnostic_nms,
        "total_frames_processed": len(frame_paths),
        "total_processing_time_seconds": round(total_time_seconds, 3),
        "fps": fps,
        "latency_metrics_ms": {
            "mean_frame_latency": round(float(np.mean(per_frame_latencies)), 2),
            "median_frame_latency": round(float(np.median(per_frame_latencies)), 2),
            "p95_frame_latency": round(float(np.percentile(per_frame_latencies, 95)), 2),
            "min_frame_latency": round(float(np.min(per_frame_latencies)), 2),
            "max_frame_latency": round(float(np.max(per_frame_latencies)), 2),
        },
        "component_breakdown_ms": {
            "load_and_decode": round(float(np.mean(load_latencies)), 2),
            "plate_detection_model": round(float(np.mean(det_latencies)), 2),
            "crop_extraction": round(float(np.mean(crop_latencies)), 2),
            "character_ocr_model": round(float(np.mean(ocr_latencies)), 2),
            "tracking_and_association": round(float(np.mean(track_latencies)), 2),
            "evidence_output": round(float(np.mean(event_latencies)), 2),
            "total_model_latency": round(float(np.mean(det_latencies) + np.mean(ocr_latencies)), 2),
            "total_non_model_latency": round(float(np.mean(load_latencies) + np.mean(crop_latencies) + np.mean(track_latencies) + np.mean(event_latencies)), 2),
        },
        "gpu_memory_usage_mb": peak_gpu_vram_mb,
        "counts": {
            "plate_detections": total_plate_detections,
            "ocr_results": total_ocr_results,
            "emitted_transit_events": emitted_transit_events,
            "duplicate_or_fragmented_events": duplicate_or_fragmented_events,
            "low_confidence_results": total_low_confidence,
        },
        "finalized_summaries": [
            {
                "track_id": s.track_id,
                "first_seen": round(s.first_seen_timestamp, 2),
                "last_seen": round(s.last_seen_timestamp, 2),
                "total_frames": s.total_frames_tracked,
                "plate_text": s.plate_reading.raw_text if s.plate_reading else "",
                "plate_confidence": round(s.plate_reading.confidence, 4) if s.plate_reading else 0.0,
                "has_conflicting_reads": s.has_conflicting_reads,
                "alternative_readings_count": len(s.alternative_readings),
                "best_evidence_frame_id": s.best_evidence_frame_id,
                "best_crop_path": s.best_crop_path,
                "needs_review": s.needs_review,
                "review_reasons": s.review_reasons,
            }
            for s in finalized_summaries
        ],
    }


def main():
    print("=== STARTING GATE REPLAY BENCHMARK (MISSION 009) ===")
    out_dir = root_dir / "artifacts" / "missions" / "mission_009"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(list((root_dir / "artifacts" / "triage" / "samples").glob("frame_*.jpg")))
    print(f"Loaded {len(frames)} gate triage frames for benchmark.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Executing benchmark on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Configuration definitions
    configs = [
        {
            "name": "Configuration A (640 / char_iou 0.70)",
            "det_imgsz": 640,
            "char_iou": 0.70,
            "agnostic_nms": True,
        },
        {
            "name": "Configuration B (640 / char_iou 0.45)",
            "det_imgsz": 640,
            "char_iou": 0.45,
            "agnostic_nms": True,
        },
        {
            "name": "Configuration C (1280 / char_iou 0.45)",
            "det_imgsz": 1280,
            "char_iou": 0.45,
            "agnostic_nms": True,
        },
    ]

    all_benchmarks = []
    per_config_csv_rows = []

    for cfg in configs:
        cfg_name = cfg["name"]
        print(f"\n--- Running {cfg_name} ---")
        cfg_out_dir = out_dir / "benchmark_runs" / cfg_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        pipeline = EndToEndALPRPipeline(
            device=device,
            det_imgsz=cfg["det_imgsz"],
            char_iou=cfg["char_iou"],
            agnostic_nms=cfg["agnostic_nms"],
        )

        res = run_single_configuration_benchmark(
            config_name=cfg_name,
            pipeline=pipeline,
            frame_paths=frames,
            output_dir=cfg_out_dir,
        )
        all_benchmarks.append(res)

        print(f"  Processed {res['total_frames_processed']} frames in {res['total_processing_time_seconds']}s -> {res['fps']} FPS")
        print(f"  Latency: mean={res['latency_metrics_ms']['mean_frame_latency']}ms, median={res['latency_metrics_ms']['median_frame_latency']}ms, p95={res['latency_metrics_ms']['p95_frame_latency']}ms")
        print(f"  Component breakdown: load={res['component_breakdown_ms']['load_and_decode']}ms, det={res['component_breakdown_ms']['plate_detection_model']}ms, ocr={res['component_breakdown_ms']['character_ocr_model']}ms, track={res['component_breakdown_ms']['tracking_and_association']}ms, event={res['component_breakdown_ms']['evidence_output']}ms")
        print(f"  Model latency={res['component_breakdown_ms']['total_model_latency']}ms | Non-model latency={res['component_breakdown_ms']['total_non_model_latency']}ms")
        print(f"  Counts: plates={res['counts']['plate_detections']}, OCR={res['counts']['ocr_results']}, events={res['counts']['emitted_transit_events']}, duplicates={res['counts']['duplicate_or_fragmented_events']}, low_conf={res['counts']['low_confidence_results']}")
        print(f"  Peak GPU VRAM: {res['gpu_memory_usage_mb']} MB")

        per_config_csv_rows.append({
            "configuration": cfg_name,
            "det_resolution": cfg["det_imgsz"],
            "char_nms_iou": cfg["char_iou"],
            "total_frames": res["total_frames_processed"],
            "total_time_sec": res["total_processing_time_seconds"],
            "fps": res["fps"],
            "mean_latency_ms": res["latency_metrics_ms"]["mean_frame_latency"],
            "median_latency_ms": res["latency_metrics_ms"]["median_frame_latency"],
            "p95_latency_ms": res["latency_metrics_ms"]["p95_frame_latency"],
            "model_latency_ms": res["component_breakdown_ms"]["total_model_latency"],
            "non_model_latency_ms": res["component_breakdown_ms"]["total_non_model_latency"],
            "plate_detections": res["counts"]["plate_detections"],
            "ocr_results": res["counts"]["ocr_results"],
            "transit_events": res["counts"]["emitted_transit_events"],
            "duplicate_or_fragmented_events": res["counts"]["duplicate_or_fragmented_events"],
            "low_confidence_count": res["counts"]["low_confidence_results"],
            "peak_vram_mb": res["gpu_memory_usage_mb"],
        })

    # Save benchmark JSON
    benchmark_json_path = out_dir / "end_to_end_benchmark.json"
    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mission": "MISSION 009",
                "title": "End-to-End Gate Pipeline Integration & Replay Benchmark",
                "date": "2026-10-30",
                "device": device,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "model_checkpoints": {
                    "plate_detector": "artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt",
                    "character_recognizer": "artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt",
                },
                "configurations": all_benchmarks,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved end-to-end benchmark JSON to: {benchmark_json_path}")

    # Save per-configuration results CSV
    per_config_csv_path = out_dir / "per_configuration_results.csv"
    with open(per_config_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_config_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_config_csv_rows)
    print(f"Saved per-configuration results CSV to: {per_config_csv_path}")


if __name__ == "__main__":
    main()
