"""Mission 009 Integration Tests: End-to-End Gate Pipeline and Event Behavior.

Tests:
1. Configuration propagation through CLI arguments and API singleton.
2. NMS and resolution settings reaching model inference.
3. Vehicle tracking stability across multiple consecutive frames.
4. Multiple plate observations associated with one track.
5. Stronger later observation improving earlier weak read.
6. Conflicting OCR results preserved as alternatives and flagged for review.
7. Missing plate detections handled gracefully without track loss.
8. Low-confidence results correctly flagged.
9. Event creation and evidence persistence.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from vehicle_intelligence.core.domain import (
    ALPRPlateResult,
    BoundingBox,
    PlateDetection,
    PlateTextResult,
    VehicleClass,
    VehicleDetection,
    VehicleSessionSummary,
)
from vehicle_intelligence.core.config import DetectorConfig, OCRConfig, SystemConfig
from vehicle_intelligence.pipeline.alpr_pipeline import EndToEndALPRPipeline
from vehicle_intelligence.tracking.aggregator import MultiFrameEvidenceAggregator
from vehicle_intelligence.tracking.tracker import CentroidIoUTracker
from vehicle_intelligence.api.app import get_pipeline, reset_pipeline


def test_configuration_propagation_cli():
    """Verify CLI argument parsing and parameter propagation to EndToEndALPRPipeline."""
    from scripts.alpr_cli import main
    import argparse

    # Test argument parsing logic
    test_args = [
        "--input", "artifacts/triage/samples/frame_001.jpg",
        "--det-imgsz", "1280",
        "--char-iou", "0.45",
        "--agnostic-nms",
    ]

    with patch("sys.argv", ["alpr_cli.py"] + test_args):
        with patch("scripts.alpr_cli.EndToEndALPRPipeline") as mock_pipeline:
            with patch("scripts.alpr_cli.process_single_image") as mock_process:
                mock_process.return_value = []
                main()
                mock_pipeline.assert_called_once()
                call_kwargs = mock_pipeline.call_args.kwargs
                assert call_kwargs["det_imgsz"] == 1280
                assert call_kwargs["char_iou"] == 0.45
                assert call_kwargs["agnostic_nms"] is True


def test_configuration_propagation_cli_historical_baseline():
    """Verify CLI supports reproducing historical baseline via --no-agnostic-nms."""
    from scripts.alpr_cli import main

    test_args = [
        "--input", "artifacts/triage/samples/frame_001.jpg",
        "--det-imgsz", "640",
        "--char-iou", "0.70",
        "--no-agnostic-nms",
    ]

    with patch("sys.argv", ["alpr_cli.py"] + test_args):
        with patch("scripts.alpr_cli.EndToEndALPRPipeline") as mock_pipeline:
            with patch("scripts.alpr_cli.process_single_image") as mock_process:
                mock_process.return_value = []
                main()
                call_kwargs = mock_pipeline.call_args.kwargs
                assert call_kwargs["det_imgsz"] == 640
                assert call_kwargs["char_iou"] == 0.70
                assert call_kwargs["agnostic_nms"] is False


def test_configuration_propagation_api():
    """Verify API singleton adopts custom configurations and environment variables."""
    reset_pipeline()
    try:
        os.environ["ALPR_DET_IMGSZ"] = "1280"
        os.environ["ALPR_CHAR_IOU"] = "0.45"
        os.environ["ALPR_AGNOSTIC_NMS"] = "true"

        with patch("vehicle_intelligence.pipeline.alpr_pipeline.YOLO"):
            pipe = get_pipeline()
            assert pipe.det_imgsz == 1280
            assert pipe.char_iou == 0.45
            assert pipe.agnostic_nms is True
    finally:
        os.environ.pop("ALPR_DET_IMGSZ", None)
        os.environ.pop("ALPR_CHAR_IOU", None)
        os.environ.pop("ALPR_AGNOSTIC_NMS", None)
        reset_pipeline()


def test_nms_and_resolution_settings_reach_model():
    """Verify NMS and resolution parameters are passed to YOLO predict calls."""
    with patch("vehicle_intelligence.pipeline.alpr_pipeline.YOLO") as mock_yolo:
        mock_det = MagicMock()
        mock_char = MagicMock()
        mock_yolo.side_effect = [mock_det, mock_char]

        mock_det_res = MagicMock()
        mock_det_res.boxes = []
        mock_det.predict.return_value = [mock_det_res]

        pipeline = EndToEndALPRPipeline(
            device="cpu",
            det_imgsz=1280,
            det_iou=0.45,
            det_agnostic_nms=False,
            char_iou=0.45,
            char_imgsz=320,
            agnostic_nms=True,
        )

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        pipeline.read_plate_image(dummy_img)

        # Check detector call
        mock_det.predict.assert_called_once()
        det_call_kwargs = mock_det.predict.call_args.kwargs
        assert det_call_kwargs["imgsz"] == 1280
        assert det_call_kwargs["iou"] == 0.45
        assert det_call_kwargs["agnostic_nms"] is False


def test_vehicle_observed_across_multiple_frames():
    """Verify tracking ID remains stable for a moving vehicle across consecutive frames."""
    tracker = CentroidIoUTracker(max_lost_frames=5, iou_threshold=0.20, max_centroid_distance=150.0)

    # Simulate a truck entering a gate, moving 20 pixels down per frame
    track_ids = []
    for frame_idx in range(1, 6):
        y_offset = (frame_idx - 1) * 20.0
        box = BoundingBox(x1=200.0, y1=100.0 + y_offset, x2=500.0, y2=400.0 + y_offset)
        det = VehicleDetection(box=box, vehicle_class=VehicleClass.TRUCK, confidence=0.88)
        
        tracked = tracker.update([det], frame_id=frame_idx, timestamp=frame_idx * 0.1)
        assert len(tracked) == 1
        track_ids.append(tracked[0].track_id)

    # All frames must have the exact same tracking ID
    assert len(set(track_ids)) == 1
    assert track_ids[0] is not None


def test_multiple_plate_observations_associated_with_one_track():
    """Verify evidence aggregator links multiple observations to a single vehicle track."""
    aggregator = MultiFrameEvidenceAggregator(min_confidence_threshold=0.80)

    for f_idx in range(1, 4):
        v_box = BoundingBox(x1=100.0, y1=100.0 + f_idx * 10, x2=400.0, y2=400.0 + f_idx * 10)
        v_det = VehicleDetection(box=v_box, confidence=0.90, track_id=42)
        p_det = PlateDetection(box=BoundingBox(x1=200.0, y1=300.0, x2=300.0, y2=340.0), confidence=0.85)
        p_read = PlateTextResult(raw_text="أ ب ج ١ ٢ ٣", confidence=0.85)
        
        aggregator.record_observation(
            track_id=42,
            timestamp=f_idx * 0.1,
            vehicle_detection=v_det,
            plate_detection=p_det,
            plate_reading=p_read,
            frame_id=f_idx,
            crop_path=f"crop_{f_idx}.jpg",
        )

    summary = aggregator.finalize_track(42)
    assert summary is not None
    assert summary.track_id == 42
    assert summary.total_frames_tracked == 3
    assert summary.plate_reading.raw_text == "أ ب ج ١ ٢ ٣"
    assert summary.has_conflicting_reads is False


def test_stronger_later_observation_improves_earlier_weak_read():
    """Verify that a stronger later plate observation upgrades an earlier weak read and selects best evidence."""
    aggregator = MultiFrameEvidenceAggregator(min_confidence_threshold=0.82)

    v_det = VehicleDetection(box=BoundingBox(x1=100.0, y1=100.0, x2=400.0, y2=400.0), confidence=0.90, track_id=10)
    p_det = PlateDetection(box=BoundingBox(x1=200.0, y1=300.0, x2=300.0, y2=340.0), confidence=0.80)

    # Frame 1: Weak/distant observation with lower confidence
    ocr_weak = PlateTextResult(raw_text="س م ج ٦٧٣٣", confidence=0.55, needs_review=True)
    aggregator.record_observation(
        track_id=10, timestamp=1.0, vehicle_detection=v_det, plate_detection=p_det, plate_reading=ocr_weak,
        frame_id=1, crop_path="crop_frame_1.jpg"
    )

    # Frame 2: Stronger closer observation with high confidence
    ocr_strong = PlateTextResult(raw_text="س م ج ٦٧٣٣", confidence=0.94, needs_review=False)
    aggregator.record_observation(
        track_id=10, timestamp=1.2, vehicle_detection=v_det, plate_detection=p_det, plate_reading=ocr_strong,
        frame_id=2, crop_path="crop_frame_2.jpg"
    )

    summary = aggregator.finalize_track(10)
    assert summary is not None
    assert summary.plate_reading.raw_text == "س م ج ٦٧٣٣"
    assert summary.plate_reading.confidence == 0.94
    assert summary.best_evidence_frame_id == 2
    assert summary.best_crop_path == "crop_frame_2.jpg"
    assert summary.needs_review is False


def test_conflicting_ocr_results_preserved_and_flagged():
    """Verify that conflicting OCR readings across a transit are preserved as alternatives and flagged for review."""
    aggregator = MultiFrameEvidenceAggregator(min_confidence_threshold=0.80)

    v_det = VehicleDetection(box=BoundingBox(x1=100.0, y1=100.0, x2=400.0, y2=400.0), confidence=0.90, track_id=15)
    p_det = PlateDetection(box=BoundingBox(x1=200.0, y1=300.0, x2=300.0, y2=340.0), confidence=0.85)

    # Frame 1: Initial read with 'س ع د'
    ocr1 = PlateTextResult(raw_text="س ع د 1234", confidence=0.88, needs_review=False)
    aggregator.record_observation(
        track_id=15, timestamp=1.0, vehicle_detection=v_det, plate_detection=p_det, plate_reading=ocr1,
        frame_id=1, crop_path="crop_1.jpg"
    )

    # Frame 2: Glare causes a conflicting read with 'س ع ر'
    ocr2 = PlateTextResult(raw_text="س ع ر 1234", confidence=0.74, needs_review=False)
    aggregator.record_observation(
        track_id=15, timestamp=1.1, vehicle_detection=v_det, plate_detection=p_det, plate_reading=ocr2,
        frame_id=2, crop_path="crop_2.jpg"
    )

    summary = aggregator.finalize_track(15)
    assert summary is not None
    # Primary read is the higher confidence one
    assert summary.plate_reading.raw_text == "س ع د 1234"
    # Must flag conflicting reads and require human review
    assert summary.has_conflicting_reads is True
    assert summary.needs_review is True
    assert any("conflicting" in r.lower() or "ambiguous" in r.lower() for r in summary.review_reasons)
    # The conflicting candidate must be preserved in alternative_readings, not discarded
    assert len(summary.alternative_readings) == 1
    assert summary.alternative_readings[0].raw_text == "س ع ر 1234"


def test_missing_plate_detections_handled_gracefully():
    """Verify that a track survives frames where plate detection temporarily fails."""
    aggregator = MultiFrameEvidenceAggregator(min_confidence_threshold=0.80)

    v_det = VehicleDetection(box=BoundingBox(x1=100.0, y1=100.0, x2=400.0, y2=400.0), confidence=0.90, track_id=20)

    # Frame 1: Plate detected
    p_det = PlateDetection(box=BoundingBox(x1=200.0, y1=300.0, x2=300.0, y2=340.0), confidence=0.85)
    p_read = PlateTextResult(raw_text="ق ط ر ٥٥٥٥", confidence=0.89)
    aggregator.record_observation(track_id=20, timestamp=1.0, vehicle_detection=v_det, plate_detection=p_det, plate_reading=p_read)

    # Frame 2: Temporary occlusion (no plate detected)
    aggregator.record_observation(track_id=20, timestamp=1.1, vehicle_detection=v_det, plate_detection=None, plate_reading=None)

    # Frame 3: Plate detected again
    aggregator.record_observation(track_id=20, timestamp=1.2, vehicle_detection=v_det, plate_detection=p_det, plate_reading=p_read)

    summary = aggregator.finalize_track(20)
    assert summary is not None
    assert summary.total_frames_tracked == 3
    assert summary.plate_reading is not None
    assert summary.plate_reading.raw_text == "ق ط ر ٥٥٥٥"


def test_low_confidence_results_flagged():
    """Verify that sub-threshold plate confidence flags needs_review."""
    aggregator = MultiFrameEvidenceAggregator(min_confidence_threshold=0.82)

    v_det = VehicleDetection(box=BoundingBox(x1=100.0, y1=100.0, x2=400.0, y2=400.0), confidence=0.90, track_id=30)
    p_det = PlateDetection(box=BoundingBox(x1=200.0, y1=300.0, x2=300.0, y2=340.0), confidence=0.60)
    p_read = PlateTextResult(raw_text="د م ل ١٢٣", confidence=0.65)

    aggregator.record_observation(track_id=30, timestamp=1.0, vehicle_detection=v_det, plate_detection=p_det, plate_reading=p_read)
    summary = aggregator.finalize_track(30)

    assert summary is not None
    assert summary.needs_review is True
    assert any("below threshold" in r for r in summary.review_reasons)


def test_event_creation_and_evidence_persistence(tmp_path: Path):
    """Verify that evidence outputs (crops and annotations) are saved with correct links."""
    crops_dir = tmp_path / "crops"
    ann_dir = tmp_path / "annotated"

    with patch("vehicle_intelligence.pipeline.alpr_pipeline.YOLO") as mock_yolo:
        mock_det = MagicMock()
        mock_char = MagicMock()
        mock_yolo.side_effect = [mock_det, mock_char]

        import torch
        mock_box = MagicMock()
        mock_box.xyxy = [torch.tensor([10.0, 10.0, 50.0, 30.0])]
        mock_box.conf = [0.85]
        mock_det_res = MagicMock()
        mock_det_res.boxes = [mock_box]
        mock_det.predict.return_value = [mock_det_res]

        mock_char_res = MagicMock()
        mock_char_res.boxes = []
        mock_char.predict.return_value = [mock_char_res]

        pipeline = EndToEndALPRPipeline(device="cpu")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        results = pipeline.read_plate_image(
            image=dummy_img,
            source_id="frame_test.jpg",
            save_crops_dir=crops_dir,
            save_annotated_dir=ann_dir,
        )

        assert len(results) == 1
        res = results[0]
        assert res.crop_path is not None
        assert Path(res.crop_path).exists()
        assert res.annotated_image_path is not None
        assert Path(res.annotated_image_path).exists()
