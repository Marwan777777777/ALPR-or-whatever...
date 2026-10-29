"""Tests for multi-frame evidence aggregation and Egyptian plate OCR."""

from vehicle_intelligence.core.domain import (
    BoundingBox,
    PlateDetection,
    PlateTextResult,
    VehicleClass,
    VehicleDetection,
)
from vehicle_intelligence.ocr.mock_ocr import MockPlateOCR
from vehicle_intelligence.tracking.aggregator import MultiFrameEvidenceAggregator


def test_mock_plate_ocr_arabic_and_latin():
    ocr = MockPlateOCR(default_letters="س ع د", default_digits="١ ٢ ٣ ٤", confidence=0.92)
    assert ocr.engine_name == "mock_egyptian_ocr_engine_v1"

    crop = np_zeros = np_zeros = None
    import numpy as np
    crop = np.zeros((50, 150, 3), dtype=np.uint8)

    res = ocr.read(crop)
    assert res.arabic_letters == "س ع د"
    assert res.latin_transcription == "S E D 1234"
    assert res.confidence == 0.92
    assert res.needs_review is False


def test_evidence_aggregator_reconciles_and_flags_review():
    aggregator = MultiFrameEvidenceAggregator(min_confidence_threshold=0.80)

    box = BoundingBox(x1=100.0, y1=100.0, x2=300.0, y2=300.0)
    vehicle = VehicleDetection(box=box, vehicle_class=VehicleClass.CAR, confidence=0.90, track_id=7)
    plate = PlateDetection(box=BoundingBox(x1=150.0, y1=240.0, x2=250.0, y2=280.0), confidence=0.85)

    # Frame 1: Low confidence OCR reading
    ocr_f1 = PlateTextResult(raw_text="س ع د 1234", confidence=0.60, needs_review=True)
    aggregator.record_observation(track_id=7, timestamp=1.0, vehicle_detection=vehicle, plate_detection=plate, plate_reading=ocr_f1)

    # Frame 2: Higher confidence OCR reading
    ocr_f2 = PlateTextResult(raw_text="س ع د 1234", confidence=0.88, needs_review=False)
    aggregator.record_observation(track_id=7, timestamp=1.1, vehicle_detection=vehicle, plate_detection=plate, plate_reading=ocr_f2)

    summary = aggregator.finalize_track(track_id=7)
    assert summary is not None
    assert summary.track_id == 7
    assert summary.total_frames_tracked == 2
    assert summary.plate_reading is not None
    assert summary.plate_reading.confidence == 0.88
    # Should accept the high confidence reading without needing review
    assert summary.needs_review is False
