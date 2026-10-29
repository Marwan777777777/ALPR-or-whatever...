"""Tests for core domain entities, bounding boxes, IoU calculation, and coordinate math."""

import pytest
from vehicle_intelligence.core.domain import (
    BoundingBox,
    PlateDetection,
    PlateTextResult,
    PlateType,
    VehicleClass,
    VehicleDetection,
    VehicleSessionSummary,
)


def test_bounding_box_valid_properties():
    box = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=220.0)
    assert box.width == 100.0
    assert box.height == 200.0
    assert box.area == 20000.0
    assert box.aspect_ratio == 0.5
    assert box.center == (60.0, 120.0)
    assert box.to_xyxy() == (10.0, 20.0, 110.0, 220.0)
    assert box.to_int_tuple() == (10, 20, 110, 220)


def test_bounding_box_invalid_raises_error():
    with pytest.raises(ValueError, match="x2 .* cannot be smaller than x1"):
        BoundingBox(x1=100.0, y1=20.0, x2=50.0, y2=220.0)

    with pytest.raises(ValueError, match="y2 .* cannot be smaller than y1"):
        BoundingBox(x1=10.0, y1=200.0, x2=50.0, y2=100.0)


def test_bounding_box_clamp():
    box = BoundingBox(x1=-10.0, y1=-5.0, x2=700.0, y2=500.0)
    clamped = box.clamp(640, 480)
    assert clamped.x1 == 0.0
    assert clamped.y1 == 0.0
    assert clamped.x2 == 640.0
    assert clamped.y2 == 480.0


def test_bounding_box_iou():
    box1 = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
    box2 = BoundingBox(x1=5.0, y1=0.0, x2=15.0, y2=10.0)
    # Intersection = 5 * 10 = 50
    # Union = 100 + 100 - 50 = 150
    # IoU = 50 / 150 = 1/3
    assert pytest.approx(box1.iou(box2), 0.01) == 0.333

    # Non-overlapping boxes
    box3 = BoundingBox(x1=20.0, y1=20.0, x2=30.0, y2=30.0)
    assert box1.iou(box3) == 0.0


def test_vehicle_detection_schema(sample_box):
    det = VehicleDetection(
        box=sample_box,
        vehicle_class=VehicleClass.SUV,
        confidence=0.89,
        track_id=42,
    )
    assert det.vehicle_class == VehicleClass.SUV
    assert det.confidence == 0.89
    assert det.track_id == 42


def test_plate_text_result_needs_review():
    res = PlateTextResult(
        raw_text="س ع د 123",
        confidence=0.60,
        needs_review=True,
        review_reasons=["Low OCR confidence"]
    )
    assert res.needs_review is True
    assert "Low OCR confidence" in res.review_reasons
