"""Tests for multi-object tracking logic."""

from vehicle_intelligence.core.domain import BoundingBox, VehicleClass, VehicleDetection
from vehicle_intelligence.tracking.tracker import CentroidIoUTracker


def test_tracker_assigns_and_maintains_track_id():
    tracker = CentroidIoUTracker(max_lost_frames=5, iou_threshold=0.20)

    # Frame 1: Vehicle appears at x=100
    box_f1 = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=200.0)
    det1 = VehicleDetection(box=box_f1, vehicle_class=VehicleClass.CAR, confidence=0.9)
    res_f1 = tracker.update([det1], frame_id=1, timestamp=0.0)

    assert len(res_f1) == 1
    track_id_1 = res_f1[0].track_id
    assert track_id_1 is not None

    # Frame 2: Vehicle moved slightly to x=105
    box_f2 = BoundingBox(x1=105.0, y1=102.0, x2=205.0, y2=202.0)
    det2 = VehicleDetection(box=box_f2, vehicle_class=VehicleClass.CAR, confidence=0.92)
    res_f2 = tracker.update([det2], frame_id=2, timestamp=0.1)

    assert len(res_f2) == 1
    assert res_f2[0].track_id == track_id_1, "Track ID must remain persistent across consecutive frames"


def test_tracker_expires_lost_tracks():
    tracker = CentroidIoUTracker(max_lost_frames=2)

    box = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=200.0)
    det = VehicleDetection(box=box, vehicle_class=VehicleClass.CAR, confidence=0.9)
    _ = tracker.update([det], frame_id=1, timestamp=0.0)

    # Update with empty detections for 3 frames
    tracker.update([], frame_id=2, timestamp=0.1)
    tracker.update([], frame_id=3, timestamp=0.2)
    tracker.update([], frame_id=4, timestamp=0.3)

    assert len(tracker._tracks) == 0, "Track should expire after max_lost_frames"
