"""Deterministic synthetic integration tests for Mission 010:
Vehicle-Level Detection, Tracking, and Plate Association.

Covers all 8 required scenarios:
1. One vehicle with one plate across multiple frames
2. Two vehicles in the same frame
3. Two vehicles passing close together
4. One vehicle with intermittent plate detection
5. A plate detected without a vehicle detection
6. An ambiguous plate-to-vehicle association
7. Conflicting OCR observations for one vehicle
8. Temporary vehicle detection loss and track recovery
"""

import pytest
from vehicle_intelligence.core.domain import (
    AssociationStatus,
    BoundingBox,
    PlateDetection,
    PlateTextResult,
    VehicleClass,
    VehicleDetection,
    VehicleTransitEvent,
)
from vehicle_intelligence.tracking.tracker import CentroidIoUTracker
from vehicle_intelligence.association.plate_vehicle_associator import PlateVehicleAssociator
from vehicle_intelligence.tracking.aggregator import MultiFrameEvidenceAggregator


def test_scenario_1_single_vehicle_single_plate_multiple_frames():
    """Scenario 1: One vehicle with one plate tracked across multiple consecutive frames."""
    tracker = CentroidIoUTracker(max_lost_frames=5)
    associator = PlateVehicleAssociator()
    aggregator = MultiFrameEvidenceAggregator()

    initial_track_id = None
    frames_count = 5

    for frame_idx in range(1, frames_count + 1):
        ts = frame_idx * 0.1
        # Vehicle moves linearly forward: x in [100..400], y advances slightly
        vx1 = 100.0 + frame_idx * 5.0
        vy1 = 200.0 + frame_idx * 10.0
        vx2 = vx1 + 250.0
        vy2 = vy1 + 350.0
        v_box = BoundingBox(x1=vx1, y1=vy1, x2=vx2, y2=vy2)
        v_det = VehicleDetection(box=v_box, vehicle_class=VehicleClass.TRUCK, confidence=0.92)

        # Plate located inside lower bumper region of vehicle
        px1 = vx1 + 75.0
        py1 = vy2 - 70.0
        px2 = px1 + 100.0
        py2 = py1 + 45.0
        p_box = BoundingBox(x1=px1, y1=py1, x2=px2, y2=py2)
        p_det = PlateDetection(box=p_box, confidence=0.90)

        # 1. Update vehicle tracker
        tracked_vehicles = tracker.update([v_det], frame_id=frame_idx, timestamp=ts)
        assert len(tracked_vehicles) == 1
        current_tid = tracked_vehicles[0].track_id

        if initial_track_id is None:
            initial_track_id = current_tid
        else:
            assert current_tid == initial_track_id, "Vehicle track ID must persist across frames"

        # 2. Plate-to-vehicle association
        associations = associator.associate([p_det], tracked_vehicles, frame_id=frame_idx, timestamp=ts)
        assert len(associations) == 1
        assoc = associations[0]
        assert assoc.status == AssociationStatus.CONFIDENT
        assert assoc.vehicle_track_id == initial_track_id
        assert assoc.containment_ratio >= 0.99

        # 3. Aggregate evidence
        plate_ocr = PlateTextResult(
            raw_text="س ق ج 8568",
            arabic_letters="س ق ج",
            arabic_digits="٨٥٦٨",
            confidence=0.91,
            needs_review=False
        )
        aggregator.record_observation(
            track_id=initial_track_id,
            timestamp=ts,
            vehicle_detection=tracked_vehicles[0],
            plate_detection=p_det,
            plate_reading=plate_ocr,
            plate_association=assoc,
            frame_id=frame_idx,
            source_id=f"frame_{frame_idx}.jpg",
        )

    # 4. Finalize event
    event: VehicleTransitEvent = aggregator.finalize_vehicle_transit_event(initial_track_id)
    assert event is not None
    assert event.vehicle_track_id == initial_track_id
    assert event.vehicle_class == VehicleClass.TRUCK
    assert event.total_frames_tracked == 5
    assert event.plate_text == "س ق ج 8568"
    assert event.plate_confidence == 0.91
    assert event.association_status == AssociationStatus.CONFIDENT
    assert event.has_conflicting_reads is False
    assert event.has_conflicting_association is False
    assert event.needs_review is False
    assert len(event.observation_timestamps) == 5


def test_scenario_2_two_vehicles_in_same_frame():
    """Scenario 2: Two vehicles simultaneously present in the same frame."""
    tracker = CentroidIoUTracker()
    associator = PlateVehicleAssociator()
    aggregator = MultiFrameEvidenceAggregator()

    # Vehicle A in lane 1 (left side)
    box_a = BoundingBox(x1=50.0, y1=200.0, x2=250.0, y2=500.0)
    det_a = VehicleDetection(box=box_a, vehicle_class=VehicleClass.CAR, confidence=0.90)
    plate_a = PlateDetection(box=BoundingBox(x1=100.0, y1=420.0, x2=200.0, y2=460.0), confidence=0.88)

    # Vehicle B in lane 2 (right side)
    box_b = BoundingBox(x1=500.0, y1=220.0, x2=750.0, y2=520.0)
    det_b = VehicleDetection(box=box_b, vehicle_class=VehicleClass.TRUCK, confidence=0.85)
    plate_b = PlateDetection(box=BoundingBox(x1=560.0, y1=440.0, x2=680.0, y2=485.0), confidence=0.92)

    # Update tracker with both vehicles
    tracked = tracker.update([det_a, det_b], frame_id=1, timestamp=1.0)
    assert len(tracked) == 2
    track_ids = {t.track_id for t in tracked}
    assert len(track_ids) == 2, "Vehicles must receive independent track IDs"

    # Associate both plates
    associations = associator.associate([plate_a, plate_b], tracked, frame_id=1, timestamp=1.0)
    assert len(associations) == 2

    # Map associations by plate position
    assoc_left = next(a for a in associations if a.plate_box.x1 < 300)
    assoc_right = next(a for a in associations if a.plate_box.x1 > 400)

    t_a = next(t for t in tracked if t.vehicle_class == VehicleClass.CAR)
    t_b = next(t for t in tracked if t.vehicle_class == VehicleClass.TRUCK)

    assert assoc_left.vehicle_track_id == t_a.track_id
    assert assoc_left.status == AssociationStatus.CONFIDENT
    assert assoc_right.vehicle_track_id == t_b.track_id
    assert assoc_right.status == AssociationStatus.CONFIDENT


def test_scenario_3_two_vehicles_passing_close_together():
    """Scenario 3: Two vehicles passing close together without track hopping or ID swaps."""
    tracker = CentroidIoUTracker(max_lost_frames=5, iou_threshold=0.20)

    # Vehicle 1 (Truck in lane 1) moving south: x in [100..300]
    # Vehicle 2 (Car in lane 2) moving north: x in [280..420] (adjacent/close)
    track_id_v1 = None
    track_id_v2 = None

    for frame in range(1, 4):
        y_v1 = 100.0 + frame * 40.0
        y_v2 = 400.0 - frame * 40.0

        v1 = VehicleDetection(
            box=BoundingBox(x1=100.0, y1=y_v1, x2=280.0, y2=y_v1 + 200.0),
            vehicle_class=VehicleClass.TRUCK,
            confidence=0.92
        )
        v2 = VehicleDetection(
            box=BoundingBox(x1=270.0, y1=y_v2, x2=410.0, y2=y_v2 + 160.0),
            vehicle_class=VehicleClass.CAR,
            confidence=0.88
        )

        tracked = tracker.update([v1, v2], frame_id=frame, timestamp=frame * 0.1)
        assert len(tracked) == 2

        t1 = next(t for t in tracked if t.vehicle_class == VehicleClass.TRUCK)
        t2 = next(t for t in tracked if t.vehicle_class == VehicleClass.CAR)

        if frame == 1:
            track_id_v1 = t1.track_id
            track_id_v2 = t2.track_id
            assert track_id_v1 != track_id_v2
        else:
            assert t1.track_id == track_id_v1, "Truck track ID swapped!"
            assert t2.track_id == track_id_v2, "Car track ID swapped!"


def test_scenario_4_intermittent_plate_detection():
    """Scenario 4: One vehicle tracked continuously with intermittent plate detections."""
    tracker = CentroidIoUTracker(max_lost_frames=5)
    associator = PlateVehicleAssociator()
    aggregator = MultiFrameEvidenceAggregator()

    v_track_id = None

    # Track across 6 frames; plate visible only in frames 1, 2, and 5
    for frame in range(1, 7):
        ts = frame * 0.1
        v_box = BoundingBox(x1=150.0, y1=100.0 + frame * 15.0, x2=400.0, y2=450.0 + frame * 15.0)
        v_det = VehicleDetection(box=v_box, vehicle_class=VehicleClass.TRUCK, confidence=0.89)

        tracked = tracker.update([v_det], frame_id=frame, timestamp=ts)
        assert len(tracked) == 1
        t_veh = tracked[0]
        if v_track_id is None:
            v_track_id = t_veh.track_id
        else:
            assert t_veh.track_id == v_track_id, "Vehicle track must not fragment"

        # Plate detection is intermittent
        plate_dets = []
        plate_reading = None
        assoc = None
        if frame in [1, 2, 5]:
            p_box = BoundingBox(x1=220.0, y1=v_box.y2 - 60.0, x2=320.0, y2=v_box.y2 - 20.0)
            p_det = PlateDetection(box=p_box, confidence=0.88)
            plate_dets.append(p_det)
            assocs = associator.associate(plate_dets, tracked, frame_id=frame, timestamp=ts)
            assoc = assocs[0]
            plate_reading = PlateTextResult(raw_text="أ ب ج 1234", confidence=0.90, needs_review=False)

        aggregator.record_observation(
            track_id=v_track_id,
            timestamp=ts,
            vehicle_detection=t_veh,
            plate_detection=plate_dets[0] if plate_dets else None,
            plate_reading=plate_reading,
            plate_association=assoc,
            frame_id=frame,
            source_id=f"frame_{frame}.jpg",
        )

    event = aggregator.finalize_vehicle_transit_event(v_track_id)
    assert event is not None
    assert event.total_frames_tracked == 6
    assert event.plate_text == "أ ب ج 1234"
    assert event.plate_confidence == 0.90
    assert event.association_status == AssociationStatus.CONFIDENT
    # Successfully consolidated despite 3 frames without plate detection
    assert event.needs_review is False


def test_scenario_5_plate_detected_without_vehicle_detection():
    """Scenario 5: Plate detected without any enclosing vehicle detection (e.g. night IR domain shift)."""
    associator = PlateVehicleAssociator()

    # Plate detected near center of image
    plate = PlateDetection(box=BoundingBox(x1=500.0, y1=600.0, x2=650.0, y2=660.0), confidence=0.89)

    # 0 vehicles detected in frame
    vehicles = []

    associations = associator.associate([plate], vehicles, frame_id=1, timestamp=0.0)
    assert len(associations) == 1
    assoc = associations[0]

    assert assoc.status == AssociationStatus.UNASSIGNED
    assert assoc.vehicle_track_id is None
    assert assoc.needs_review is True
    assert any("UNASSIGNED_PLATE" in reason for reason in assoc.review_reasons)


def test_scenario_6_ambiguous_plate_vehicle_association():
    """Scenario 6: Ambiguous plate-to-vehicle association where multiple vehicles overlap plate."""
    associator = PlateVehicleAssociator(ambiguity_margin=0.15)

    # Plate at [200, 300, 320, 350]
    plate = PlateDetection(box=BoundingBox(x1=200.0, y1=300.0, x2=320.0, y2=350.0), confidence=0.88)

    # Two overlapping vehicles (e.g. car towing trailer or tight tailgating/queue)
    v1 = VehicleDetection(
        box=BoundingBox(x1=150.0, y1=100.0, x2=350.0, y2=400.0),
        vehicle_class=VehicleClass.CAR,
        confidence=0.85,
        track_id=10
    )
    v2 = VehicleDetection(
        box=BoundingBox(x1=160.0, y1=120.0, x2=360.0, y2=420.0),
        vehicle_class=VehicleClass.VAN,
        confidence=0.84,
        track_id=11
    )

    associations = associator.associate([plate], [v1, v2], frame_id=1, timestamp=1.0)
    assert len(associations) == 1
    assoc = associations[0]

    assert assoc.status == AssociationStatus.AMBIGUOUS
    assert assoc.needs_review is True
    assert 10 in assoc.competing_vehicle_ids
    assert 11 in assoc.competing_vehicle_ids
    assert any("AMBIGUOUS_ASSOCIATION" in r for r in assoc.review_reasons)


def test_scenario_7_conflicting_ocr_observations_for_one_vehicle():
    """Scenario 7: Conflicting OCR observations for a single vehicle across transit."""
    aggregator = MultiFrameEvidenceAggregator()

    v_box = BoundingBox(x1=100.0, y1=100.0, x2=400.0, y2=450.0)
    v_det = VehicleDetection(box=v_box, vehicle_class=VehicleClass.CAR, confidence=0.90, track_id=42)
    p_det = PlateDetection(box=BoundingBox(x1=200.0, y1=380.0, x2=300.0, y2=420.0), confidence=0.85)

    # Frame 1: Reads "س ع د 1234" (conf=0.72)
    ocr_1 = PlateTextResult(raw_text="س ع د 1234", arabic_letters="س ع د", arabic_digits="١٢٣٤", confidence=0.72)
    aggregator.record_observation(track_id=42, timestamp=1.0, vehicle_detection=v_det, plate_detection=p_det, plate_reading=ocr_1, frame_id=1)

    # Frame 2: Reads "س ع ر 1234" with higher confidence (0.91)
    ocr_2 = PlateTextResult(raw_text="س ع ر 1234", arabic_letters="س ع ر", arabic_digits="١٢٣٤", confidence=0.91)
    aggregator.record_observation(track_id=42, timestamp=1.1, vehicle_detection=v_det, plate_detection=p_det, plate_reading=ocr_2, frame_id=2)

    event = aggregator.finalize_vehicle_transit_event(42)
    assert event is not None
    assert event.plate_text == "س ع ر 1234"
    assert event.has_conflicting_reads is True
    assert event.needs_review is True
    assert len(event.alternative_readings) >= 1
    assert event.alternative_readings[0].raw_text == "س ع د 1234"
    assert any("Conflicting plate reads" in r for r in event.review_reasons)


def test_scenario_8_temporary_vehicle_detection_loss_and_recovery():
    """Scenario 8: Temporary vehicle detection loss and track recovery within max_lost_frames."""
    tracker = CentroidIoUTracker(max_lost_frames=3, iou_threshold=0.20)

    # Frame 1: Initial detection
    v_f1 = VehicleDetection(
        box=BoundingBox(x1=100.0, y1=100.0, x2=300.0, y2=350.0),
        vehicle_class=VehicleClass.BUS,
        confidence=0.88
    )
    tracked_1 = tracker.update([v_f1], frame_id=1, timestamp=0.1)
    track_id = tracked_1[0].track_id

    # Frame 2 & Frame 3: Detector misses the vehicle completely (0 detections)
    _ = tracker.update([], frame_id=2, timestamp=0.2)
    _ = tracker.update([], frame_id=3, timestamp=0.3)

    # Active tracks should still hold the lost track
    active = tracker.get_active_tracks()
    assert track_id in active
    assert active[track_id]["lost"] == 2

    # Frame 4: Vehicle is detected again at x=115, y=120 (moved slightly)
    v_f4 = VehicleDetection(
        box=BoundingBox(x1=115.0, y1=120.0, x2=315.0, y2=370.0),
        vehicle_class=VehicleClass.BUS,
        confidence=0.89
    )
    tracked_4 = tracker.update([v_f4], frame_id=4, timestamp=0.4)
    assert len(tracked_4) == 1
    assert tracked_4[0].track_id == track_id, "Tracker must recover track ID after temporary detection loss"
    assert tracker.get_active_tracks()[track_id]["lost"] == 0
