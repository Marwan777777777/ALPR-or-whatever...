"""Vehicle Intelligence orchestration engine coordinating detectors, tracker, and OCR."""

from __future__ import annotations

import time
from typing import Dict, List, Optional
import cv2
import numpy as np
from pydantic import BaseModel, Field

from vehicle_intelligence.core.domain import (
    PlateDetection,
    PlateTextResult,
    VehicleDetection,
    VehicleSessionSummary,
)
from vehicle_intelligence.core.interfaces import (
    BaseEvidenceAggregator,
    BasePlateDetector,
    BasePlateOCR,
    BaseVehicleDetector,
    BaseVehicleTracker,
)
from vehicle_intelligence.detectors.mock_detector import MockPlateDetector, MockVehicleDetector
from vehicle_intelligence.ocr.mock_ocr import MockPlateOCR
from vehicle_intelligence.tracking.aggregator import MultiFrameEvidenceAggregator
from vehicle_intelligence.tracking.tracker import CentroidIoUTracker


class FrameProcessingResult(BaseModel):
    """Result of processing a single image or video frame."""
    frame_id: int
    timestamp: float
    latency_ms: float
    vehicle_detections: List[VehicleDetection] = Field(default_factory=list)
    plate_detections: List[PlateDetection] = Field(default_factory=list)
    plate_readings: List[PlateTextResult] = Field(default_factory=list)


class VehicleIntelligenceEngine:
    """Production coordinator executing the end-to-end vehicle intelligence pipeline."""

    def __init__(
        self,
        vehicle_detector: Optional[BaseVehicleDetector] = None,
        plate_detector: Optional[BasePlateDetector] = None,
        plate_ocr: Optional[BasePlateOCR] = None,
        tracker: Optional[BaseVehicleTracker] = None,
        aggregator: Optional[BaseEvidenceAggregator] = None,
    ):
        self.vehicle_detector = vehicle_detector or MockVehicleDetector()
        self.plate_detector = plate_detector or MockPlateDetector()
        self.plate_ocr = plate_ocr or MockPlateOCR()
        self.tracker = tracker or CentroidIoUTracker()
        self.aggregator = aggregator or MultiFrameEvidenceAggregator()

    def process_frame(
        self,
        image: np.ndarray,
        frame_id: int = 0,
        timestamp: Optional[float] = None
    ) -> FrameProcessingResult:
        """Runs vehicle detection, tracking, plate localization, and OCR on a single frame."""
        t_start = time.perf_counter()
        ts = timestamp if timestamp is not None else time.time()

        # Step 1: Detect vehicles
        vehicles = self.vehicle_detector.detect(image)

        # Step 2: Update tracking
        tracked_vehicles = self.tracker.update(vehicles, frame_id=frame_id, timestamp=ts)

        all_plates: List[PlateDetection] = []
        all_readings: List[PlateTextResult] = []

        h, w = image.shape[:2]

        # Step 3: For each tracked vehicle, localize plate and run OCR
        for vehicle in tracked_vehicles:
            plates = self.plate_detector.detect(image, vehicle_box=vehicle.box)
            for plate in plates:
                plate.vehicle_track_id = vehicle.track_id
                all_plates.append(plate)

                # Extract plate crop safely
                clamped = plate.box.clamp(w, h)
                x1, y1, x2, y2 = clamped.to_int_tuple()
                if x2 > x1 and y2 > y1:
                    plate_crop = image[y1:y2, x1:x2]
                    reading = self.plate_ocr.read(plate_crop)
                    all_readings.append(reading)

                    # Accumulate observation
                    if vehicle.track_id is not None:
                        self.aggregator.record_observation(
                            track_id=vehicle.track_id,
                            timestamp=ts,
                            vehicle_detection=vehicle,
                            plate_detection=plate,
                            plate_reading=reading,
                        )

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return FrameProcessingResult(
            frame_id=frame_id,
            timestamp=ts,
            latency_ms=round(latency_ms, 2),
            vehicle_detections=tracked_vehicles,
            plate_detections=all_plates,
            plate_readings=all_readings,
        )

    def finalize_vehicle_session(self, track_id: int) -> Optional[VehicleSessionSummary]:
        """Finalizes and compiles aggregated evidence for a vehicle track."""
        return self.aggregator.finalize_track(track_id)
