"""Deterministic mock vehicle and plate detectors for testing and offline benchmarking."""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from vehicle_intelligence.core.domain import (
    BoundingBox,
    PlateDetection,
    PlateType,
    VehicleClass,
    VehicleDetection,
)
from vehicle_intelligence.core.interfaces import BasePlateDetector, BaseVehicleDetector


class MockVehicleDetector(BaseVehicleDetector):
    """Generates synthetic, deterministic vehicle detections for test suites and pipelines."""

    def __init__(
        self,
        default_confidence: float = 0.92,
        fixed_detections: Optional[List[VehicleDetection]] = None
    ):
        self._default_confidence = default_confidence
        self._fixed_detections = fixed_detections

    @property
    def model_name(self) -> str:
        return "mock_vehicle_detector_v1"

    def detect(self, image: np.ndarray) -> List[VehicleDetection]:
        if image is None or image.size == 0:
            return []

        if self._fixed_detections is not None:
            return self._fixed_detections

        h, w = image.shape[:2]
        # Generate a realistic centered vehicle bounding box
        box = BoundingBox(
            x1=float(w * 0.20),
            y1=float(h * 0.30),
            x2=float(w * 0.80),
            y2=float(h * 0.85),
        )

        detection = VehicleDetection(
            box=box,
            vehicle_class=VehicleClass.CAR,
            confidence=self._default_confidence,
            metadata={"source": "mock_generator", "simulated_latency_ms": 1.2}
        )
        return [detection]


class MockPlateDetector(BasePlateDetector):
    """Generates synthetic, deterministic plate detections within vehicle bounding boxes."""

    def __init__(self, default_confidence: float = 0.88):
        self._default_confidence = default_confidence

    @property
    def model_name(self) -> str:
        return "mock_plate_detector_v1"

    def detect(
        self,
        image: np.ndarray,
        vehicle_box: Optional[BoundingBox] = None
    ) -> List[PlateDetection]:
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        if vehicle_box is not None:
            # Place plate at the lower center of the vehicle box
            vw = vehicle_box.width
            vh = vehicle_box.height
            px1 = vehicle_box.x1 + vw * 0.35
            py1 = vehicle_box.y1 + vh * 0.70
            px2 = vehicle_box.x1 + vw * 0.65
            py2 = vehicle_box.y1 + vh * 0.85
            box = BoundingBox(x1=px1, y1=py1, x2=px2, y2=py2).clamp(w, h)
        else:
            box = BoundingBox(
                x1=float(w * 0.40),
                y1=float(h * 0.70),
                x2=float(w * 0.60),
                y2=float(h * 0.80),
            ).clamp(w, h)

        plate = PlateDetection(
            box=box,
            confidence=self._default_confidence,
            plate_type=PlateType.PRIVATE,
            metadata={"source": "mock_plate_generator"}
        )
        return [plate]
