"""Test fixtures and synthetic media generators for pytest."""

import numpy as np
import pytest
import cv2

from vehicle_intelligence.core.domain import BoundingBox, VehicleClass, VehicleDetection


@pytest.fixture
def synthetic_image():
    """Generates a synthetic 640x480 BGR image with a white rectangle simulating a vehicle."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simulated vehicle body
    cv2.rectangle(img, (150, 150), (490, 380), (180, 180, 180), -1)
    # Draw simulated license plate
    cv2.rectangle(img, (270, 320), (370, 360), (255, 255, 255), -1)
    return img


@pytest.fixture
def sample_box():
    return BoundingBox(x1=100.0, y1=150.0, x2=300.0, y2=400.0)


@pytest.fixture
def sample_vehicle_detection(sample_box):
    return VehicleDetection(
        box=sample_box,
        vehicle_class=VehicleClass.CAR,
        confidence=0.92,
        track_id=1,
    )
