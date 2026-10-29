"""Vehicle and Plate Detectors."""

from vehicle_intelligence.detectors.mock_detector import MockPlateDetector, MockVehicleDetector
from vehicle_intelligence.detectors.motion_detector import MotionVehicleDetector
from vehicle_intelligence.detectors.pytorch_yolo import PyTorchYoloVehicleDetector

__all__ = [
    "MockPlateDetector",
    "MockVehicleDetector",
    "MotionVehicleDetector",
    "PyTorchYoloVehicleDetector",
]
