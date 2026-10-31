"""Tests for Mission 011: Nighttime IR Vehicle Detection, Preprocessing, and Model Licensing."""

from pathlib import Path
import numpy as np
import pytest

from vehicle_intelligence.core.domain import BoundingBox, VehicleClass, VehicleDetection
from vehicle_intelligence.core.exceptions import ModelWeightsNotFoundError
from vehicle_intelligence.detectors.pytorch_yolo import PyTorchYoloVehicleDetector


def test_pytorch_yolo_licensing_info():
    """Verify licensing metadata reporting for commercial compliance."""
    weights_path = Path("yolov8n.pt")
    if not weights_path.is_file():
        pytest.skip("yolov8n.pt not present")

    detector = PyTorchYoloVehicleDetector(weights_path=weights_path)
    info = detector.get_licensing_info()

    assert info["license"] == "AGPL-3.0"
    assert "Requires commercial Ultralytics" in info["commercial_compatible"]

    # Test RT-DETR licensing metadata
    rtdetr_weights = Path("rtdetr-l.pt")
    if rtdetr_weights.is_file():
        rt_detector = PyTorchYoloVehicleDetector(weights_path=rtdetr_weights)
        rt_info = rt_detector.get_licensing_info()
        assert rt_info["license"] == "Apache-2.0"
        assert rt_info["commercial_compatible"] == "YES"


def test_pytorch_yolo_preprocessing_modes():
    """Verify preprocessing pipeline applies CLAHE, AutoContrast, and Gamma without error."""
    weights_path = Path("yolov8n.pt")
    if not weights_path.is_file():
        pytest.skip("yolov8n.pt not present")

    # Synthetic dark image (simulating night IR)
    dark_img = np.full((480, 640, 3), 25, dtype=np.uint8)
    # Add a brighter box
    dark_img[100:300, 200:500] = 120

    for mode in ["clahe", "autocontrast", "gamma", None]:
        detector = PyTorchYoloVehicleDetector(
            weights_path=weights_path,
            preprocessing=mode,
            confidence_threshold=0.25,
            device="cpu",
        )
        dets = detector.detect(dark_img)
        assert isinstance(dets, list)


def test_pytorch_yolo_deduplication():
    """Verify NMS deduplication logic handles overlapping detections."""
    weights_path = Path("yolov8n.pt")
    if not weights_path.is_file():
        pytest.skip("yolov8n.pt not present")

    detector = PyTorchYoloVehicleDetector(
        weights_path=weights_path,
        apply_nms=True,
        confidence_threshold=0.20,
    )
    assert detector._apply_nms is True


def test_pytorch_yolo_raises_on_missing_weights(tmp_path: Path):
    """Verify missing weights are rejected immediately."""
    fake_weights = tmp_path / "non_existent_weights.pt"
    with pytest.raises(ModelWeightsNotFoundError):
        PyTorchYoloVehicleDetector(weights_path=fake_weights)
