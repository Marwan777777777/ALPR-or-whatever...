"""Tests for vehicle and plate detectors."""

from pathlib import Path
import numpy as np
import pytest

from vehicle_intelligence.core.domain import VehicleClass
from vehicle_intelligence.core.exceptions import ModelWeightsNotFoundError
from vehicle_intelligence.detectors.mock_detector import MockPlateDetector, MockVehicleDetector
from vehicle_intelligence.detectors.motion_detector import MotionVehicleDetector
from vehicle_intelligence.detectors.pytorch_yolo import PyTorchYoloVehicleDetector


def test_mock_vehicle_detector(synthetic_image):
    detector = MockVehicleDetector(default_confidence=0.95)
    assert detector.model_name == "mock_vehicle_detector_v1"

    detections = detector.detect(synthetic_image)
    assert len(detections) == 1
    assert detections[0].vehicle_class == VehicleClass.CAR
    assert detections[0].confidence == 0.95
    assert detections[0].box.width > 0
    assert detections[0].box.height > 0


def test_mock_plate_detector(synthetic_image):
    v_detector = MockVehicleDetector()
    p_detector = MockPlateDetector(default_confidence=0.88)

    v_dets = v_detector.detect(synthetic_image)
    assert len(v_dets) == 1

    plates = p_detector.detect(synthetic_image, vehicle_box=v_dets[0].box)
    assert len(plates) == 1
    assert plates[0].confidence == 0.88
    # Plate should be localized inside or near the lower portion of the vehicle box
    assert plates[0].box.y1 >= v_dets[0].box.y1


def test_motion_detector(synthetic_image):
    detector = MotionVehicleDetector(min_area=500)
    assert detector.model_name == "opencv_mog2_motion_detector"

    # Feed background frame
    bg_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        _ = detector.detect(bg_frame)

    # Frame with moving object (synthetic image with large white rectangle)
    dets = detector.detect(synthetic_image)
    assert isinstance(dets, list)


def test_pytorch_yolo_raises_when_weights_missing(tmp_path: Path):
    non_existent = tmp_path / "missing_yolov10m.pt"
    with pytest.raises(ModelWeightsNotFoundError) as exc_info:
        PyTorchYoloVehicleDetector(weights_path=non_existent)

    assert "missing_yolov10m" in str(exc_info.value)
    assert "not found at" in str(exc_info.value)


def test_pytorch_yolo_plate_raises_when_weights_missing(tmp_path: Path):
    from vehicle_intelligence.detectors.pytorch_yolo import PyTorchYoloPlateDetector
    non_existent = tmp_path / "missing_plate_weights.pt"
    with pytest.raises(ModelWeightsNotFoundError) as exc_info:
        PyTorchYoloPlateDetector(weights_path=non_existent)

    assert "missing_plate_weights" in str(exc_info.value)
    assert "not found at" in str(exc_info.value)


def test_pytorch_yolo_plate_detector_inference_on_checkpoint():
    weights_path = Path("artifacts/runs/detect/ealpr_baseline_yolov8n/weights/best.pt")
    if not weights_path.is_file():
        pytest.skip("Trained weights not found")

    import cv2
    from vehicle_intelligence.detectors.pytorch_yolo import PyTorchYoloPlateDetector
    detector = PyTorchYoloPlateDetector(weights_path=weights_path, device="cuda")
    assert detector.model_name == "pytorch_yolo_plate_best"

    test_img_path = Path("artifacts/ealpr_yolo_dataset/images/test/0003.jpg")
    if not test_img_path.is_file():
        pytest.skip("Test image not found")

    img = cv2.imread(str(test_img_path))
    assert img is not None

    detections = detector.detect(img)
    assert len(detections) >= 1
    det = detections[0]
    assert det.confidence > 0.5
    assert det.box.width > 0
    assert det.box.height > 0


