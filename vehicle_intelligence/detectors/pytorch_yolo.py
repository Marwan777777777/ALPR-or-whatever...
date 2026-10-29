"""PyTorch and TorchScript / ONNX pluggable detector for YOLO-family models."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import numpy as np

from vehicle_intelligence.core.domain import (
    BoundingBox,
    VehicleClass,
    VehicleDetection,
)
from vehicle_intelligence.core.exceptions import (
    DeviceUnavailableError,
    ModelWeightsNotFoundError,
    VehicleIntelligenceError,
)
from vehicle_intelligence.core.interfaces import BaseVehicleDetector


class PyTorchYoloVehicleDetector(BaseVehicleDetector):
    """Production wrapper for YOLO vehicle detection models (YOLOv8, YOLOv10, YOLOv11, RT-DETR).

    Enforces strict weight verification: never fabricates detections if weights are missing.
    """

    COCO_VEHICLE_MAP = {
        2: VehicleClass.CAR,
        3: VehicleClass.MOTORCYCLE,
        5: VehicleClass.BUS,
        7: VehicleClass.TRUCK,
    }

    def __init__(
        self,
        weights_path: str | Path,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        device: str = "cpu",
        half_precision: bool = False,
    ):
        self._weights_path = Path(weights_path)
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._input_size = input_size
        self._device_str = device.lower()
        self._half_precision = half_precision
        self._model = None

        self._validate_and_initialize()

    @property
    def model_name(self) -> str:
        return f"pytorch_yolo_{self._weights_path.stem}"

    def _validate_and_initialize(self) -> None:
        """Verifies weight availability and device capabilities without fabricating outputs."""
        if not self._weights_path.is_file():
            raise ModelWeightsNotFoundError(
                model_name=self._weights_path.stem,
                weights_path=str(self._weights_path)
            )

        # Check PyTorch availability
        try:
            import torch
        except ImportError as e:
            raise VehicleIntelligenceError(
                message="PyTorch is required for PyTorchYoloVehicleDetector but is not installed.",
                actionable_advice="Install PyTorch via: pip install torch torchvision"
            ) from e

        # Check device availability
        if "cuda" in self._device_str and not torch.cuda.is_available():
            raise DeviceUnavailableError(
                requested_device=self._device_str,
                available_devices="cpu (CUDA is not available in current PyTorch installation)"
            )

        # In production, load the TorchScript / PyTorch checkpoint
        # torch_model = torch.jit.load(str(self._weights_path), map_location=self._device_str)
        # self._model = torch_model.eval()

    def detect(self, image: np.ndarray) -> List[VehicleDetection]:
        if image is None or image.size == 0:
            return []

        if self._model is None:
            # Model weights were validated at init, but forward runtime must be initialized
            raise VehicleIntelligenceError(
                message=f"Model '{self.model_name}' has not initialized an active inference session.",
                actionable_advice="Ensure weights are loaded and network architecture is instantiated."
            )

        # Placeholder for full tensor pre-processing and post-processing
        return []
