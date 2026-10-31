"""PyTorch and TorchScript / ONNX pluggable detector for YOLO-family models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from vehicle_intelligence.core.domain import (
    BoundingBox,
    PlateDetection,
    PlateType,
    VehicleClass,
    VehicleDetection,
)
from vehicle_intelligence.core.exceptions import (
    DeviceUnavailableError,
    ModelWeightsNotFoundError,
    VehicleIntelligenceError,
)
from vehicle_intelligence.core.interfaces import BasePlateDetector, BaseVehicleDetector


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
        preprocessing: Optional[str] = None,
        apply_nms: bool = True,
    ):
        self._weights_path = Path(weights_path)
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._input_size = input_size
        self._device_str = device.lower()
        self._half_precision = half_precision
        self._preprocessing = preprocessing.lower() if preprocessing else None
        self._apply_nms = apply_nms
        self._model = None

        self._validate_and_initialize()

    @property
    def model_name(self) -> str:
        return f"pytorch_yolo_{self._weights_path.stem}"

    def get_licensing_info(self) -> Dict[str, str]:
        """Returns licensing and commercial deployment metadata for active model."""
        stem = self._weights_path.stem.lower()
        if "rtdetr" in stem:
            return {
                "architecture": "RT-DETR (Real-Time Detection Transformer)",
                "weights_path": str(self._weights_path),
                "license": "Apache-2.0",
                "commercial_compatible": "YES",
                "notes": "Permissively licensed; no viral copyleft obligations for SaaS or embedded use."
            }
        else:
            return {
                "architecture": "YOLO-family (Ultralytics)",
                "weights_path": str(self._weights_path),
                "license": "AGPL-3.0",
                "commercial_compatible": "NO (Requires commercial Ultralytics enterprise license)",
                "notes": "Copyleft terms apply if exposed over a network or distributed."
            }

    def _apply_preprocessing(self, image: np.ndarray) -> np.ndarray:
        if self._preprocessing is None:
            return image
        if self._preprocessing == "clahe":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            res = clahe.apply(gray)
            return cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        elif self._preprocessing == "autocontrast":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            p1, p99 = np.percentile(gray, (1, 99))
            res = np.clip((gray - p1) / max(1.0, (p99 - p1)) * 255.0, 0, 255).astype(np.uint8)
            return cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        elif self._preprocessing == "gamma":
            lut = np.array([((i / 255.0) ** 0.6) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, lut)
        return image

    def _validate_and_initialize(self) -> None:
        """Verifies weight availability and device capabilities without fabricating outputs."""
        if not self._weights_path.is_file():
            raise ModelWeightsNotFoundError(
                model_name=self._weights_path.stem,
                weights_path=str(self._weights_path)
            )

        try:
            import torch
        except ImportError as e:
            raise VehicleIntelligenceError(
                message="PyTorch is required for PyTorchYoloVehicleDetector but is not installed.",
                actionable_advice="Install PyTorch via: pip install torch torchvision"
            ) from e

        if "cuda" in self._device_str and not torch.cuda.is_available():
            raise DeviceUnavailableError(
                requested_device=self._device_str,
                available_devices="cpu (CUDA is not available in current PyTorch installation)"
            )

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self._weights_path))
        except Exception:
            self._model = None

    def detect(self, image: np.ndarray) -> List[VehicleDetection]:
        if image is None or image.size == 0:
            return []

        if self._model is None:
            raise VehicleIntelligenceError(
                message=f"Model '{self.model_name}' has not initialized an active inference session.",
                actionable_advice="Ensure weights are loaded and network architecture is instantiated."
            )

        processed_img = self._apply_preprocessing(image)

        # Classes to detect: COCO vehicle classes (2: car, 3: motorcycle, 5: bus, 7: truck)
        coco_classes = list(self.COCO_VEHICLE_MAP.keys())

        results = self._model.predict(
            source=processed_img,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            classes=coco_classes,
            imgsz=self._input_size,
            device=self._device_str,
            half=self._half_precision,
            verbose=False,
        )

        detections: List[VehicleDetection] = []
        res = results[0]
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)

            # Optional NMS deduplication (especially valuable for transformer-based set predictions)
            if self._apply_nms and len(boxes) > 1:
                indices = cv2.dnn.NMSBoxes(
                    bboxes=[[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes],
                    scores=[float(s) for s in confs],
                    score_threshold=self._confidence_threshold,
                    nms_threshold=self._iou_threshold,
                )
                if len(indices) > 0:
                    keep_idx = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in indices]
                    boxes = boxes[keep_idx]
                    confs = confs[keep_idx]
                    cls_ids = cls_ids[keep_idx]

            for (bx1, by1, bx2, by2), score, cid in zip(boxes, confs, cls_ids):
                v_class = self.COCO_VEHICLE_MAP.get(cid, VehicleClass.UNKNOWN)
                detections.append(
                    VehicleDetection(
                        box=BoundingBox(x1=float(bx1), y1=float(by1), x2=float(bx2), y2=float(by2)),
                        confidence=round(float(score), 4),
                        vehicle_class=v_class,
                        metadata={
                            "coco_class_id": int(cid),
                            "preprocessing": self._preprocessing,
                            "detector_model": self.model_name,
                        },
                    )
                )

        return detections



class PyTorchYoloPlateDetector(BasePlateDetector):
    """Production wrapper for YOLO license plate detection models (YOLOv8, YOLOv10, YOLOv11).

    Enforces strict weight verification and device capability checks.
    """

    def __init__(
        self,
        weights_path: str | Path,
        confidence_threshold: float = 0.25,
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
        return f"pytorch_yolo_plate_{self._weights_path.stem}"

    def _validate_and_initialize(self) -> None:
        if not self._weights_path.is_file():
            raise ModelWeightsNotFoundError(
                model_name=self._weights_path.stem,
                weights_path=str(self._weights_path)
            )

        try:
            import torch
        except ImportError as e:
            raise VehicleIntelligenceError(
                message="PyTorch is required for PyTorchYoloPlateDetector but is not installed.",
                actionable_advice="Install PyTorch via: pip install torch torchvision"
            ) from e

        if "cuda" in self._device_str and not torch.cuda.is_available():
            raise DeviceUnavailableError(
                requested_device=self._device_str,
                available_devices="cpu (CUDA is not available in current PyTorch installation)"
            )

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self._weights_path))
        except Exception:
            # Fallback or delay initialization if ultralytics is not yet installed
            pass

    def detect(
        self,
        image: np.ndarray,
        vehicle_box: Optional[BoundingBox] = None
    ) -> List[PlateDetection]:
        if image is None or image.size == 0:
            return []

        if self._model is None:
            raise VehicleIntelligenceError(
                message=f"Model '{self.model_name}' has not initialized an active inference session.",
                actionable_advice="Ensure weights are loaded and network architecture is instantiated."
            )

        h, w = image.shape[:2]
        crop = image
        offset_x, offset_y = 0.0, 0.0

        if vehicle_box is not None:
            # Clamp and crop vehicle ROI
            clamped = vehicle_box.clamp(w, h)
            x1, y1, x2, y2 = clamped.to_int_tuple()
            if x2 > x1 and y2 > y1:
                crop = image[y1:y2, x1:x2]
                offset_x, offset_y = float(x1), float(y1)

        results = self._model.predict(
            source=crop,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            imgsz=self._input_size,
            device=self._device_str,
            half=self._half_precision,
            verbose=False,
        )

        detections: List[PlateDetection] = []
        res = results[0]
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()

            for (bx1, by1, bx2, by2), score in zip(boxes, confs):
                abs_x1 = bx1 + offset_x
                abs_y1 = by1 + offset_y
                abs_x2 = bx2 + offset_x
                abs_y2 = by2 + offset_y

                detections.append(
                    PlateDetection(
                        box=BoundingBox(x1=float(abs_x1), y1=float(abs_y1), x2=float(abs_x2), y2=float(abs_y2)),
                        confidence=round(float(score), 4),
                        plate_type=PlateType.UNKNOWN,
                    )
                )

        return detections
