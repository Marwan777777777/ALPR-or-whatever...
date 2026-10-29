"""Classical OpenCV background-subtraction vehicle detector for motion-triggered CCTV gates."""

from __future__ import annotations

from typing import List, Optional
import cv2
import numpy as np

from vehicle_intelligence.core.domain import (
    BoundingBox,
    VehicleClass,
    VehicleDetection,
)
from vehicle_intelligence.core.interfaces import BaseVehicleDetector


class MotionVehicleDetector(BaseVehicleDetector):
    """Detects moving vehicle candidates at gate cameras using background subtraction and contour analysis."""

    def __init__(
        self,
        min_area: int = 4000,
        max_area: Optional[int] = None,
        history: int = 300,
        var_threshold: float = 25.0,
        detect_shadows: bool = False
    ):
        self._min_area = min_area
        self._max_area = max_area
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    @property
    def model_name(self) -> str:
        return "opencv_mog2_motion_detector"

    def detect(self, image: np.ndarray) -> List[VehicleDetection]:
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        fg_mask = self._bg_subtractor.apply(image)

        # Morphological opening and dilation to remove noise and bridge gaps
        cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel)
        dilated = cv2.dilate(cleaned, self._kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[VehicleDetection] = []
        max_limit = self._max_area or (h * w * 0.95)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self._min_area <= area <= max_limit:
                x, y, cw, ch = cv2.boundingRect(cnt)
                # Compute aspect ratio check (vehicles typically aspect ratio between 0.6 and 3.5)
                aspect = cw / float(ch) if ch > 0 else 0
                if 0.5 <= aspect <= 4.0:
                    box = BoundingBox(
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + cw),
                        y2=float(y + ch),
                    ).clamp(w, h)

                    # Estimate confidence proportionally to contour area stability
                    confidence = min(0.95, max(0.50, 0.50 + (area / (w * h)) * 2.0))

                    detections.append(
                        VehicleDetection(
                            box=box,
                            vehicle_class=VehicleClass.UNKNOWN,
                            confidence=round(confidence, 3),
                            metadata={"area": area, "aspect_ratio": round(aspect, 2)}
                        )
                    )

        return detections
