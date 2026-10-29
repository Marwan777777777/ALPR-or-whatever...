"""Privacy-aware processing, redaction, and data retention management for CCTV imagery."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

from vehicle_intelligence.core.domain import BoundingBox, PlateDetection, VehicleDetection


class PrivacyRedactor:
    """Provides privacy-compliant redaction for stored CCTV imagery and metadata."""

    @staticmethod
    def blur_roi(
        image: np.ndarray,
        box: BoundingBox,
        kernel_size: int = 51
    ) -> np.ndarray:
        """Applies strong Gaussian blur to a specified region of interest (e.g. driver face)."""
        output = image.copy()
        h, w = image.shape[:2]
        clamped = box.clamp(w, h)
        x1, y1, x2, y2 = clamped.to_int_tuple()

        if x2 > x1 and y2 > y1:
            # Ensure odd kernel size
            ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            roi = output[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(roi, (ksize, ksize), 30)
            output[y1:y2, x1:x2] = blurred

        return output

    @staticmethod
    def mask_surroundings(
        image: np.ndarray,
        keep_boxes: List[BoundingBox],
        mask_color: int = 40
    ) -> np.ndarray:
        """Dim non-relevant surrounding regions to protect pedestrians and bystanders."""
        h, w = image.shape[:2]
        mask = np.full((h, w), mask_color, dtype=np.uint8)

        for box in keep_boxes:
            clamped = box.clamp(w, h)
            x1, y1, x2, y2 = clamped.to_int_tuple()
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255

        # Blend original with dimmed mask
        dimmed = (image.astype(np.float32) * 0.3).astype(np.uint8)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        output = np.where(mask_3ch == 255, image, dimmed)
        return output

    @staticmethod
    def anonymize_plate_text(raw_text: str, salt: str = "egypt_cctv_salt") -> str:
        """Hashes plate text with a cryptographic salt for pseudonymized analytics."""
        clean = "".join(raw_text.split()).strip()
        return hashlib.sha256(f"{clean}_{salt}".encode("utf-8")).hexdigest()[:16]


class RetentionPolicyManager:
    """Manages data retention periods and automated pruning of CCTV recordings and plate crops."""

    def __init__(self, raw_retention_days: int = 30, metadata_retention_days: int = 90):
        self.raw_retention_seconds = raw_retention_days * 86400
        self.metadata_retention_seconds = metadata_retention_days * 86400

    def is_expired(self, creation_timestamp: float, is_raw_media: bool = True) -> bool:
        """Checks if an artifact has exceeded its legal retention period."""
        age = time.time() - creation_timestamp
        limit = self.raw_retention_seconds if is_raw_media else self.metadata_retention_seconds
        return age > limit

    def scan_and_prune_expired_files(self, directory: Path | str, is_raw_media: bool = True) -> List[Path]:
        """Identifies and purges files older than the retention threshold."""
        dir_path = Path(directory)
        pruned_files: List[Path] = []
        if not dir_path.exists():
            return pruned_files

        for file_path in dir_path.glob("*"):
            if file_path.is_file():
                mtime = file_path.stat().st_mtime
                if self.is_expired(mtime, is_raw_media=is_raw_media):
                    file_path.unlink()
                    pruned_files.append(file_path)

        return pruned_files
