"""Abstract interfaces for pluggable modules: detectors, OCR, trackers, aggregators, and classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from vehicle_intelligence.core.domain import (
    BoundingBox,
    PlateDetection,
    PlateTextResult,
    PermitResult,
    StickerDetection,
    VehicleAttributes,
    VehicleDetection,
    VehicleSessionSummary,
)


class BaseVehicleDetector(ABC):
    """Interface for vehicle detection models."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[VehicleDetection]:
        """Detect vehicles in an input image (BGR numpy array).

        Args:
            image: Image array formatted as BGR (OpenCV standard), shape [H, W, C].

        Returns:
            List of VehicleDetection instances with bounding boxes and confidence scores.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the detector model."""
        pass


class BasePlateDetector(ABC):
    """Interface for license plate detection models."""

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        vehicle_box: Optional[BoundingBox] = None
    ) -> List[PlateDetection]:
        """Detect license plates either within the full image or within a vehicle ROI.

        Args:
            image: Image array formatted as BGR.
            vehicle_box: Optional bounding box of the parent vehicle.

        Returns:
            List of PlateDetection instances.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the plate detector."""
        pass


class BasePlateOCR(ABC):
    """Interface for Egyptian license plate optical character recognition."""

    @abstractmethod
    def read(self, plate_crop: np.ndarray) -> PlateTextResult:
        """Extract text, letters, and numbers from a license plate crop.

        Args:
            plate_crop: Cropped image of the license plate (BGR numpy array).

        Returns:
            PlateTextResult containing extracted characters, confidence, and review flags.
        """
        pass

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Name of the OCR engine."""
        pass


class BaseVehicleTracker(ABC):
    """Interface for multi-object tracking across video frames."""

    @abstractmethod
    def update(
        self,
        detections: List[VehicleDetection],
        frame_id: int,
        timestamp: float
    ) -> List[VehicleDetection]:
        """Update tracker with detections from current frame and return tracked objects with track_id assigned.

        Args:
            detections: Vehicle detections in the current frame.
            frame_id: Monotonically increasing frame index.
            timestamp: Frame presentation timestamp in seconds.

        Returns:
            Vehicle detections with populated track_id.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracking state."""
        pass


class BaseVehicleAttributeClassifier(ABC):
    """Interface for vehicle make, model, year, and color classification."""

    @abstractmethod
    def classify(self, vehicle_crop: np.ndarray) -> VehicleAttributes:
        """Estimate vehicle attributes from a high-quality vehicle crop.

        Args:
            vehicle_crop: Cropped vehicle image.

        Returns:
            VehicleAttributes with estimated make, model, year range, and color.
        """
        pass


class BaseStickerDetector(ABC):
    """Interface for identifying decals, parking tags, and security stickers."""

    @abstractmethod
    def detect(self, vehicle_crop: np.ndarray) -> List[StickerDetection]:
        """Detect visible stickers or decals on vehicle surface."""
        pass


class BasePermitReader(ABC):
    """Interface for reading windshield parking/facility permits."""

    @abstractmethod
    def read(self, windshield_crop: np.ndarray) -> PermitResult:
        """Read and validate windshield permits."""
        pass


class BaseEvidenceAggregator(ABC):
    """Interface for per-vehicle multi-frame evidence collection."""

    @abstractmethod
    def record_observation(
        self,
        track_id: int,
        timestamp: float,
        vehicle_detection: VehicleDetection,
        plate_detection: Optional[PlateDetection] = None,
        plate_reading: Optional[PlateTextResult] = None,
        attributes: Optional[VehicleAttributes] = None,
    ) -> None:
        """Add a frame observation for a tracked vehicle."""
        pass

    @abstractmethod
    def finalize_track(self, track_id: int) -> Optional[VehicleSessionSummary]:
        """Finalize and aggregate evidence for a vehicle that has exited or completed transit."""
        pass
