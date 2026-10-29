"""Core domain models, interfaces, configuration, logging, and privacy rules."""

from vehicle_intelligence.core.domain import (
    BoundingBox,
    PlateDetection,
    PlateTextResult,
    PlateType,
    PermitResult,
    StickerDetection,
    VehicleAttributes,
    VehicleClass,
    VehicleDetection,
    VehicleSessionSummary,
)
from vehicle_intelligence.core.exceptions import (
    DeviceUnavailableError,
    InvalidConfigurationError,
    MediaReadError,
    ModelWeightsNotFoundError,
    VehicleIntelligenceError,
)
from vehicle_intelligence.core.interfaces import (
    BaseEvidenceAggregator,
    BasePermitReader,
    BasePlateDetector,
    BasePlateOCR,
    BaseStickerDetector,
    BaseVehicleAttributeClassifier,
    BaseVehicleDetector,
    BaseVehicleTracker,
)
from vehicle_intelligence.core.config import SystemConfig
from vehicle_intelligence.core.logging import setup_logger
from vehicle_intelligence.core.privacy import PrivacyRedactor, RetentionPolicyManager

__all__ = [
    "BoundingBox",
    "PlateDetection",
    "PlateTextResult",
    "PlateType",
    "PermitResult",
    "StickerDetection",
    "VehicleAttributes",
    "VehicleClass",
    "VehicleDetection",
    "VehicleSessionSummary",
    "DeviceUnavailableError",
    "InvalidConfigurationError",
    "MediaReadError",
    "ModelWeightsNotFoundError",
    "VehicleIntelligenceError",
    "BaseEvidenceAggregator",
    "BasePermitReader",
    "BasePlateDetector",
    "BasePlateOCR",
    "BaseStickerDetector",
    "BaseVehicleAttributeClassifier",
    "BaseVehicleDetector",
    "BaseVehicleTracker",
    "SystemConfig",
    "setup_logger",
    "PrivacyRedactor",
    "RetentionPolicyManager",
]
