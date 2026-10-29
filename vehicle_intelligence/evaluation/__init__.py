"""Evaluation contracts and metrics for detection and OCR."""

from vehicle_intelligence.evaluation.detection_eval import (
    DetectionMetricsReport,
    GroundTruthPlate,
    ImageDetectionMatch,
    PlateDetectionEvaluator,
    PredictedPlate,
)
from vehicle_intelligence.evaluation.ocr_eval import (
    EgyptianPlateTextNormalizer,
    GroundTruthPlateText,
    OCRAggregateMetrics,
    PlateMatchScore,
    PlateOCREvaluator,
    PredictedPlateText,
)

__all__ = [
    "DetectionMetricsReport",
    "GroundTruthPlate",
    "ImageDetectionMatch",
    "PlateDetectionEvaluator",
    "PredictedPlate",
    "EgyptianPlateTextNormalizer",
    "GroundTruthPlateText",
    "OCRAggregateMetrics",
    "PlateMatchScore",
    "PlateOCREvaluator",
    "PredictedPlateText",
]
