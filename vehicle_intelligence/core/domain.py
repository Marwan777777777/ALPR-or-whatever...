"""Domain models and data structures for vehicle intelligence."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, model_validator


class VehicleClass(str, Enum):
    """Recognized vehicle classes."""
    CAR = "car"
    SUV = "suv"
    VAN = "van"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    UNKNOWN = "unknown"


class PlateType(str, Enum):
    """Egyptian license plate category classes."""
    PRIVATE = "private"          # Light Blue banner (ملاكي)
    TAXI = "taxi"                # Orange banner (أجرة)
    COMMERCIAL = "commercial"    # Red/Green banner (نقل / تجاري)
    POLICE = "police"            # Dark Blue banner (شرطة)
    DIPLOMATIC = "diplomatic"    # Green banner (هيئات دبلوماسية)
    CUSTOMS = "customs"          # Yellow banner (جمارك)
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """Axis-aligned 2D bounding box [x1, y1, x2, y2] in pixel or normalized space."""
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")

    @model_validator(mode="after")
    def validate_coordinates(self) -> BoundingBox:
        if self.x2 < self.x1:
            raise ValueError(f"Invalid BoundingBox: x2 ({self.x2}) cannot be smaller than x1 ({self.x1})")
        if self.y2 < self.y1:
            raise ValueError(f"Invalid BoundingBox: y2 ({self.y2}) cannot be smaller than y1 ({self.y1})")
        return self

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        if self.height <= 0:
            return 0.0
        return self.width / self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_int_tuple(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))

    def clamp(self, width: int, height: int) -> BoundingBox:
        """Clamps the bounding box to image boundaries [0, 0, width, height]."""
        return BoundingBox(
            x1=max(0.0, min(float(width), self.x1)),
            y1=max(0.0, min(float(height), self.y1)),
            x2=max(0.0, min(float(width), self.x2)),
            y2=max(0.0, min(float(height), self.y2)),
        )

    def iou(self, other: BoundingBox) -> float:
        """Computes Intersection over Union (IoU) with another bounding box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        intersection_w = max(0.0, ix2 - ix1)
        intersection_h = max(0.0, iy2 - iy1)
        intersection_area = intersection_w * intersection_h

        union_area = self.area + other.area - intersection_area
        if union_area <= 0.0:
            return 0.0
        return intersection_area / union_area

    def intersection(self, other: BoundingBox) -> float:
        """Computes intersection area with another bounding box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        w = max(0.0, ix2 - ix1)
        h = max(0.0, iy2 - iy1)
        return w * h

    def intersection_over_self(self, other: BoundingBox) -> float:
        """Computes ratio of intersection area over self area (IoP)."""
        if self.area <= 0.0:
            return 0.0
        return self.intersection(other) / self.area


class VehicleDetection(BaseModel):
    """Structured detection result for a vehicle candidate."""
    box: BoundingBox
    vehicle_class: VehicleClass = VehicleClass.UNKNOWN
    confidence: float = Field(..., ge=0.0, le=1.0)
    track_id: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlateDetection(BaseModel):
    """Structured detection result for a license plate candidate."""
    box: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    plate_type: PlateType = PlateType.UNKNOWN
    vehicle_track_id: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlateTextResult(BaseModel):
    """Structured Egyptian license plate reading result."""
    raw_text: str = Field(default="", description="Consolidated raw OCR text")
    arabic_letters: Optional[str] = Field(default=None, description="Extracted Arabic letters e.g. 'س ع د'")
    arabic_digits: Optional[str] = Field(default=None, description="Extracted Arabic or Eastern-Arabic numerals e.g. '١ ٢ ٣' or '1 2 3'")
    latin_transcription: Optional[str] = Field(default=None, description="Standardized Latin transliteration")
    confidence: float = Field(..., ge=0.0, le=1.0)
    needs_review: bool = Field(default=False, description="Flag indicating human operator review is required")
    review_reasons: List[str] = Field(default_factory=list)


class VehicleAttributes(BaseModel):
    """Progressive identification attributes for a vehicle."""
    make: Optional[str] = None
    model: Optional[str] = None
    year_range: Optional[str] = None
    primary_color: Optional[str] = None
    body_type: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_review: bool = Field(default=True)
    review_reasons: List[str] = Field(default_factory=list)


class StickerDetection(BaseModel):
    """Visible vehicle stickers or decals."""
    box: BoundingBox
    sticker_type: str = "unknown"
    confidence: float = Field(..., ge=0.0, le=1.0)
    text_content: Optional[str] = None


class PermitResult(BaseModel):
    """Visible gate / parking permit identification."""
    permit_type: str = "unknown"
    permit_id: Optional[str] = None
    validity_status: str = "needs_review"  # 'valid', 'expired', 'unreadable', 'needs_review'
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_review: bool = True
    review_reasons: List[str] = Field(default_factory=list)


class VehicleSessionSummary(BaseModel):
    """Aggregated per-vehicle evidence over multiple CCTV frames."""
    track_id: int
    first_seen_timestamp: float
    last_seen_timestamp: float
    total_frames_tracked: int
    best_vehicle_box: BoundingBox
    best_vehicle_confidence: float
    plate_detection: Optional[PlateDetection] = None
    plate_reading: Optional[PlateTextResult] = None
    attributes: Optional[VehicleAttributes] = None
    stickers: List[StickerDetection] = Field(default_factory=list)
    permits: List[PermitResult] = Field(default_factory=list)
    alternative_readings: List[PlateTextResult] = Field(
        default_factory=list,
        description="Conflicting or alternative candidate readings observed during the track",
    )
    best_evidence_frame_id: Optional[int] = Field(
        default=None,
        description="Frame index of the highest-confidence evidence observation",
    )
    best_crop_path: Optional[str] = Field(
        default=None,
        description="Path to best cropped plate image evidence",
    )
    has_conflicting_reads: bool = Field(
        default=False,
        description="True if multiple distinct non-empty plate strings were observed across the transit",
    )
    needs_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)


class AssociationStatus(str, Enum):
    """Plate-to-vehicle association status."""
    CONFIDENT = "confident"
    AMBIGUOUS = "ambiguous"
    UNASSIGNED = "unassigned"


class PlateVehicleAssociation(BaseModel):
    """Structured result of associating a detected plate with a vehicle candidate/track."""
    plate_box: BoundingBox
    plate_confidence: float = Field(..., ge=0.0, le=1.0)
    vehicle_track_id: Optional[int] = None
    vehicle_box: Optional[BoundingBox] = None
    vehicle_class: Optional[VehicleClass] = None
    association_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: AssociationStatus = AssociationStatus.UNASSIGNED
    competing_vehicle_ids: List[int] = Field(default_factory=list)
    containment_ratio: float = Field(default=0.0, description="Intersection over plate area (IoP)")
    relative_center_offset: Tuple[float, float] = Field(default=(0.0, 0.0))
    scale_ratio: float = Field(default=0.0, description="Plate area / Vehicle area")
    needs_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)


class VehicleTransitEvent(BaseModel):
    """Upgraded coherent vehicle transit event consolidating vehicle track and associated plate readings."""
    event_type: str = "VEHICLE_TRANSIT"
    vehicle_track_id: int = Field(..., description="Independent vehicle track identifier")
    vehicle_class: VehicleClass = Field(default=VehicleClass.UNKNOWN, description="Vehicle classification")
    vehicle_confidence: float = Field(..., ge=0.0, le=1.0, description="Vehicle detection/track confidence")
    first_seen_timestamp: float = Field(..., description="Timestamp of first vehicle observation")
    last_seen_timestamp: float = Field(..., description="Timestamp of last vehicle observation")
    total_frames_tracked: int = Field(default=1, description="Total frames vehicle was tracked")
    plate_reading: Optional[PlateTextResult] = Field(default=None, description="Consolidated primary plate OCR reading")
    plate_text: str = Field(default="", description="Primary plate logical text")
    letters: str = Field(default="", description="Primary plate Arabic letters")
    digits: str = Field(default="", description="Primary plate Arabic numerals")
    plate_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence of primary plate reading")
    plate_association_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence of plate-to-vehicle association")
    association_status: AssociationStatus = Field(default=AssociationStatus.CONFIDENT)
    best_vehicle_box: BoundingBox = Field(..., description="Best vehicle bounding box across transit")
    best_plate_box: Optional[BoundingBox] = Field(default=None, description="Best plate bounding box")
    best_evidence_frame_id: Optional[int] = Field(default=None, description="Frame index of best evidence")
    best_crop_path: Optional[str] = Field(default=None, description="Path to best cropped plate image")
    best_evidence_source_id: Optional[str] = Field(default=None, description="Source ID of best evidence frame")
    observation_timestamps: List[float] = Field(default_factory=list, description="Timestamps of all observations in track")
    alternative_readings: List[PlateTextResult] = Field(default_factory=list, description="Differing candidate readings observed")
    has_conflicting_reads: bool = Field(default=False, description="True if multiple distinct plate strings were read")
    has_conflicting_association: bool = Field(default=False, description="True if plate association was ambiguous across transit")
    competing_vehicle_ids: List[int] = Field(default_factory=list, description="Alternative vehicle candidate IDs if ambiguous")
    attributes: Optional[VehicleAttributes] = None
    needs_review: bool = Field(default=False, description="Requires human verification")
    review_reasons: List[str] = Field(default_factory=list, description="Specific triggers requiring human review")


class ALPRPlateResult(BaseModel):
    """Complete, end-to-end ALPR result schema for CLI and API."""
    source_id: str = Field(..., description="Image name, video frame, or camera stream ID")
    timestamp: float = Field(default=0.0, description="Timestamp of inference")
    plate_text: str = Field(default="", description="Logical full plate text in standard order")
    letters: str = Field(default="", description="Extracted Arabic letters (read RTL)")
    digits: str = Field(default="", description="Extracted Arabic numerals (read LTR)")
    per_character_confidences: List[float] = Field(default_factory=list, description="Confidence per detected character")
    plate_confidence: float = Field(..., ge=0.0, le=1.0, description="Composite plate confidence")
    detector_confidence: float = Field(..., ge=0.0, le=1.0, description="Plate detector bounding box confidence")
    needs_review: bool = Field(default=False, description="Flag indicating human review is required")
    review_reasons: List[str] = Field(default_factory=list, description="Explanations if needs_review is True")
    bounding_box: BoundingBox = Field(..., description="Plate bounding box [x1, y1, x2, y2]")
    vehicle_bounding_box: Optional[BoundingBox] = Field(default=None, description="Enclosing vehicle bounding box")
    track_id: Optional[int] = Field(default=None, description="Tracking ID if video transit mode")
    vehicle_track_id: Optional[int] = Field(default=None, description="Associated vehicle track ID")
    association_confidence: float = Field(default=0.0, description="Confidence of plate-to-vehicle association")
    association_status: AssociationStatus = Field(default=AssociationStatus.UNASSIGNED)
    crop_path: Optional[str] = Field(default=None, description="Path to saved plate crop image")
    annotated_image_path: Optional[str] = Field(default=None, description="Path to saved annotated visualization")


