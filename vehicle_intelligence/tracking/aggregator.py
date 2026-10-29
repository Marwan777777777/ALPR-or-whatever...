"""Multi-frame evidence aggregator for per-vehicle session consolidation."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional
import numpy as np

from vehicle_intelligence.core.domain import (
    BoundingBox,
    PlateDetection,
    PlateTextResult,
    VehicleAttributes,
    VehicleDetection,
    VehicleSessionSummary,
)
from vehicle_intelligence.core.interfaces import BaseEvidenceAggregator


class MultiFrameEvidenceAggregator(BaseEvidenceAggregator):
    """Accumulates and reconciles evidence over consecutive CCTV frames for each vehicle transit."""

    def __init__(self, min_confidence_threshold: float = 0.75):
        self._min_confidence_threshold = min_confidence_threshold
        # track_id -> observation buffer
        self._buffers: Dict[int, Dict] = {}

    def record_observation(
        self,
        track_id: int,
        timestamp: float,
        vehicle_detection: VehicleDetection,
        plate_detection: Optional[PlateDetection] = None,
        plate_reading: Optional[PlateTextResult] = None,
        attributes: Optional[VehicleAttributes] = None,
    ) -> None:
        if track_id not in self._buffers:
            self._buffers[track_id] = {
                "first_seen": timestamp,
                "last_seen": timestamp,
                "frames_count": 0,
                "vehicle_detections": [],
                "plate_detections": [],
                "plate_readings": [],
                "attributes_list": [],
            }

        buf = self._buffers[track_id]
        buf["last_seen"] = timestamp
        buf["frames_count"] += 1
        buf["vehicle_detections"].append(vehicle_detection)

        if plate_detection is not None:
            buf["plate_detections"].append(plate_detection)
        if plate_reading is not None:
            buf["plate_readings"].append(plate_reading)
        if attributes is not None:
            buf["attributes_list"].append(attributes)

    def finalize_track(self, track_id: int) -> Optional[VehicleSessionSummary]:
        """Consolidates buffered observations into a finalized VehicleSessionSummary."""
        buf = self._buffers.pop(track_id, None)
        if not buf or buf["frames_count"] == 0:
            return None

        # Determine best vehicle bounding box (highest confidence and largest area)
        best_vehicle = max(
            buf["vehicle_detections"],
            key=lambda d: d.confidence * np.sqrt(d.box.area)
        )

        # Consolidate license plate OCR
        review_reasons: List[str] = []
        needs_review = False
        final_plate_reading: Optional[PlateTextResult] = None
        best_plate_det: Optional[PlateDetection] = None

        if buf["plate_detections"]:
            best_plate_det = max(buf["plate_detections"], key=lambda p: p.confidence)

        if buf["plate_readings"]:
            # Pick highest confidence OCR reading
            final_plate_reading = max(buf["plate_readings"], key=lambda r: r.confidence)

            # Check if there was disagreement among candidate readings
            texts = [r.raw_text for r in buf["plate_readings"] if r.raw_text]
            if len(set(texts)) > 1:
                review_reasons.append(f"Ambiguous OCR: multiple variations observed ({len(set(texts))} unique variants)")

            if final_plate_reading.confidence < self._min_confidence_threshold:
                needs_review = True
                review_reasons.append(
                    f"Plate OCR confidence ({final_plate_reading.confidence:.2f}) is below threshold ({self._min_confidence_threshold})"
                )
            if final_plate_reading.needs_review:
                needs_review = True
                review_reasons.extend(final_plate_reading.review_reasons)
        else:
            needs_review = True
            review_reasons.append("No license plate detected or readable across all observed frames")

        # Consolidate attributes if available
        final_attributes: Optional[VehicleAttributes] = None
        if buf["attributes_list"]:
            final_attributes = max(buf["attributes_list"], key=lambda a: a.confidence)

        summary = VehicleSessionSummary(
            track_id=track_id,
            first_seen_timestamp=buf["first_seen"],
            last_seen_timestamp=buf["last_seen"],
            total_frames_tracked=buf["frames_count"],
            best_vehicle_box=best_vehicle.box,
            best_vehicle_confidence=best_vehicle.confidence,
            plate_detection=best_plate_det,
            plate_reading=final_plate_reading,
            attributes=final_attributes,
            stickers=[],
            permits=[],
            needs_review=needs_review,
            review_reasons=list(dict.fromkeys(review_reasons)),  # deduplicate
        )
        return summary
