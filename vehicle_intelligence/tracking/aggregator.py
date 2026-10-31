"""Multi-frame evidence aggregator for per-vehicle session consolidation."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional
import numpy as np

from vehicle_intelligence.core.domain import (
    AssociationStatus,
    BoundingBox,
    PlateDetection,
    PlateTextResult,
    PlateVehicleAssociation,
    VehicleAttributes,
    VehicleDetection,
    VehicleSessionSummary,
    VehicleTransitEvent,
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
        frame_id: Optional[int] = None,
        crop_path: Optional[str] = None,
        plate_association: Optional[PlateVehicleAssociation] = None,
        source_id: Optional[str] = None,
    ) -> None:
        if track_id not in self._buffers:
            self._buffers[track_id] = {
                "first_seen": timestamp,
                "last_seen": timestamp,
                "frames_count": 0,
                "timestamps": [],
                "vehicle_detections": [],
                "plate_detections": [],
                "plate_readings": [],
                "plate_associations": [],
                "attributes_list": [],
                "frame_ids": [],
                "crop_paths": [],
                "source_ids": [],
            }

        buf = self._buffers[track_id]
        buf["last_seen"] = timestamp
        buf["frames_count"] += 1
        buf["timestamps"].append(timestamp)
        buf["vehicle_detections"].append(vehicle_detection)
        buf["frame_ids"].append(frame_id)
        buf["crop_paths"].append(crop_path)
        buf["source_ids"].append(source_id)

        if plate_detection is not None:
            buf["plate_detections"].append(plate_detection)
        if plate_reading is not None:
            buf["plate_readings"].append(plate_reading)
        if attributes is not None:
            buf["attributes_list"].append(attributes)
        if plate_association is not None:
            buf["plate_associations"].append(plate_association)

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
        alternative_readings: List[PlateTextResult] = []
        best_evidence_frame_id: Optional[int] = None
        best_crop_path: Optional[str] = None
        has_conflicting_reads = False

        if buf["plate_detections"]:
            best_plate_det = max(buf["plate_detections"], key=lambda p: p.confidence)

        if buf["plate_readings"]:
            # Valid non-empty candidate readings
            valid_readings = [r for r in buf["plate_readings"] if r.raw_text]
            if valid_readings:
                # Pick highest confidence OCR reading
                final_plate_reading = max(valid_readings, key=lambda r: r.confidence)

                # Check for conflicting candidate reads
                unique_texts = list(dict.fromkeys(r.raw_text for r in valid_readings))
                if len(unique_texts) > 1:
                    has_conflicting_reads = True
                    needs_review = True
                    review_reasons.append(
                        f"Ambiguous OCR: conflicting reads observed across transit ({len(unique_texts)} variants: {unique_texts})"
                    )
                    seen_alts = {final_plate_reading.raw_text}
                    for cand in sorted(valid_readings, key=lambda r: r.confidence, reverse=True):
                        if cand.raw_text not in seen_alts:
                            alternative_readings.append(cand)
                            seen_alts.add(cand.raw_text)

                best_idx = None
                for i, r in enumerate(buf["plate_readings"]):
                    if r is final_plate_reading:
                        best_idx = i
                        break
                if best_idx is not None and best_idx < len(buf["frame_ids"]):
                    best_evidence_frame_id = buf["frame_ids"][best_idx]
                if best_idx is not None and best_idx < len(buf["crop_paths"]):
                    best_crop_path = buf["crop_paths"][best_idx]
            else:
                final_plate_reading = max(buf["plate_readings"], key=lambda r: r.confidence)

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
            alternative_readings=alternative_readings,
            best_evidence_frame_id=best_evidence_frame_id,
            best_crop_path=best_crop_path,
            has_conflicting_reads=has_conflicting_reads,
            attributes=final_attributes,
            stickers=[],
            permits=[],
            needs_review=needs_review,
            review_reasons=list(dict.fromkeys(review_reasons)),
        )
        return summary

    def finalize_vehicle_transit_event(self, track_id: int) -> Optional[VehicleTransitEvent]:
        """Consolidates buffered observations into a finalized, upgraded VehicleTransitEvent."""
        buf = self._buffers.pop(track_id, None)
        if not buf or buf["frames_count"] == 0:
            return None

        # Determine best vehicle detection across track
        best_vehicle = max(
            buf["vehicle_detections"],
            key=lambda d: d.confidence * np.sqrt(max(1.0, d.box.area))
        )

        review_reasons: List[str] = []
        needs_review = False
        final_plate_reading: Optional[PlateTextResult] = None
        best_plate_det: Optional[PlateDetection] = None
        alternative_readings: List[PlateTextResult] = []
        best_evidence_frame_id: Optional[int] = None
        best_crop_path: Optional[str] = None
        best_evidence_source_id: Optional[str] = None
        has_conflicting_reads = False

        if buf["plate_detections"]:
            best_plate_det = max(buf["plate_detections"], key=lambda p: p.confidence)

        if buf["plate_readings"]:
            valid_readings = [r for r in buf["plate_readings"] if r.raw_text]
            if valid_readings:
                final_plate_reading = max(valid_readings, key=lambda r: r.confidence)

                unique_texts = list(dict.fromkeys(r.raw_text for r in valid_readings))
                if len(unique_texts) > 1:
                    has_conflicting_reads = True
                    needs_review = True
                    review_reasons.append(
                        f"Conflicting plate reads observed across transit: {unique_texts}"
                    )
                    seen_alts = {final_plate_reading.raw_text}
                    for cand in sorted(valid_readings, key=lambda r: r.confidence, reverse=True):
                        if cand.raw_text not in seen_alts:
                            alternative_readings.append(cand)
                            seen_alts.add(cand.raw_text)

                best_idx = None
                for i, r in enumerate(buf["plate_readings"]):
                    if r is final_plate_reading:
                        best_idx = i
                        break
                if best_idx is not None:
                    if best_idx < len(buf["frame_ids"]):
                        best_evidence_frame_id = buf["frame_ids"][best_idx]
                    if best_idx < len(buf["crop_paths"]):
                        best_crop_path = buf["crop_paths"][best_idx]
                    if best_idx < len(buf["source_ids"]):
                        best_evidence_source_id = buf["source_ids"][best_idx]
            else:
                final_plate_reading = max(buf["plate_readings"], key=lambda r: r.confidence)

            if final_plate_reading.confidence < self._min_confidence_threshold:
                needs_review = True
                review_reasons.append(
                    f"LOW_PLATE_CONFIDENCE ({final_plate_reading.confidence:.3f} < {self._min_confidence_threshold})"
                )
            if final_plate_reading.needs_review:
                needs_review = True
                review_reasons.extend(final_plate_reading.review_reasons)
        else:
            needs_review = True
            review_reasons.append("NO_PLATE_OBSERVED: No license plate associated across vehicle transit")

        # Consolidate plate association evidence
        has_conflicting_association = False
        competing_vehicle_ids: List[int] = []
        assoc_status = AssociationStatus.CONFIDENT
        best_assoc_conf = 0.0

        if buf["plate_associations"]:
            # Pick highest confidence association
            best_assoc = max(buf["plate_associations"], key=lambda a: a.association_confidence)
            best_assoc_conf = best_assoc.association_confidence

            for assoc in buf["plate_associations"]:
                if assoc.status == AssociationStatus.AMBIGUOUS:
                    has_conflicting_association = True
                    assoc_status = AssociationStatus.AMBIGUOUS
                    needs_review = True
                    competing_vehicle_ids.extend(assoc.competing_vehicle_ids)
                    review_reasons.extend(assoc.review_reasons)
                elif assoc.status == AssociationStatus.UNASSIGNED:
                    needs_review = True
                    review_reasons.extend(assoc.review_reasons)
            competing_vehicle_ids = sorted(list(set(competing_vehicle_ids)))
        elif not buf["plate_detections"]:
            assoc_status = AssociationStatus.UNASSIGNED
        else:
            assoc_status = AssociationStatus.CONFIDENT
            best_assoc_conf = 1.0

        # Consolidate attributes
        final_attributes: Optional[VehicleAttributes] = None
        if buf["attributes_list"]:
            final_attributes = max(buf["attributes_list"], key=lambda a: a.confidence)

        event = VehicleTransitEvent(
            vehicle_track_id=track_id,
            vehicle_class=best_vehicle.vehicle_class,
            vehicle_confidence=round(best_vehicle.confidence, 4),
            first_seen_timestamp=buf["first_seen"],
            last_seen_timestamp=buf["last_seen"],
            total_frames_tracked=buf["frames_count"],
            observation_timestamps=buf["timestamps"],
            plate_reading=final_plate_reading,
            plate_text=final_plate_reading.raw_text if final_plate_reading else "",
            letters=final_plate_reading.arabic_letters or "" if final_plate_reading else "",
            digits=final_plate_reading.arabic_digits or "" if final_plate_reading else "",
            plate_confidence=round(final_plate_reading.confidence, 4) if final_plate_reading else 0.0,
            plate_association_confidence=round(best_assoc_conf, 4),
            association_status=assoc_status,
            best_vehicle_box=best_vehicle.box,
            best_plate_box=best_plate_det.box if best_plate_det else None,
            best_evidence_frame_id=best_evidence_frame_id,
            best_crop_path=best_crop_path,
            best_evidence_source_id=best_evidence_source_id,
            alternative_readings=alternative_readings,
            has_conflicting_reads=has_conflicting_reads,
            has_conflicting_association=has_conflicting_association,
            competing_vehicle_ids=competing_vehicle_ids,
            attributes=final_attributes,
            needs_review=needs_review,
            review_reasons=list(dict.fromkeys(review_reasons)),
        )
        return event

    def finalize_all_transit_events(self) -> List[VehicleTransitEvent]:
        """Finalizes and returns VehicleTransitEvent instances for all active tracks."""
        track_ids = list(self._buffers.keys())
        events: List[VehicleTransitEvent] = []
        for tid in track_ids:
            ev = self.finalize_vehicle_transit_event(tid)
            if ev is not None:
                events.append(ev)
        return events

