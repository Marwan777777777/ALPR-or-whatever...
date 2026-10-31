"""Confidence-aware plate-to-vehicle association layer.

Associates detected Egyptian license plates with vehicle tracks based on:
1. Spatial containment (IoP: Intersection over Plate Area)
2. Relative geometry (vertical positioning and horizontal alignment)
3. Physical scale consistency (plate area to vehicle area ratio)
4. Temporal consistency across consecutive frames
5. Ambiguity detection when multiple vehicles are close or overlapping
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from vehicle_intelligence.core.domain import (
    AssociationStatus,
    BoundingBox,
    PlateDetection,
    PlateVehicleAssociation,
    VehicleDetection,
)
from vehicle_intelligence.core.interfaces import BasePlateVehicleAssociator


class PlateVehicleAssociator(BasePlateVehicleAssociator):
    """Associates license plate detections with corresponding vehicle candidates.

    Never forces an association. Preserves unassigned plates and explicitly flags
    ambiguous associations when multiple vehicles compete for the same plate.
    """

    def __init__(
        self,
        min_containment_iop: float = 0.50,
        confident_containment_iop: float = 0.80,
        min_association_score: float = 0.40,
        confident_association_score: float = 0.65,
        ambiguity_margin: float = 0.15,
        min_scale_ratio: float = 0.0005,
        max_scale_ratio: float = 0.35,
    ):
        self._min_containment_iop = min_containment_iop
        self._confident_containment_iop = confident_containment_iop
        self._min_association_score = min_association_score
        self._confident_association_score = confident_association_score
        self._ambiguity_margin = ambiguity_margin
        self._min_scale_ratio = min_scale_ratio
        self._max_scale_ratio = max_scale_ratio

        # Temporal memory: track previous associations for continuity
        # plate_key -> (vehicle_track_id, timestamp)
        self._temporal_memory: Dict[str, Tuple[int, float]] = {}

    def compute_candidate_score(
        self,
        plate: PlateDetection,
        vehicle: VehicleDetection
    ) -> Tuple[float, float, Tuple[float, float], float, List[str]]:
        """Computes geometric match score between a plate and a vehicle.

        Returns:
            (total_score, iop, relative_offset, scale_ratio, flags)
        """
        pb: BoundingBox = plate.box
        vb: BoundingBox = vehicle.box

        flags: List[str] = []

        # 1. Spatial Containment (IoP)
        iop = pb.intersection_over_self(vb)
        if iop < self._min_containment_iop:
            return 0.0, iop, (0.0, 0.0), 0.0, ["LOW_CONTAINMENT"]

        # 2. Scale consistency
        p_area = pb.area
        v_area = max(1.0, vb.area)
        scale_ratio = p_area / v_area

        scale_score = 1.0
        if scale_ratio < self._min_scale_ratio:
            scale_score = 0.4
            flags.append("UNUSUALLY_SMALL_PLATE_RATIO")
        elif scale_ratio > self._max_scale_ratio:
            scale_score = 0.3
            flags.append("UNUSUALLY_LARGE_PLATE_RATIO")

        # 3. Relative spatial geometry
        px_c, py_c = pb.center
        vx_c, vy_c = vb.center
        vw = max(1.0, vb.width)
        vh = max(1.0, vb.height)

        # Horizontal alignment: plate center should be reasonably aligned with vehicle center
        dx = abs(px_c - vx_c) / vw
        horiz_score = max(0.0, 1.0 - (dx / 0.50))

        # Vertical placement: plate is usually on lower portion of front/rear
        y_rel = (py_c - vb.y1) / vh
        if 0.25 <= y_rel <= 0.98:
            vert_score = 1.0
        elif 0.15 <= y_rel < 0.25:
            vert_score = 0.6
            flags.append("HIGH_PLATE_PLACEMENT")
        else:
            vert_score = 0.3
            flags.append("ABNORMAL_VERTICAL_PLACEMENT")

        # Composite score
        # 50% containment, 20% horizontal, 15% vertical, 15% vehicle detector confidence
        total_score = (
            (0.50 * iop) +
            (0.20 * horiz_score) +
            (0.15 * vert_score * scale_score) +
            (0.15 * vehicle.confidence)
        )
        total_score = float(np.clip(total_score, 0.0, 1.0))

        return total_score, round(iop, 4), (round(dx, 4), round(y_rel, 4)), round(scale_ratio, 6), flags

    def associate(
        self,
        plate_detections: List[PlateDetection],
        vehicle_detections: List[VehicleDetection],
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> List[PlateVehicleAssociation]:
        """Performs confidence-aware plate-to-vehicle association."""
        if not plate_detections:
            return []

        associations: List[PlateVehicleAssociation] = []

        for p_idx, plate in enumerate(plate_detections):
            # If no vehicles detected, plate is unassigned
            if not vehicle_detections:
                associations.append(
                    PlateVehicleAssociation(
                        plate_box=plate.box,
                        plate_confidence=plate.confidence,
                        vehicle_track_id=None,
                        vehicle_box=None,
                        vehicle_class=None,
                        association_confidence=0.0,
                        status=AssociationStatus.UNASSIGNED,
                        competing_vehicle_ids=[],
                        containment_ratio=0.0,
                        relative_center_offset=(0.0, 0.0),
                        scale_ratio=0.0,
                        needs_review=True,
                        review_reasons=["UNASSIGNED_PLATE: No vehicle detected in frame"],
                    )
                )
                continue

            # Evaluate against all candidate vehicles
            candidates = []
            for v_idx, veh in enumerate(vehicle_detections):
                score, iop, rel_offset, scale_r, flags = self.compute_candidate_score(plate, veh)
                if score >= self._min_association_score:
                    candidates.append({
                        "vehicle": veh,
                        "score": score,
                        "iop": iop,
                        "rel_offset": rel_offset,
                        "scale_r": scale_r,
                        "flags": flags,
                        "track_id": veh.track_id if veh.track_id is not None else v_idx,
                    })

            # Case A: No suitable candidate found
            if not candidates:
                associations.append(
                    PlateVehicleAssociation(
                        plate_box=plate.box,
                        plate_confidence=plate.confidence,
                        vehicle_track_id=None,
                        vehicle_box=None,
                        vehicle_class=None,
                        association_confidence=0.0,
                        status=AssociationStatus.UNASSIGNED,
                        competing_vehicle_ids=[],
                        containment_ratio=0.0,
                        relative_center_offset=(0.0, 0.0),
                        scale_ratio=0.0,
                        needs_review=True,
                        review_reasons=["UNASSIGNED_PLATE: No enclosing vehicle matched association criteria"],
                    )
                )
                continue

            # Sort candidate matches by score descending
            candidates.sort(key=lambda c: c["score"], reverse=True)
            top = candidates[0]
            top_veh = top["vehicle"]

            # Case B: Multiple candidates competing (Ambiguity Detection)
            if len(candidates) > 1:
                runner_up = candidates[1]
                score_delta = top["score"] - runner_up["score"]
                both_contained = (top["iop"] >= 0.70 and runner_up["iop"] >= 0.50)

                if score_delta < self._ambiguity_margin or both_contained:
                    # Ambiguous case: two vehicles overlap or are very close
                    competing_ids = [c["track_id"] for c in candidates[:3]]
                    review_reasons = [
                        f"AMBIGUOUS_ASSOCIATION: Competing vehicle tracks {competing_ids} with score delta {score_delta:.3f} < {self._ambiguity_margin}"
                    ]
                    review_reasons.extend(top["flags"])

                    associations.append(
                        PlateVehicleAssociation(
                            plate_box=plate.box,
                            plate_confidence=plate.confidence,
                            vehicle_track_id=top["track_id"],
                            vehicle_box=top_veh.box,
                            vehicle_class=top_veh.vehicle_class,
                            association_confidence=round(top["score"], 4),
                            status=AssociationStatus.AMBIGUOUS,
                            competing_vehicle_ids=competing_ids,
                            containment_ratio=top["iop"],
                            relative_center_offset=top["rel_offset"],
                            scale_ratio=top["scale_r"],
                            needs_review=True,
                            review_reasons=review_reasons,
                        )
                    )
                    continue

            # Case C: Single clear candidate
            review_reasons = list(top["flags"])
            needs_review = top["score"] < self._confident_association_score or len(review_reasons) > 0

            if top["score"] < self._confident_association_score:
                review_reasons.append(
                    f"LOW_ASSOCIATION_CONFIDENCE ({top['score']:.3f} < {self._confident_association_score})"
                )

            associations.append(
                PlateVehicleAssociation(
                    plate_box=plate.box,
                    plate_confidence=plate.confidence,
                    vehicle_track_id=top["track_id"],
                    vehicle_box=top_veh.box,
                    vehicle_class=top_veh.vehicle_class,
                    association_confidence=round(top["score"], 4),
                    status=AssociationStatus.CONFIDENT,
                    competing_vehicle_ids=[],
                    containment_ratio=top["iop"],
                    relative_center_offset=top["rel_offset"],
                    scale_ratio=top["scale_r"],
                    needs_review=needs_review,
                    review_reasons=review_reasons,
                )
            )

        return associations
