"""Robust multi-object vehicle tracker across sequential CCTV frames."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from vehicle_intelligence.core.domain import BoundingBox, VehicleClass, VehicleDetection
from vehicle_intelligence.core.interfaces import BaseVehicleTracker


class CentroidIoUTracker(BaseVehicleTracker):
    """Fast, lightweight multi-object tracker combining normalized centroid distance and bounding box IoU.

    Maintains independent vehicle tracks across sequential CCTV frames, handles temporary
    detection loss, multiple vehicles, and trajectory history.
    """

    def __init__(
        self,
        max_lost_frames: int = 15,
        iou_threshold: float = 0.20,
        max_centroid_distance: float = 150.0,
        max_normalized_distance: float = 1.0,
    ):
        self._max_lost_frames = max_lost_frames
        self._iou_threshold = iou_threshold
        self._max_centroid_distance = max_centroid_distance
        self._max_normalized_distance = max_normalized_distance
        self._next_id: int = 1

        # Track state: id -> dict of metadata
        # {
        #     "box": BoundingBox,
        #     "lost": int,
        #     "class": VehicleClass,
        #     "hits": int,
        #     "confidence_history": List[float],
        #     "history_boxes": List[BoundingBox],
        #     "timestamps": List[float],
        #     "frame_ids": List[int],
        # }
        self._tracks: Dict[int, Dict] = {}

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def get_active_tracks(self) -> Dict[int, Dict]:
        return self._tracks.copy()

    def get_track_history(self, track_id: int) -> Optional[Dict]:
        return self._tracks.get(track_id)

    def update(
        self,
        detections: List[VehicleDetection],
        frame_id: int,
        timestamp: float
    ) -> List[VehicleDetection]:
        if not detections:
            # Increment lost counter for all existing tracks
            expired_ids = []
            for track_id, track in self._tracks.items():
                track["lost"] += 1
                if track["lost"] > self._max_lost_frames:
                    expired_ids.append(track_id)
            for tid in expired_ids:
                del self._tracks[tid]
            return []

        # If no active tracks, initialize all detections as new independent tracks
        if not self._tracks:
            for det in detections:
                det.track_id = self._next_id
                self._tracks[self._next_id] = {
                    "box": det.box,
                    "lost": 0,
                    "class": det.vehicle_class,
                    "hits": 1,
                    "confidence_history": [det.confidence],
                    "history_boxes": [det.box],
                    "timestamps": [timestamp],
                    "frame_ids": [frame_id],
                }
                self._next_id += 1
            return detections

        # Compute cost matrix between existing tracks and incoming detections
        track_ids = list(self._tracks.keys())
        cost_matrix = np.full((len(track_ids), len(detections)), -1e5, dtype=np.float32)

        for i, tid in enumerate(track_ids):
            prev_box: BoundingBox = self._tracks[tid]["box"]
            prev_class = self._tracks[tid]["class"]

            for j, det in enumerate(detections):
                iou = prev_box.iou(det.box)
                w_avg = max(1.0, (prev_box.width + det.box.width) / 2.0)
                h_avg = max(1.0, (prev_box.height + det.box.height) / 2.0)

                # Normalized centroid distance relative to vehicle dimensions
                dx = abs(prev_box.center[0] - det.box.center[0]) / w_avg
                dy = abs(prev_box.center[1] - det.box.center[1]) / h_avg
                norm_dist = np.hypot(dx, dy)
                abs_dist = np.hypot(prev_box.center[0] - det.box.center[0], prev_box.center[1] - det.box.center[1])

                # Class consistency bonus
                class_bonus = 0.2 if (prev_class == det.vehicle_class and prev_class != VehicleClass.UNKNOWN) else 0.0

                # Score combines IoU and spatial proximity
                score = (2.0 * iou) + max(0.0, 1.0 - (norm_dist / self._max_normalized_distance)) + class_bonus
                cost_matrix[i, j] = score

        assigned_tracks = set()
        assigned_detections = set()

        # Greedy bipartite matching on highest score
        flat_indices = np.argsort(-cost_matrix, axis=None)
        for idx in flat_indices:
            i, j = np.unravel_index(idx, cost_matrix.shape)
            if i in assigned_tracks or j in assigned_detections:
                continue

            tid = track_ids[i]
            prev_box = self._tracks[tid]["box"]
            det = detections[j]
            iou = prev_box.iou(det.box)

            w_avg = max(1.0, (prev_box.width + det.box.width) / 2.0)
            h_avg = max(1.0, (prev_box.height + det.box.height) / 2.0)
            dx = abs(prev_box.center[0] - det.box.center[0]) / w_avg
            dy = abs(prev_box.center[1] - det.box.center[1]) / h_avg
            norm_dist = np.hypot(dx, dy)
            abs_dist = np.hypot(prev_box.center[0] - det.box.center[0], prev_box.center[1] - det.box.center[1])

            # Accept match if IoU is above threshold or normalized distance is within bounding limits
            is_valid_match = (
                iou >= self._iou_threshold or
                (norm_dist <= self._max_normalized_distance and abs_dist <= self._max_centroid_distance)
            )

            if is_valid_match:
                det.track_id = tid
                self._tracks[tid]["box"] = det.box
                self._tracks[tid]["lost"] = 0
                self._tracks[tid]["hits"] += 1
                self._tracks[tid]["confidence_history"].append(det.confidence)
                self._tracks[tid]["history_boxes"].append(det.box)
                self._tracks[tid]["timestamps"].append(timestamp)
                self._tracks[tid]["frame_ids"].append(frame_id)
                if det.vehicle_class != VehicleClass.UNKNOWN:
                    self._tracks[tid]["class"] = det.vehicle_class

                assigned_tracks.add(i)
                assigned_detections.add(j)

        # Increment lost counter for unassigned tracks
        expired = []
        for i, tid in enumerate(track_ids):
            if i not in assigned_tracks:
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self._max_lost_frames:
                    expired.append(tid)
        for tid in expired:
            del self._tracks[tid]

        # Register new independent tracks for unassigned detections
        for j, det in enumerate(detections):
            if j not in assigned_detections:
                det.track_id = self._next_id
                self._tracks[self._next_id] = {
                    "box": det.box,
                    "lost": 0,
                    "class": det.vehicle_class,
                    "hits": 1,
                    "confidence_history": [det.confidence],
                    "history_boxes": [det.box],
                    "timestamps": [timestamp],
                    "frame_ids": [frame_id],
                }
                self._next_id += 1

        return detections

