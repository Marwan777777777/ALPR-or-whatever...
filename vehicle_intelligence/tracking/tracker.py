"""Robust multi-object vehicle tracker across sequential CCTV frames."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from vehicle_intelligence.core.domain import BoundingBox, VehicleDetection
from vehicle_intelligence.core.interfaces import BaseVehicleTracker


class CentroidIoUTracker(BaseVehicleTracker):
    """Fast, lightweight multi-object tracker combining centroid distance and bounding box IoU.

    Ideal for fixed gate CCTV where vehicles move linearly through the field of view.
    """

    def __init__(
        self,
        max_lost_frames: int = 15,
        iou_threshold: float = 0.25,
        max_centroid_distance: float = 120.0
    ):
        self._max_lost_frames = max_lost_frames
        self._iou_threshold = iou_threshold
        self._max_centroid_distance = max_centroid_distance
        self._next_id: int = 1

        # Track state: id -> dict of {"box": BoundingBox, "lost": int, "class": VehicleClass}
        self._tracks: Dict[int, Dict] = {}

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

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

        # If no active tracks, initialize all detections as new tracks
        if not self._tracks:
            for det in detections:
                det.track_id = self._next_id
                self._tracks[self._next_id] = {
                    "box": det.box,
                    "lost": 0,
                    "class": det.vehicle_class,
                }
                self._next_id += 1
            return detections

        # Compute cost matrix between existing tracks and incoming detections
        track_ids = list(self._tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            prev_box: BoundingBox = self._tracks[tid]["box"]
            for j, det in enumerate(detections):
                iou = prev_box.iou(det.box)
                # Distance between centroids
                dist = np.hypot(prev_box.center[0] - det.box.center[0], prev_box.center[1] - det.box.center[1])
                # Combined metric: high IoU and low distance yields high score
                score = iou - (dist / self._max_centroid_distance) * 0.3
                cost_matrix[i, j] = score

        assigned_tracks = set()
        assigned_detections = set()

        # Greedy bipartite matching
        flat_indices = np.argsort(-cost_matrix, axis=None)
        for idx in flat_indices:
            i, j = np.unravel_index(idx, cost_matrix.shape)
            if i in assigned_tracks or j in assigned_detections:
                continue

            tid = track_ids[i]
            prev_box = self._tracks[tid]["box"]
            det = detections[j]
            iou = prev_box.iou(det.box)
            dist = np.hypot(prev_box.center[0] - det.box.center[0], prev_box.center[1] - det.box.center[1])

            if iou >= self._iou_threshold or dist <= self._max_centroid_distance:
                det.track_id = tid
                self._tracks[tid]["box"] = det.box
                self._tracks[tid]["lost"] = 0
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

        # Register new tracks for unassigned detections
        for j, det in enumerate(detections):
            if j not in assigned_detections:
                det.track_id = self._next_id
                self._tracks[self._next_id] = {
                    "box": det.box,
                    "lost": 0,
                    "class": det.vehicle_class,
                }
                self._next_id += 1

        return detections
