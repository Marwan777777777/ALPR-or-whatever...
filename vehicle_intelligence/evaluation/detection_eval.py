"""Plate detection evaluation contract: IoU matching, Precision, Recall, and AP."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from vehicle_intelligence.core.domain import BoundingBox


class GroundTruthPlate(BaseModel):
    """Ground-truth license plate target."""
    image_id: str
    box: BoundingBox
    class_name: str = "license_plate"
    is_difficult: bool = False


class PredictedPlate(BaseModel):
    """Model predicted license plate candidate."""
    image_id: str
    box: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_name: str = "license_plate"


class ImageDetectionMatch(BaseModel):
    """Per-image matching outcome."""
    image_id: str
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_ious: List[float] = Field(default_factory=list)


class DetectionMetricsReport(BaseModel):
    """Aggregated detection evaluation metrics."""
    iou_threshold: float
    total_ground_truth: int
    total_predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    average_precision: float
    ap_11point: float = 0.0
    mean_matched_iou: float
    per_image_matches: Dict[str, ImageDetectionMatch] = Field(default_factory=dict)


class PlateDetectionEvaluator:
    """Evaluates predicted license plates against ground-truth bounding boxes."""

    def __init__(self, iou_threshold: float = 0.50):
        if not (0.0 < iou_threshold <= 1.0):
            raise ValueError(f"IoU threshold must be in (0, 1], got {iou_threshold}")
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        ground_truths: List[GroundTruthPlate],
        predictions: List[PredictedPlate]
    ) -> DetectionMetricsReport:
        """Evaluates plate detection precision, recall, and AP at specified IoU threshold."""
        # Group by image_id
        gt_by_image: Dict[str, List[GroundTruthPlate]] = {}
        for gt in ground_truths:
            gt_by_image.setdefault(gt.image_id, []).append(gt)

        pred_by_image: Dict[str, List[PredictedPlate]] = {}
        for pred in predictions:
            pred_by_image.setdefault(pred.image_id, []).append(pred)

        all_image_ids = sorted(list(set(gt_by_image.keys()).union(set(pred_by_image.keys()))))

        total_gt = len(ground_truths)
        total_pred = len(predictions)

        # For AP calculation: collect (confidence, is_tp) pairs
        scored_detections: List[Tuple[float, bool]] = []
        per_image_matches: Dict[str, ImageDetectionMatch] = {}
        all_matched_ious: List[float] = []

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for img_id in all_image_ids:
            gts = gt_by_image.get(img_id, [])
            preds = sorted(pred_by_image.get(img_id, []), key=lambda p: p.confidence, reverse=True)

            gt_matched = [False] * len(gts)
            img_tp = 0
            img_fp = 0
            img_ious: List[float] = []

            for pred in preds:
                best_iou = 0.0
                best_gt_idx = -1

                for idx, gt in enumerate(gts):
                    if gt.class_name == pred.class_name:
                        iou = pred.box.iou(gt.box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx

                if best_iou >= self.iou_threshold and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
                    gt_matched[best_gt_idx] = True
                    img_tp += 1
                    img_ious.append(best_iou)
                    all_matched_ious.append(best_iou)
                    scored_detections.append((pred.confidence, True))
                else:
                    img_fp += 1
                    scored_detections.append((pred.confidence, False))

            img_fn = len(gts) - sum(gt_matched)

            total_tp += img_tp
            total_fp += img_fp
            total_fn += img_fn

            per_image_matches[img_id] = ImageDetectionMatch(
                image_id=img_id,
                true_positives=img_tp,
                false_positives=img_fp,
                false_negatives=img_fn,
                matched_ious=[round(x, 4) for x in img_ious],
            )

        precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        recall = (total_tp / total_gt) if total_gt > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Calculate Average Precision (Continuous all-point PR AUC and legacy 11-point)
        ap_continuous, ap_11pt = self._calculate_ap(scored_detections, total_gt)

        mean_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

        return DetectionMetricsReport(
            iou_threshold=self.iou_threshold,
            total_ground_truth=total_gt,
            total_predictions=total_pred,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            average_precision=round(ap_continuous, 4),
            ap_11point=round(ap_11pt, 4),
            mean_matched_iou=round(mean_iou, 4),
            per_image_matches=per_image_matches,
        )

    def _calculate_ap(self, scored_detections: List[Tuple[float, bool]], total_gt: int) -> Tuple[float, float]:
        if total_gt == 0 or not scored_detections:
            return 0.0, 0.0

        # Sort by confidence descending
        scored_detections.sort(key=lambda x: x[0], reverse=True)

        tp_accum = 0
        fp_accum = 0
        precisions = []
        recalls = []

        for _, is_tp in scored_detections:
            if is_tp:
                tp_accum += 1
            else:
                fp_accum += 1

            precisions.append(tp_accum / (tp_accum + fp_accum))
            recalls.append(tp_accum / total_gt)

        # 1. Continuous all-point interpolation (VOC 2010+ / COCO standard)
        mrec = np.concatenate(([0.0], recalls, [recalls[-1]]))
        mpre = np.concatenate(([1.0], precisions, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap_continuous = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))

        # 2. Legacy 11-point interpolation (PASCAL VOC 2007)
        ap_11pt = 0.0
        for r_thresh in np.linspace(0, 1.0, 11):
            valid_p = [p for p, r in zip(precisions, recalls) if r >= r_thresh]
            p_max = max(valid_p) if valid_p else 0.0
            ap_11pt += p_max / 11.0

        return ap_continuous, ap_11pt
