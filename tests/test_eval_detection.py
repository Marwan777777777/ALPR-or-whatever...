"""Tests for plate detection evaluation metrics and IoU matching."""

import pytest
from vehicle_intelligence.core.domain import BoundingBox
from vehicle_intelligence.evaluation.detection_eval import (
    GroundTruthPlate,
    PlateDetectionEvaluator,
    PredictedPlate,
)


def test_evaluator_perfect_match():
    box = BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=30.0)
    gts = [GroundTruthPlate(image_id="img_1", box=box)]
    preds = [PredictedPlate(image_id="img_1", box=box, confidence=0.95)]

    evaluator = PlateDetectionEvaluator(iou_threshold=0.50)
    report = evaluator.evaluate(gts, preds)

    assert report.total_ground_truth == 1
    assert report.total_predictions == 1
    assert report.true_positives == 1
    assert report.false_positives == 0
    assert report.false_negatives == 0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1_score == 1.0
    assert report.average_precision == 1.0
    assert report.mean_matched_iou == 1.0


def test_evaluator_false_positives_and_negatives():
    gt_box = BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=30.0)
    pred_box_far = BoundingBox(x1=100.0, y1=100.0, x2=150.0, y2=130.0)

    gts = [GroundTruthPlate(image_id="img_1", box=gt_box)]
    preds = [PredictedPlate(image_id="img_1", box=pred_box_far, confidence=0.8)]

    evaluator = PlateDetectionEvaluator(iou_threshold=0.50)
    report = evaluator.evaluate(gts, preds)

    assert report.true_positives == 0
    assert report.false_positives == 1
    assert report.false_negatives == 1
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.f1_score == 0.0


def test_evaluator_empty_predictions():
    gt_box = BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=30.0)
    gts = [GroundTruthPlate(image_id="img_1", box=gt_box)]
    preds = []

    evaluator = PlateDetectionEvaluator(iou_threshold=0.50)
    report = evaluator.evaluate(gts, preds)

    assert report.true_positives == 0
    assert report.false_positives == 0
    assert report.false_negatives == 1
    assert report.recall == 0.0
    assert report.precision == 0.0


def test_evaluator_invalid_iou_threshold():
    with pytest.raises(ValueError):
        PlateDetectionEvaluator(iou_threshold=0.0)

    with pytest.raises(ValueError):
        PlateDetectionEvaluator(iou_threshold=1.5)


def test_evaluator_reproduces_known_ap_value():
    """Validates that all-point interpolated AP matches the exact theoretical integral.

    Theoretical setup: 3 ground truth plates across 3 images.
    5 predictions in order of descending confidence:
      1. TP (conf 0.9) -> P = 1/1 = 1.0, R = 1/3
      2. FP (conf 0.8) -> P = 1/2 = 0.5, R = 1/3
      3. TP (conf 0.7) -> P = 2/3 ≈ 0.6667, R = 2/3
      4. FP (conf 0.6) -> P = 2/4 = 0.5, R = 2/3
      5. TP (conf 0.5) -> P = 3/5 = 0.6, R = 3/3 = 1.0

    PR segments:
      [0.0, 1/3]: P_max = 1.0     -> Area = (1/3) * 1.0 = 1/3
      [1/3, 2/3]: P_max = 2/3     -> Area = (1/3) * (2/3) = 2/9
      [2/3, 1.0]: P_max = 0.6     -> Area = (1/3) * (3/5) = 1/5
    Theoretical AP = 1/3 + 2/9 + 1/5 = 34/45 ≈ 0.7556.
    """
    box_gt = BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=30.0)
    box_fp = BoundingBox(x1=100.0, y1=100.0, x2=150.0, y2=130.0)

    gts = [
        GroundTruthPlate(image_id="img_1", box=box_gt),
        GroundTruthPlate(image_id="img_2", box=box_gt),
        GroundTruthPlate(image_id="img_3", box=box_gt),
    ]

    preds = [
        PredictedPlate(image_id="img_1", box=box_gt, confidence=0.9),  # TP
        PredictedPlate(image_id="img_1", box=box_fp, confidence=0.8),  # FP
        PredictedPlate(image_id="img_2", box=box_gt, confidence=0.7),  # TP
        PredictedPlate(image_id="img_2", box=box_fp, confidence=0.6),  # FP
        PredictedPlate(image_id="img_3", box=box_gt, confidence=0.5),  # TP
    ]

    evaluator = PlateDetectionEvaluator(iou_threshold=0.50)
    report = evaluator.evaluate(gts, preds)

    assert report.total_ground_truth == 3
    assert report.total_predictions == 5
    assert report.true_positives == 3
    assert report.false_positives == 2
    assert report.false_negatives == 0
    assert report.precision == round(3 / 5, 4)
    assert report.recall == 1.0

    # Verify exact match with 34 / 45 = 0.7556
    expected_ap = round(34 / 45, 4)
    assert report.average_precision == expected_ap

