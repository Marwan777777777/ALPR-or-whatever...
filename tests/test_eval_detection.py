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
