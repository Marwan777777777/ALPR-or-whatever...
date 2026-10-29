#!/usr/bin/env python3
"""CLI evaluation harness for Egyptian license plate detection and OCR predictions."""

import argparse
import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals for Arabic characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.core.domain import BoundingBox
from vehicle_intelligence.core.logging import setup_logger
from vehicle_intelligence.evaluation.detection_eval import (
    GroundTruthPlate,
    PlateDetectionEvaluator,
    PredictedPlate,
)
from vehicle_intelligence.evaluation.ocr_eval import (
    GroundTruthPlateText,
    PlateOCREvaluator,
    PredictedPlateText,
)


def load_ground_truth_plates_from_yolo(yolo_dir: Path, vehicle_ids: list[str]) -> list[GroundTruthPlate]:
    """Loads plate ground-truth bounding boxes from YOLO txt files."""
    gts: list[GroundTruthPlate] = []
    for vid in vehicle_ids:
        lbl_file = yolo_dir / f"{vid}.txt"
        if lbl_file.is_file():
            lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        cls_id = int(parts[0])
                        xc, yc, w, h = [float(p) for p in parts[1:]]
                        x1 = xc - w / 2.0
                        y1 = yc - h / 2.0
                        x2 = xc + w / 2.0
                        y2 = yc + h / 2.0
                        gts.append(GroundTruthPlate(
                            image_id=vid,
                            box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                            class_name="license_plate"
                        ))
                    except ValueError:
                        pass
    return gts


def main():
    parser = argparse.ArgumentParser(description="Evaluate plate detection or OCR predictions against ground truth.")
    parser.add_argument("--mode", choices=["detection", "ocr"], required=True, help="Evaluation mode ('detection' or 'ocr')")
    parser.add_argument("--predictions-json", type=str, required=True, help="Path to JSON file containing model predictions.")
    parser.add_argument("--ground-truth-json", type=str, default=None, help="Path to JSON ground truth file (optional if using YOLO dataset).")
    parser.add_argument("--iou-threshold", type=float, default=0.50, help="IoU threshold for detection matching (default: 0.50).")
    args = parser.parse_args()

    logger = setup_logger("eval_harness")
    pred_path = Path(args.predictions_json)
    if not pred_path.is_file():
        logger.error(f"Predictions file not found: {pred_path}")
        sys.exit(1)

    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    if args.mode == "detection":
        evaluator = PlateDetectionEvaluator(iou_threshold=args.iou_threshold)
        preds = [PredictedPlate(**p) for p in pred_data]

        if not args.ground_truth_json:
            logger.error("--ground-truth-json is required for detection evaluation.")
            sys.exit(1)

        gt_path = Path(args.ground_truth_json)
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        gts = [GroundTruthPlate(**g) for g in gt_data]

        report = evaluator.evaluate(gts, preds)
        logger.info(f"Evaluation complete (IoU @ {args.iou_threshold}):")
        logger.info(f"  Precision: {report.precision:.4f} | Recall: {report.recall:.4f} | F1: {report.f1_score:.4f} | AP: {report.average_precision:.4f}")
        print(report.model_dump_json(indent=2))

    elif args.mode == "ocr":
        evaluator = PlateOCREvaluator(normalize_eastern_digits=True)
        preds = [PredictedPlateText(**p) for p in pred_data]

        if not args.ground_truth_json:
            logger.error("--ground-truth-json is required for OCR evaluation.")
            sys.exit(1)

        gt_path = Path(args.ground_truth_json)
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        gts = [GroundTruthPlateText(**g) for g in gt_data]

        metrics = evaluator.evaluate(gts, preds)
        logger.info("OCR Evaluation complete:")
        logger.info(f"  Exact Match Accuracy: {metrics.exact_match_accuracy * 100:.2f}% | Mean CER: {metrics.mean_character_error_rate:.4f} | Mean NED: {metrics.mean_normalized_edit_distance:.4f}")
        print(metrics.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
