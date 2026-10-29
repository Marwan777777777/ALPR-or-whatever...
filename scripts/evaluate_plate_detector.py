#!/usr/bin/env python3
"""Comprehensive evaluation script for the trained Egyptian License Plate Detector.

Runs inference on validation and held-out test splits, evaluates with our repository's
canonical PlateDetectionEvaluator (IoU @ 0.50 and 0.75), saves prediction JSONs and metrics,
and renders visual comparisons (true positives and failure cases).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

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


def run_evaluation_on_split(
    model,
    split_name: str,
    images_dir: Path,
    gt_json_path: Path,
    output_dir: Path,
    conf_thresh: float = 0.25,
) -> dict:
    logger = setup_logger(f"eval_{split_name}")
    logger.info(f"Evaluating split '{split_name}' from {images_dir} against GT {gt_json_path.name}...")

    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    ground_truths = [GroundTruthPlate(**g) for g in gt_data]

    # Gather images
    valid_exts = {".jpg", ".jpeg", ".png"}
    img_files = sorted([f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])

    logger.info(f"Loaded {len(ground_truths)} ground truth boxes across {len(img_files)} images.")

    predictions: List[PredictedPlate] = []
    per_image_preds: dict[str, list[PredictedPlate]] = {}

    for img_file in img_files:
        stem = img_file.stem
        # Run inference via YOLO
        results = model.predict(
            source=str(img_file),
            conf=conf_thresh,
            device=0,
            verbose=False,
            save=False,
        )

        res = results[0]
        h, w = res.orig_shape[:2]
        img_preds: List[PredictedPlate] = []

        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for (bx1, by1, bx2, by2), score in zip(boxes, confs):
                # Normalize coordinates to [0, 1]
                nx1 = max(0.0, min(1.0, float(bx1 / w)))
                ny1 = max(0.0, min(1.0, float(by1 / h)))
                nx2 = max(0.0, min(1.0, float(bx2 / w)))
                ny2 = max(0.0, min(1.0, float(by2 / h)))

                pred = PredictedPlate(
                    image_id=stem,
                    box=BoundingBox(x1=nx1, y1=ny1, x2=nx2, y2=ny2),
                    confidence=round(float(score), 4),
                    class_name="license_plate",
                )
                predictions.append(pred)
                img_preds.append(pred)

        per_image_preds[stem] = img_preds

    # Evaluate at IoU = 0.50
    evaluator_50 = PlateDetectionEvaluator(iou_threshold=0.50)
    report_50 = evaluator_50.evaluate(ground_truths, predictions)

    # Evaluate at IoU = 0.75
    evaluator_75 = PlateDetectionEvaluator(iou_threshold=0.75)
    report_75 = evaluator_75.evaluate(ground_truths, predictions)

    # Save predictions JSON
    preds_file = output_dir / f"{split_name}_predictions.json"
    with open(preds_file, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in predictions], f, indent=2)

    # Combined summary dict
    combined_metrics = {
        "split": split_name,
        "total_images": len(img_files),
        "total_ground_truth": len(ground_truths),
        "total_predictions": len(predictions),
        "confidence_threshold": conf_thresh,
        "metrics_iou_0_50": {
            "iou_threshold": 0.50,
            "true_positives": report_50.true_positives,
            "false_positives": report_50.false_positives,
            "false_negatives": report_50.false_negatives,
            "precision": report_50.precision,
            "recall": report_50.recall,
            "f1_score": report_50.f1_score,
            "average_precision": report_50.average_precision,
            "mean_matched_iou": report_50.mean_matched_iou,
        },
        "metrics_iou_0_75": {
            "iou_threshold": 0.75,
            "true_positives": report_75.true_positives,
            "false_positives": report_75.false_positives,
            "false_negatives": report_75.false_negatives,
            "precision": report_75.precision,
            "recall": report_75.recall,
            "f1_score": report_75.f1_score,
            "average_precision": report_75.average_precision,
            "mean_matched_iou": report_75.mean_matched_iou,
        },
    }

    # Save metrics JSON
    metrics_json_file = output_dir / f"{split_name}_detection_metrics.json"
    with open(metrics_json_file, "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2)

    # Save Markdown report
    md_report = f"""# License Plate Detection Evaluation: `{split_name.upper()}` Split

- **Dataset Size:** {len(img_files)} images, {len(ground_truths)} ground-truth plates
- **Model Predictions:** {len(predictions)} candidate detections (Confidence >= {conf_thresh})

| Metric | IoU @ 0.50 | IoU @ 0.75 |
|---|---|---|
| **True Positives (TP)** | {report_50.true_positives} | {report_75.true_positives} |
| **False Positives (FP)** | {report_50.false_positives} | {report_75.false_positives} |
| **False Negatives (FN)** | {report_50.false_negatives} | {report_75.false_negatives} |
| **Precision** | {report_50.precision:.4f} | {report_75.precision:.4f} |
| **Recall** | {report_50.recall:.4f} | {report_75.recall:.4f} |
| **F1-Score** | {report_50.f1_score:.4f} | {report_75.f1_score:.4f} |
| **Average Precision (AP)** | {report_50.average_precision:.4f} | {report_75.average_precision:.4f} |
| **Mean Matched IoU** | {report_50.mean_matched_iou:.4f} | {report_75.mean_matched_iou:.4f} |
"""
    md_file = output_dir / f"{split_name}_detection_metrics.md"
    md_file.write_text(md_report, encoding="utf-8")

    logger.info(
        f"Split '{split_name}' Results (IoU@0.50): Precision={report_50.precision:.4f}, "
        f"Recall={report_50.recall:.4f}, F1={report_50.f1_score:.4f}, AP={report_50.average_precision:.4f}"
    )

    gt_by_image_list: dict[str, list[GroundTruthPlate]] = {}
    for g in ground_truths:
        gt_by_image_list.setdefault(g.image_id, []).append(g)

    return {
        "metrics": combined_metrics,
        "report_50": report_50,
        "report_75": report_75,
        "per_image_preds": per_image_preds,
        "img_files": {f.stem: f for f in img_files},
        "gt_by_image": gt_by_image_list,
    }


def render_visual_sample(
    img_path: Path,
    gt_boxes: list[BoundingBox],
    pred_boxes: list[PredictedPlate],
    save_path: Path,
    title: str = "",
) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    canvas = img.copy()

    # Draw Ground Truths in GREEN
    for idx, gt_box in enumerate(gt_boxes, start=1):
        gx1, gy1 = int(round(gt_box.x1 * w)), int(round(gt_box.y1 * h))
        gx2, gy2 = int(round(gt_box.x2 * w)), int(round(gt_box.y2 * h))
        cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        lbl = f"GT Plate {idx}" if len(gt_boxes) > 1 else "GT Plate"
        cv2.putText(canvas, lbl, (gx1, max(15, gy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Draw Predictions in CYAN / BLUE
    for p in pred_boxes:
        px1, py1 = int(round(p.box.x1 * w)), int(round(p.box.y1 * h))
        px2, py2 = int(round(p.box.x2 * w)), int(round(p.box.y2 * h))
        color = (255, 200, 0)
        cv2.rectangle(canvas, (px1, py1), (px2, py2), color, 2)
        label_text = f"Pred: {p.confidence:.2f}"
        cv2.putText(canvas, label_text, (px1, min(h - 5, py2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if title:
        cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas)


def main():
    parser = argparse.ArgumentParser(description="Evaluate plate detector model.")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt weights")
    parser.add_argument("--dataset-dir", type=str, default="artifacts/ealpr_yolo_dataset", help="YOLO dataset dir")
    parser.add_argument("--splits-dir", type=str, default="artifacts/splits", help="Splits directory")
    parser.add_argument("--output-dir", type=str, default="artifacts/evaluation", help="Evaluation output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = out_dir / "visual_samples"
    visual_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.weights)

    data_dir = Path(args.dataset_dir)
    splits_dir = Path(args.splits_dir)

    # 1. Validation split evaluation
    val_res = run_evaluation_on_split(
        model=model,
        split_name="val",
        images_dir=data_dir / "images" / "val",
        gt_json_path=splits_dir / "val_ground_truth_plates.json",
        output_dir=out_dir,
        conf_thresh=args.conf,
    )

    # 2. Held-out test split evaluation
    test_res = run_evaluation_on_split(
        model=model,
        split_name="test",
        images_dir=data_dir / "images" / "test",
        gt_json_path=splits_dir / "test_ground_truth_plates.json",
        output_dir=out_dir,
        conf_thresh=args.conf,
    )

    # 3. Generate visual samples on test split (True Positives and Edge Cases)
    test_matches = test_res["report_50"].per_image_matches
    test_files = test_res["img_files"]
    gt_map = test_res["gt_by_image"]
    pred_map = test_res["per_image_preds"]

    # Pick 2 perfect TPs
    tps = [img_id for img_id, m in test_matches.items() if m.true_positives >= 1 and m.false_positives == 0]
    for i, stem in enumerate(tps[:2], start=1):
        if stem in test_files:
            render_visual_sample(
                img_path=test_files[stem],
                gt_boxes=[g.box for g in gt_map.get(stem, [])],
                pred_boxes=pred_map.get(stem, []),
                save_path=visual_dir / f"test_sample_tp_{i}_{stem}.jpg",
                title=f"Test Sample {i} [TP Match, IoU={test_matches[stem].matched_ious[0] if test_matches[stem].matched_ious else 0:.2f}]",
            )

    # Pick false negative or false positive or lowest matched IoU
    fns = [img_id for img_id, m in test_matches.items() if m.false_negatives >= 1]
    if fns:
        stem = fns[0]
        if stem in test_files:
            render_visual_sample(
                img_path=test_files[stem],
                gt_boxes=[g.box for g in gt_map.get(stem, [])],
                pred_boxes=pred_map.get(stem, []),
                save_path=visual_dir / f"test_sample_failure_fn_{stem}.jpg",
                title=f"Test Hard Case [False Negative: Missing Detection]",
            )

    fps = [img_id for img_id, m in test_matches.items() if m.false_positives >= 1]
    if fps:
        stem = fps[0]
        if stem in test_files:
            render_visual_sample(
                img_path=test_files[stem],
                gt_boxes=[g.box for g in gt_map.get(stem, [])],
                pred_boxes=pred_map.get(stem, []),
                save_path=visual_dir / f"test_sample_fp_or_low_iou_{stem}.jpg",
                title=f"Test Edge Case [Extra Box or Low IoU]",
            )

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Validation Split (IoU@0.50): AP={val_res['metrics']['metrics_iou_0_50']['average_precision']:.4f}, F1={val_res['metrics']['metrics_iou_0_50']['f1_score']:.4f}")
    print(f"Held-out Test (IoU@0.50):    AP={test_res['metrics']['metrics_iou_0_50']['average_precision']:.4f}, F1={test_res['metrics']['metrics_iou_0_50']['f1_score']:.4f}")
    print(f"Held-out Test (IoU@0.75):    AP={test_res['metrics']['metrics_iou_0_75']['average_precision']:.4f}, F1={test_res['metrics']['metrics_iou_0_75']['f1_score']:.4f}")
    print(f"Visual QA samples saved to:  {visual_dir}")
    print("====================================================\n")


if __name__ == "__main__":
    main()
