#!/usr/bin/env python3
"""Generates a complete, mathematically rigorous 27-Class Confusion Matrix for Egyptian Plate OCR.

Defines one-to-one spatial IoU matching (IoU >= 0.50) between predicted and ground-truth boxes.
Separates:
- True classification substitutions (matched boxes with wrong class)
- Localization misses (unmatched GT -> Background/FN)
- Hallucinations (unmatched Pred -> Background/FP)

Computes per-class support, precision, recall, F1, and ranks the most frequent confusion pairs.
Saves machine-readable JSON/CSV and human-readable reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.core.domain import BoundingBox
from vehicle_intelligence.data.audit import EALPR_CHARACTER_CLASSES
from scripts.evaluate_character_model import (
    parse_character_boxes_from_yolo_result,
    parse_gt_character_boxes_from_file,
)


def compute_character_confusion_matrix(
    char_model: YOLO,
    char_dataset_dir: Path,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
) -> dict:
    test_img_dir = char_dataset_dir / "images" / "test"
    test_lbl_dir = char_dataset_dir / "labels" / "test"

    img_files = sorted(list(test_img_dir.glob("*.png")))
    num_classes = 27
    # Matrix dimensions: 28 x 28 (0..26 classes, 27 = Background)
    # Rows = Ground Truth, Columns = Prediction
    BG_INDEX = num_classes
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    confusion_pairs = defaultdict(int)
    localization_misses = defaultdict(int)
    hallucinations = defaultdict(int)

    total_gt_characters = 0
    total_pred_characters = 0

    for img_path in img_files:
        stem = img_path.stem
        lbl_path = test_lbl_dir / f"{stem}.txt"

        gt_boxes = parse_gt_character_boxes_from_file(lbl_path)
        total_gt_characters += len(gt_boxes)

        res = char_model.predict(source=str(img_path), conf=conf_thresh, imgsz=320, verbose=False)[0]
        pred_boxes = parse_character_boxes_from_yolo_result(res, conf_thresh=conf_thresh)
        total_pred_characters += len(pred_boxes)

        # Convert to BoundingBox objects for IoU calculation
        gt_bb = [
            BoundingBox(
                x1=b.x_center - b.width / 2.0,
                y1=b.y_center - b.height / 2.0,
                x2=b.x_center + b.width / 2.0,
                y2=b.y_center + b.height / 2.0,
            )
            for b in gt_boxes
        ]
        pred_bb = [
            BoundingBox(
                x1=b.x_center - b.width / 2.0,
                y1=b.y_center - b.height / 2.0,
                x2=b.x_center + b.width / 2.0,
                y2=b.y_center + b.height / 2.0,
            )
            for b in pred_boxes
        ]

        # Greedy matching based on IoU
        matched_gt = set()
        matched_pred = set()

        # Compute IoU matrix
        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=float)
        for i, gbox in enumerate(gt_bb):
            for j, pbox in enumerate(pred_bb):
                iou_matrix[i, j] = gbox.iou(pbox)

        # Match pairs in descending order of IoU
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = np.max(iou_matrix)
            if max_iou < iou_thresh:
                break
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            gt_cls = gt_boxes[i].class_id
            pred_cls = pred_boxes[j].class_id

            matrix[gt_cls, pred_cls] += 1
            if gt_cls != pred_cls:
                c_gt_name = EALPR_CHARACTER_CLASSES[gt_cls]
                c_pred_name = EALPR_CHARACTER_CLASSES[pred_cls]
                confusion_pairs[f"{c_gt_name} (cls {gt_cls}) -> {c_pred_name} (cls {pred_cls})"] += 1

            matched_gt.add(i)
            matched_pred.add(j)
            # Invalidate matched row and column
            iou_matrix[i, :] = -1.0
            iou_matrix[:, j] = -1.0

        # Unmatched GT -> Missed Detection (FN)
        for i, gbox in enumerate(gt_boxes):
            if i not in matched_gt:
                matrix[gbox.class_id, BG_INDEX] += 1
                c_gt_name = EALPR_CHARACTER_CLASSES[gbox.class_id]
                localization_misses[f"{c_gt_name} (cls {gbox.class_id})"] += 1

        # Unmatched Pred -> False Alarm / Hallucination (FP)
        for j, pbox in enumerate(pred_boxes):
            if j not in matched_pred:
                matrix[BG_INDEX, pbox.class_id] += 1
                c_pred_name = EALPR_CHARACTER_CLASSES[pbox.class_id]
                hallucinations[f"{c_pred_name} (cls {pbox.class_id})"] += 1

    # Compute Per-Class Metrics
    per_class_metrics = {}
    for c in range(num_classes):
        c_name = EALPR_CHARACTER_CLASSES[c]
        support = int(np.sum(matrix[c, :]))  # all GT occurrences of class c
        tp = int(matrix[c, c])
        fn = int(np.sum(matrix[c, :]) - tp)
        fp = int(np.sum(matrix[:, c]) - tp)

        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0

        per_class_metrics[c_name] = {
            "class_id": c,
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        }

    total_tp = sum(m["true_positives"] for m in per_class_metrics.values())
    total_fp = sum(m["false_positives"] for m in per_class_metrics.values())
    total_fn = sum(m["false_negatives"] for m in per_class_metrics.values())
    macro_f1 = round(float(np.mean([m["f1_score"] for m in per_class_metrics.values() if m["support"] > 0])), 4)

    return {
        "total_test_plates": len(img_files),
        "total_gt_characters": total_gt_characters,
        "total_pred_characters": total_pred_characters,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "overall_character_precision": round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0.0,
        "overall_character_recall": round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) > 0 else 0.0,
        "macro_f1": macro_f1,
        "per_class_metrics": per_class_metrics,
        "top_confusion_pairs": sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True),
        "top_localization_misses_fn": sorted(localization_misses.items(), key=lambda x: x[1], reverse=True),
        "top_hallucinations_fp": sorted(hallucinations.items(), key=lambda x: x[1], reverse=True),
        "raw_matrix": matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate 27-Class Confusion Matrix for OCR.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt"),
    )
    parser.add_argument(
        "--char-dataset",
        type=Path,
        default=Path("artifacts/ealpr_characters_yolo_dataset"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/evaluation_ocr/confusion_matrix.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/evaluation_ocr/confusion_matrix.csv"),
    )
    args = parser.parse_args()

    print(f"Loading character model from {args.weights}...")
    char_model = YOLO(str(args.weights))

    results = compute_character_confusion_matrix(char_model, args.char_dataset)

    # Save JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save CSV
    labels = [EALPR_CHARACTER_CLASSES[i] for i in range(27)] + ["Background"]
    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["GT \\ Pred"] + labels)
        for i, row in enumerate(results["raw_matrix"]):
            writer.writerow([labels[i]] + row)

    print("\n" + "=" * 70)
    print("27-CLASS EGYPTIAN PLATE OCR CONFUSION MATRIX SUMMARY")
    print("=" * 70)
    print(f"Total Test Plates:       {results['total_test_plates']}")
    print(f"Total GT Characters:     {results['total_gt_characters']}")
    print(f"Overall Character Recall:    {results['overall_character_recall']*100:.2f}%")
    print(f"Overall Character Precision: {results['overall_character_precision']*100:.2f}%")
    print(f"Macro F1 Score:          {results['macro_f1']:.4f}")

    print("\nTop Character Confusion Substitution Pairs (GT -> PRED):")
    if results["top_confusion_pairs"]:
        for pair, count in results["top_confusion_pairs"][:10]:
            print(f"  {pair}: {count}")
    else:
        print("  None (0 substitutions)")

    print("\nTop Localization Misses (GT -> FN/Unmatched):")
    for item, count in results["top_localization_misses_fn"][:5]:
        print(f"  {item}: {count}")

    print("\nTop Hallucinations (FP/Unmatched Pred):")
    for item, count in results["top_hallucinations_fp"][:5]:
        print(f"  {item}: {count}")

    print("\nPer-Class Metrics (Worst 5 by F1 Score):")
    sorted_classes = sorted(
        results["per_class_metrics"].items(),
        key=lambda x: x[1]["f1_score"] if x[1]["support"] > 0 else 1.0,
    )
    for c_name, m in sorted_classes[:5]:
        print(f"  '{c_name}' (cls {m['class_id']:2d}): Support={m['support']:3d}, Prec={m['precision']:.3f}, Rec={m['recall']:.3f}, F1={m['f1_score']:.3f}")

    print("\n" + "=" * 70)
    print(f"Saved JSON report to: {args.output_json}")
    print(f"Saved CSV matrix to:  {args.output_csv}\n")


if __name__ == "__main__":
    main()
