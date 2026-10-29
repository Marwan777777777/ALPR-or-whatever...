#!/usr/bin/env python3
"""Evaluate ALPR Pipeline on Labeled Gate Frames with Environmental Slicing (Mission 6 Part E).
Evaluates full detection and OCR pipeline on gate frames, splitting metrics by lighting condition
(day / night / glare) when a condition tags file is provided.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.core.domain import BoundingBox
from vehicle_intelligence.pipeline.alpr_pipeline import EndToEndALPRPipeline
from scripts.audit_end_to_end_and_ocr import levenshtein_distance


def evaluate_gate_dataset(
    dataset_dir: Path,
    tags_file: Optional[Path] = None,
    output_dir: Path = Path("artifacts/evaluation_gate"),
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory under {dataset_dir}")

    # Load condition tags if available
    # Supports JSON: {"frame_stem": "day|night|glare"} or CSV: stem,condition
    conditions: Dict[str, str] = {}
    if tags_file and tags_file.is_file():
        if tags_file.suffix.lower() == ".json":
            conditions = json.loads(tags_file.read_text(encoding="utf-8"))
        else:
            with open(tags_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        conditions[row[0].strip()] = row[1].strip().lower()

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted([f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
    print(f"Loaded {len(img_files)} gate frames from {images_dir}...")

    pipeline = EndToEndALPRPipeline()

    # Track metrics overall and per condition
    stats = defaultdict(lambda: {
        "total_frames": 0,
        "total_gt_plates": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "exact_matches": 0,
        "total_edit_distance": 0,
        "total_gt_characters": 0,
    })

    for f in img_files:
        stem = f.stem
        cond = conditions.get(stem, "unspecified")

        im = cv2.imread(str(f))
        if im is None:
            continue
        h, w = im.shape[:2]

        # Read GT if label file exists
        gt_boxes = []
        gt_texts = []
        lbl_file = labels_dir / f"{stem}.txt"
        if lbl_file.is_file():
            for line in lbl_file.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    xc, yc, bw, bh = [float(p) for p in parts[1:5]]
                    gt_boxes.append(BoundingBox(
                        x1=(xc - bw / 2.0) * w,
                        y1=(yc - bh / 2.0) * h,
                        x2=(xc + bw / 2.0) * w,
                        y2=(yc + bh / 2.0) * h,
                    ))
                    # If label includes ground truth text e.g. after coordinates
                    txt = " ".join(parts[5:]) if len(parts) > 5 else ""
                    gt_texts.append(txt.replace(" ", ""))

        res = pipeline.read_plate_image(im, source_id=f.name)

        # Match predictions to GT boxes
        matched_gt = [False] * len(gt_boxes)
        matched_pred = [False] * len(res)

        for p_idx, r in enumerate(res):
            best_iou = 0.0
            best_g = -1
            for g_idx, g_box in enumerate(gt_boxes):
                if not matched_gt[g_idx]:
                    iou = r.bounding_box.iou(g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_g = g_idx
            if best_iou >= 0.50 and best_g >= 0:
                matched_gt[best_g] = True
                matched_pred[p_idx] = True

                # Compare OCR text if GT text is present
                gt_txt = gt_texts[best_g]
                pred_txt = r.plate_text.replace(" ", "")
                if gt_txt:
                    dist = levenshtein_distance(gt_txt, pred_txt)
                    is_exact = (dist == 0)
                    for target in ("overall", cond):
                        stats[target]["total_gt_characters"] += len(gt_txt)
                        stats[target]["total_edit_distance"] += dist
                        if is_exact:
                            stats[target]["exact_matches"] += 1

        tp = sum(matched_gt)
        fn = len(gt_boxes) - tp
        fp = len(res) - sum(matched_pred)

        for target in ("overall", cond):
            stats[target]["total_frames"] += 1
            stats[target]["total_gt_plates"] += len(gt_boxes)
            stats[target]["true_positives"] += tp
            stats[target]["false_positives"] += fp
            stats[target]["false_negatives"] += fn

    # Compile summarized metrics
    report: Dict[str, dict] = {}
    for group, s in stats.items():
        tp, fp, fn = s["true_positives"], s["false_positives"], s["false_negatives"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / s["total_gt_plates"] if s["total_gt_plates"] > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        n_ocr = s["total_gt_characters"]
        cer = s["total_edit_distance"] / n_ocr if n_ocr > 0 else 0.0
        exact_acc = (s["exact_matches"] / tp * 100.0) if tp > 0 and s["exact_matches"] > 0 else 0.0

        report[group] = {
            "total_frames": s["total_frames"],
            "total_gt_plates": s["total_gt_plates"],
            "detection": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
            },
            "ocr": {
                "exact_matches": s["exact_matches"],
                "exact_match_acc_pct": round(exact_acc, 2),
                "corpus_cer": round(cer, 4),
                "total_gt_characters": n_ocr,
                "total_edit_distance": s["total_edit_distance"],
            },
        }

    report_path = output_dir / "gate_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nGate Evaluation Report saved to {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate ALPR pipeline on gate frames with environmental slicing.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to gate dataset containing images/ and labels/")
    parser.add_argument("--tags", type=Path, default=None, help="Path to tags JSON or CSV (day/night/glare)")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation_gate"))
    args = parser.parse_args()

    evaluate_gate_dataset(args.dataset_dir, args.tags, args.output_dir)


if __name__ == "__main__":
    main()
