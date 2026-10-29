#!/usr/bin/env python3
"""Investigate character insertion errors and evaluate NMS parameters on Validation split first, then Test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ultralytics import YOLO

from scripts.audit_end_to_end_and_ocr import (
    parse_character_boxes_from_yolo,
    parse_gt_characters_from_file,
    levenshtein_distance,
)
from vehicle_intelligence.ocr.reading_order import EgyptianPlateReadingOrder


def evaluate_nms_on_split(
    char_model: YOLO,
    char_dataset_dir: Path,
    split: str = "val",
    iou_thresh: float = 0.70,
    agnostic_nms: bool = False,
    conf_thresh: float = 0.25,
) -> dict:
    img_dir = char_dataset_dir / "images" / split
    lbl_dir = char_dataset_dir / "labels" / split
    img_files = sorted(list(img_dir.glob("*.png")))

    exact_matches = 0
    total_edit_distance = 0
    total_gt_chars = 0
    insertions_count = 0

    for img_p in img_files:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        gt_boxes = parse_gt_characters_from_file(lbl_p)
        gt_str, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(gt_boxes)
        gt_norm = gt_str.replace(" ", "")

        res = char_model.predict(
            source=str(img_p),
            conf=conf_thresh,
            iou=iou_thresh,
            agnostic_nms=agnostic_nms,
            imgsz=320,
            verbose=False,
        )[0]
        pred_boxes = parse_character_boxes_from_yolo(res, conf_thresh=conf_thresh)
        pred_str, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(pred_boxes)
        pred_norm = pred_str.replace(" ", "")

        dist = levenshtein_distance(gt_norm, pred_norm)
        if dist == 0:
            exact_matches += 1
        elif len(pred_norm) > len(gt_norm):
            insertions_count += 1

        total_edit_distance += dist
        total_gt_chars += len(gt_norm)

    total_plates = len(img_files)
    acc = exact_matches / total_plates * 100.0 if total_plates else 0.0
    cer = total_edit_distance / total_gt_chars if total_gt_chars else 0.0

    return {
        "split": split,
        "iou_thresh": iou_thresh,
        "agnostic_nms": agnostic_nms,
        "total_plates": total_plates,
        "exact_matches": exact_matches,
        "exact_match_acc": round(acc, 2),
        "corpus_cer": round(cer, 4),
        "insertions_count": insertions_count,
    }


def main():
    print("Loading character model...")
    char_model = YOLO("artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt")
    char_dataset_dir = Path("artifacts/ealpr_characters_yolo_dataset")

    print("\n--- TUNING ON VALIDATION SPLIT ---")
    configs = [
        (0.70, False),  # Baseline
        (0.60, False),
        (0.50, False),
        (0.45, False),
        (0.40, False),
        (0.70, True),   # Class-agnostic NMS
        (0.50, True),
        (0.45, True),
    ]

    val_results = []
    print(f"{'IoU Thresh':<12} | {'Agnostic NMS':<14} | {'Exact Match (%)':<17} | {'Corpus CER':<12} | {'Insertions':<12}")
    print("-" * 75)

    best_cfg = None
    best_val_acc = -1.0

    for iou_t, agn in configs:
        res = evaluate_nms_on_split(char_model, char_dataset_dir, split="val", iou_thresh=iou_t, agnostic_nms=agn)
        val_results.append(res)
        print(f"{iou_t:<12} | {str(agn):<14} | {res['exact_match_acc']:<17.2f} | {res['corpus_cer']:<12.4f} | {res['insertions_count']:<12}")
        if res["exact_match_acc"] > best_val_acc:
            best_val_acc = res["exact_match_acc"]
            best_cfg = (iou_t, agn)

    print(f"\nBest configuration chosen on VAL: IoU={best_cfg[0]}, Agnostic={best_cfg[1]} (Val Acc: {best_val_acc}%)")

    print("\n--- REPORTING ONCE ON TEST SPLIT ---")
    baseline_test = evaluate_nms_on_split(char_model, char_dataset_dir, split="test", iou_thresh=0.70, agnostic_nms=False)
    tuned_test = evaluate_nms_on_split(char_model, char_dataset_dir, split="test", iou_thresh=best_cfg[0], agnostic_nms=best_cfg[1])

    print(f"Test Baseline (IoU=0.70, Agnostic=False): Acc={baseline_test['exact_match_acc']}%, CER={baseline_test['corpus_cer']}, Insertions={baseline_test['insertions_count']}")
    print(f"Test Tuned    (IoU={best_cfg[0]}, Agnostic={best_cfg[1]}): Acc={tuned_test['exact_match_acc']}%, CER={tuned_test['corpus_cer']}, Insertions={tuned_test['insertions_count']}")

    out_file = Path("artifacts/evaluation_ocr/character_nms_tuning_report.json")
    out_file.write_text(json.dumps({
        "val_tuning_results": val_results,
        "best_config_selected_on_val": {"iou_thresh": best_cfg[0], "agnostic_nms": best_cfg[1]},
        "test_baseline": baseline_test,
        "test_tuned": tuned_test,
    }, indent=2), encoding="utf-8")
    print(f"Saved NMS tuning report to {out_file}")


if __name__ == "__main__":
    main()
