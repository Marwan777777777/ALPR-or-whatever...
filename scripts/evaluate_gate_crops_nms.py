#!/usr/bin/env python3
"""Mission 008: Controlled Character Detector NMS Evaluation.

Evaluates character recognition across all 81 gate triage crops under:
1. Historical baseline: agnostic_nms=False
2. Mission 008 intended configuration: agnostic_nms=True

Validates individual corrected crops (Rows 25, 45, 47, 94, 96) and outputs:
- artifacts/missions/mission_008/nms_comparison.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.audit_end_to_end_and_ocr import (
    parse_character_boxes_from_yolo,
)
from scripts.score_labeled_triage import (
    normalize_plate_string,
    align_and_classify_errors,
    VALID_CHARACTERS,
    DOCUMENTED_OOV_LETTERS,
)
from vehicle_intelligence.ocr.reading_order import EgyptianPlateReadingOrder

sys.stdout.reconfigure(encoding="utf-8")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_gate_crops(device: str = "cuda:0") -> Dict[str, Any]:
    char_weights_p = root_dir / "artifacts" / "runs" / "detect_chars" / "ealpr_characters_yolov8n" / "weights" / "best.pt"
    det_weights_p = root_dir / "artifacts" / "runs" / "detect" / "ealpr_cleaned_yolov8n" / "weights" / "best.pt"
    csv_p = root_dir / "artifacts" / "triage" / "completed_annotations_007e.csv"

    out_dir = root_dir / "artifacts" / "missions" / "mission_008"
    out_dir.mkdir(parents=True, exist_ok=True)

    char_hash = compute_sha256(char_weights_p)
    det_hash = compute_sha256(det_weights_p)

    char_model = YOLO(str(char_weights_p))

    with open(csv_p, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # Evaluate 81 crops
    # Collect results under False and True
    results_by_nms = {"false": [], "true": []}

    target_5_rows = [25, 45, 47, 94, 96]
    target_5_results = {}

    start_time_gpu = time.time()
    for agn in [False, True]:
        key = "true" if agn else "false"
        for idx, r in enumerate(rows, start=1):
            crop_p = root_dir / "artifacts" / "triage" / "samples" / f"crop_{idx:03d}.jpg"
            has_crop = crop_p.exists()
            status = r.get("status", "").strip().lower()
            unsupported = r.get("unsupported_chars", "").strip().lower() in ("yes", "true", "1")
            gt_letters = r.get("true_letters", "").strip()
            gt_digits = r.get("true_digits", "").strip()
            gt_raw = f"{gt_letters} {gt_digits}".strip()
            gt_norm = normalize_plate_string(gt_raw)

            has_oov = any(ch in DOCUMENTED_OOV_LETTERS or ch not in VALID_CHARACTERS for ch in gt_letters + gt_digits)

            if not has_crop:
                continue

            im = cv2.imread(str(crop_p))
            res = char_model.predict(
                source=im,
                conf=0.25,
                iou=0.70,
                agnostic_nms=agn,
                imgsz=320,
                device=device,
                verbose=False,
            )[0]
            boxes = parse_character_boxes_from_yolo(res, conf_thresh=0.25)
            pred_text, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(boxes)
            pred_norm = normalize_plate_string(pred_text)

            is_in_vocab_readable = (status == "readable" and not unsupported and not has_oov)
            is_exact = (pred_norm == gt_norm and len(gt_norm) > 0) if is_in_vocab_readable else False
            align = align_and_classify_errors(gt_norm, pred_norm) if is_in_vocab_readable else None

            rec = {
                "idx": idx,
                "sample_id": f"sample_{idx:03d}",
                "status": status,
                "gt_raw": gt_raw,
                "gt_norm": gt_norm,
                "pred_text": pred_text,
                "pred_norm": pred_norm,
                "is_in_vocab_readable": is_in_vocab_readable,
                "is_exact": is_exact,
                "edit_distance": align["edit_distance"] if align else None,
                "insertions": align["insertions"] if align else None,
                "deletions": align["deletions"] if align else None,
                "substitutions": align["substitutions"] if align else None,
            }
            results_by_nms[key].append(rec)

            if idx in target_5_rows:
                if idx not in target_5_results:
                    target_5_results[idx] = {
                        "row": idx,
                        "sample_id": f"sample_{idx:03d}",
                        "ground_truth": gt_raw,
                    }
                target_5_results[idx][f"pred_agnostic_{key}"] = pred_text
                target_5_results[idx][f"exact_agnostic_{key}"] = is_exact

    total_infer_time = time.time() - start_time_gpu

    # Summary metrics
    summaries = {}
    for key in ["false", "true"]:
        recs = results_by_nms[key]
        eval_recs = [r for r in recs if r["is_in_vocab_readable"]]
        n_eval = len(eval_recs)  # 77
        exact_cnt = sum(1 for r in eval_recs if r["is_exact"])
        total_ed = sum(r["edit_distance"] for r in eval_recs)
        total_gt_len = sum(len(r["gt_norm"]) for r in eval_recs)
        total_ins = sum(r["insertions"] for r in eval_recs)
        total_del = sum(r["deletions"] for r in eval_recs)
        total_sub = sum(r["substitutions"] for r in eval_recs)

        wrong_recs = [r for r in eval_recs if not r["is_exact"]]

        summaries[key] = {
            "agnostic_nms": (key == "true"),
            "total_crops_evaluated": len(recs),
            "in_vocab_readable_denominator": n_eval,
            "exact_matches_numerator": exact_cnt,
            "exact_match_accuracy_pct": round(exact_cnt / n_eval * 100.0, 2),
            "total_edit_distance": total_ed,
            "total_gt_characters": total_gt_len,
            "micro_cer": round(total_ed / max(1, total_gt_len), 4),
            "macro_cer": round(float(np.mean([r["edit_distance"] / len(r["gt_norm"]) for r in eval_recs])), 4),
            "insertions": total_ins,
            "deletions": total_del,
            "substitutions": total_sub,
            "wrong_plates_count": len(wrong_recs),
            "wrong_plates_details": [
                {
                    "row": r["idx"],
                    "sample_id": r["sample_id"],
                    "gt": r["gt_norm"],
                    "pred": r["pred_norm"],
                    "edit_distance": r["edit_distance"],
                    "insertions": r["insertions"],
                    "deletions": r["deletions"],
                    "substitutions": r["substitutions"],
                }
                for r in wrong_recs
            ],
        }

    comparison_report = {
        "metadata": {
            "mission": "MISSION 008",
            "objective": "Objective 1: Controlled NMS Correction",
            "eval_date": "2026-10-30",
            "hardware": {
                "device": device,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "total_inference_time_seconds": round(total_infer_time, 2),
            },
            "model_checkpoints": {
                "plate_detector": {
                    "path": str(det_weights_p.relative_to(root_dir)),
                    "sha256": det_hash,
                },
                "character_recognizer": {
                    "path": str(char_weights_p.relative_to(root_dir)),
                    "sha256": char_hash,
                },
            },
            "fixed_parameters": {
                "conf_thresh": 0.25,
                "iou_thresh": 0.70,
                "imgsz": 320,
            },
        },
        "target_5_corrected_crops_verification": [target_5_results[k] for k in sorted(target_5_results.keys())],
        "historical_baseline_agnostic_false": summaries["false"],
        "mission_008_intended_agnostic_true": summaries["true"],
        "delta_improvement": {
            "exact_matches_gain": summaries["true"]["exact_matches_numerator"] - summaries["false"]["exact_matches_numerator"],
            "exact_match_pct_gain": round(summaries["true"]["exact_match_accuracy_pct"] - summaries["false"]["exact_match_accuracy_pct"], 2),
            "cer_reduction_pct": round((summaries["false"]["micro_cer"] - summaries["true"]["micro_cer"]) / summaries["false"]["micro_cer"] * 100.0, 2),
            "insertions_eliminated": summaries["false"]["insertions"] - summaries["true"]["insertions"],
        },
    }

    out_json = out_dir / "nms_comparison.json"
    out_json.write_text(json.dumps(comparison_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated: {out_json}")
    print(f"Historical (False): {summaries['false']['exact_matches_numerator']}/{summaries['false']['in_vocab_readable_denominator']} ({summaries['false']['exact_match_accuracy_pct']}%), CER={summaries['false']['micro_cer']}")
    print(f"Mission 008 (True): {summaries['true']['exact_matches_numerator']}/{summaries['true']['in_vocab_readable_denominator']} ({summaries['true']['exact_match_accuracy_pct']}%), CER={summaries['true']['micro_cer']}")
    print(f"Gain: +{comparison_report['delta_improvement']['exact_matches_gain']} plates (+{comparison_report['delta_improvement']['exact_match_pct_gain']}%), -{comparison_report['delta_improvement']['insertions_eliminated']} insertions")

    return comparison_report


if __name__ == "__main__":
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    evaluate_gate_crops(device=dev)
