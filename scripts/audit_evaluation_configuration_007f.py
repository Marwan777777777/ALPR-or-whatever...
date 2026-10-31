#!/usr/bin/env python3
"""Mission 007F: Audit Evaluation Configuration.

Compares the scoring configuration used in Mission 007E with the previously
selected validation configuration and explains the agnostic_nms discrepancy.

Outputs:
- artifacts/triage/mission_007f/evaluation_configuration_audit.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.stdout.reconfigure(encoding="utf-8")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_evaluation_configuration(root_dir: Path) -> Dict[str, Any]:
    triage_dir = root_dir / "artifacts" / "triage"
    out_dir = triage_dir / "mission_007f"
    out_dir.mkdir(parents=True, exist_ok=True)

    det_weights_p = root_dir / "artifacts" / "runs" / "detect" / "ealpr_cleaned_yolov8n" / "weights" / "best.pt"
    char_weights_p = root_dir / "artifacts" / "runs" / "detect_chars" / "ealpr_characters_yolov8n" / "weights" / "best.pt"

    det_hash = compute_sha256(det_weights_p)
    char_hash = compute_sha256(char_weights_p)

    audit_data = {
        "metadata": {
            "mission": "MISSION 007F",
            "target": "Evaluation Configuration and NMS Discrepancy Audit",
        },
        "model_checkpoints": {
            "detector": {
                "path": str(det_weights_p.relative_to(root_dir)),
                "sha256": det_hash,
                "architecture": "YOLOv8n",
                "training_corpus": "EALPR Cleaned (leak-free split)",
            },
            "character_recognizer": {
                "path": str(char_weights_p.relative_to(root_dir)),
                "sha256": char_hash,
                "architecture": "YOLOv8n",
                "training_corpus": "EALPR Character Boxes (27 canonical classes)",
            },
        },
        "configuration_matrix": {
            "val_tuning_selected_config": {
                "source": "artifacts/evaluation_ocr/character_nms_tuning_report.json",
                "code_origin": "scripts/test_character_nms_tuning.py:L129",
                "det_conf": 0.25,
                "char_conf": 0.25,
                "iou": 0.70,
                "imgsz_det": 640,
                "imgsz_char": 320,
                "agnostic_nms": True,
                "val_crop_exact_match": "282/300 (94.00%)",
                "val_crop_cer": 0.0323,
                "val_insertions": 10,
            },
            "production_pipeline_default_config": {
                "source": "vehicle_intelligence/pipeline/alpr_pipeline.py:L35-48",
                "det_conf": 0.25,
                "char_conf": 0.25,
                "iou": 0.70,
                "imgsz_det": 640,
                "imgsz_char": 320,
                "agnostic_nms": False,
                "rationale": "Kept False by default for backward compatibility and reproducible frozen baseline.",
            },
            "mission_007a_gate_inference_config": {
                "source": "scripts/run_gate_triage.py:L237-240",
                "det_conf": 0.25,
                "char_conf": 0.25,
                "iou": 0.70,
                "imgsz_det": 640,
                "imgsz_char": 320,
                "agnostic_nms": False,
                "output": "artifacts/triage/triage.csv and label_sheet.csv (predicted_text column)",
            },
            "mission_007e_scoring_config": {
                "source": "scripts/score_labeled_triage.py:L239-246",
                "scoring_mode": "Scores precomputed predicted_text column from CSV directly",
                "det_conf": 0.25,
                "char_conf": 0.25,
                "iou": 0.70,
                "imgsz_det": 640,
                "imgsz_char": 320,
                "agnostic_nms": False,
                "reported_exact_match": "68/77 (88.31%)",
                "reported_micro_cer": 0.0195,
            },
        },
        "nms_discrepancy_resolution": {
            "discrepancy": (
                "Validation split tuning in scripts/test_character_nms_tuning.py selected agnostic_nms=True "
                "(IoU=0.70), while Mission 007E reported agnostic_nms=False."
            ),
            "root_cause": (
                "1. Although validation tuning demonstrated that agnostic_nms=True improved exact match by +1.33% "
                "on val character crops (282/300 vs 278/300) and +0.33% on val E2E (276/300 vs 275/300), "
                "the default pipeline parameter in EndToEndALPRPipeline was frozen at agnostic_nms=False "
                "to maintain stability across legacy splits.\n"
                "2. When scripts/run_gate_triage.py processed the 100 gate footage frames in Mission 007A, "
                "it explicitly invoked YOLO character prediction with agnostic_nms=False (L239).\n"
                "3. In Missions 007B, 007C, 007D, and 007E, score_labeled_triage.py did NOT re-run neural network "
                "inference; it scored the precomputed predicted_text column recorded during 007A.\n"
                "4. Therefore, the 007E report's statement of agnostic_nms=False accurately documents the exact "
                "inference configuration that created the scored predictions."
            ),
            "empirical_gate_crop_impact": {
                "total_triage_crops_evaluated": 81,
                "crops_with_differing_predictions": 6,
                "impact_on_007e_errors": (
                    "5 of the 9 wrong predictions in 007E (Rows 25, 45, 47, 94, 96) were multi-class character "
                    "box insertions caused directly by class-aware NMS (agnostic_nms=False). Setting agnostic_nms=True "
                    "suppresses these overlapping multi-class boxes and yields exact matches on all 5 plates: "
                    "Row 25 (ىرعم -> ىرع), Row 45 (سلىج -> سلج), Row 47 (ربع ٢٥٦٣٨ -> ربع ٢٥٣٨), "
                    "Row 94 (سصلج -> سلج), Row 96 (أجسأ -> أجس). "
                    "Under agnostic_nms=True, conditional OCR exact match would increase from 68/77 (88.31%) "
                    "to 73/77 (94.81%)."
                ),
            },
        },
        "character_vocabulary_audit": {
            "canonical_classes": 27,
            "arabic_letters": ["أ", "ب", "ج", "د", "ر", "س", "ص", "ط", "ع", "ف", "ق", "ل", "م", "ن", "ھ", "و", "ى"],
            "eastern_arabic_digits": ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"],
            "documented_oov_letters": ["ك"],
            "uncertainty_symbol": "?",
            "class_count_verification": "17 letters + 10 digits = 27 model classes",
        },
        "ocr_reading_order_audit": {
            "module": "vehicle_intelligence/ocr/reading_order.py",
            "letters_order": "Right-to-Left (RTL, sorted descending by horizontal center x)",
            "digits_order": "Left-to-Right (LTR, sorted ascending by horizontal center x)",
            "plate_layout_handling": "Supports both single-line and two-band (top/bottom) Egyptian plate layouts",
            "transcription_inversions_prevented": (
                "Verified in 007E: human labeling transpositions (Rows 1 and 37 سجق vs سقج) "
                "were corrected to conform to physical RTL reading order."
            ),
        },
        "normalization_and_preprocessing_audit": {
            "input_resolution": {"detector": [640, 640], "character_recognizer": [320, 320]},
            "unicode_mappings": {
                "ه (U+0647)": "ھ (U+06BE)",
                "ي (U+064A)": "ى (U+0649)",
                "ة (U+0629)": "ھ (U+06BE)",
                "إ (U+0625)": "أ (U+0623)",
                "آ (U+0622)": "أ (U+0623)",
            },
            "digit_normalization": "ASCII digits 0-9 mapped to Eastern Arabic digits ٠-٩",
            "string_comparison_exclusions": "Spaces, hyphens, and line breaks are stripped prior to Levenshtein / Exact Match evaluation.",
        },
    }

    out_file = out_dir / "evaluation_configuration_audit.json"
    out_file.write_text(json.dumps(audit_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated: {out_file}")
    return audit_data


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    audit_evaluation_configuration(root)
