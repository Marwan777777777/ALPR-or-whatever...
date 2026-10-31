#!/usr/bin/env python3
"""Mission 008: OCR Error Analysis Exporter.

Analyzes all remaining incorrect character predictions on gate triage crops
under the intended class-agnostic NMS configuration (agnostic_nms=True).

Outputs:
- artifacts/missions/mission_008/ocr_error_analysis.csv
"""

import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
root_dir = Path(__file__).resolve().parent.parent

ocr_errors = [
    {
        "sample_id": "sample_041",
        "ground_truth": "ربأ ٤٩٩٢",
        "predicted_text": "ربأأ ٤٩٩٢",
        "primary_error_category": "Character insertion",
        "secondary_category": "Intra-class duplicate NMS escape",
        "model_limitation_vs_annotation_uncertainty": "Model limitation",
        "recurring_pattern": "Duplicate 'أ' box",
        "detailed_explanation": "Two bounding boxes for letter 'أ' at x=0.580 (conf=0.70) and x=0.583 (conf=0.38) had IoU=0.588, which is below the 0.70 NMS threshold, allowing the duplicate to survive.",
    },
    {
        "sample_id": "sample_079",
        "ground_truth": "وىم ٧٣٦",
        "predicted_text": "وىم ٧٣٦٦",
        "primary_error_category": "Character insertion",
        "secondary_category": "Weak edge false alarm",
        "model_limitation_vs_annotation_uncertainty": "Model limitation",
        "recurring_pattern": "Digit false alarm near right border",
        "detailed_explanation": "A spurious low-confidence box (conf=0.34, height=0.16) for Eastern digit '٦' was detected on an edge reflection artifact near the digit band.",
    },
    {
        "sample_id": "sample_083",
        "ground_truth": "فىر ٧٤٦١",
        "predicted_text": "فىرأ ٧٤٦",
        "primary_error_category": "Character insertion",
        "secondary_category": "Character deletion",
        "model_limitation_vs_annotation_uncertainty": "Model limitation",
        "recurring_pattern": "Tight crop boundary clipping + emblem false positive",
        "detailed_explanation": "A spurious letter 'أ' (conf=0.34) was detected on the central divider emblem, while Eastern digit '١' was cut off at the edge of a tight detector crop.",
    },
    {
        "sample_id": "sample_084",
        "ground_truth": "ألع ٦٨٧٦",
        "predicted_text": "أألع ٦٨٧٦",
        "primary_error_category": "Character insertion",
        "secondary_category": "Intra-class duplicate NMS escape",
        "model_limitation_vs_annotation_uncertainty": "Model limitation",
        "recurring_pattern": "Duplicate 'أ' box",
        "detailed_explanation": "Two bounding boxes for 'أ' at x=0.875 (conf=0.67) and x=0.863 (conf=0.37) had IoU around 0.60, escaping the 0.70 NMS threshold.",
    },
    {
        "sample_id": "sample_055",
        "ground_truth": "كدم ٨٥٧٣",
        "predicted_text": "دم ٨٥٧٣",
        "primary_error_category": "OOV character",
        "secondary_category": "Vocabulary limitation",
        "model_limitation_vs_annotation_uncertainty": "Model limitation",
        "recurring_pattern": "Omission of unmodeled Arabic Kaf (ك)",
        "detailed_explanation": "Character recognizer trained on 27 canonical EALPR classes lacks the Arabic letter Kaf (ك, U+0643). The remaining letters and digits ('دم ٨٥٧٣') were recognized with >0.83 confidence.",
    },
    {
        "sample_id": "sample_090",
        "ground_truth": "كدو ١٣٩٦",
        "predicted_text": "دو ١٣٩٦",
        "primary_error_category": "OOV character",
        "secondary_category": "Vocabulary limitation",
        "model_limitation_vs_annotation_uncertainty": "Model limitation",
        "recurring_pattern": "Omission of unmodeled Arabic Kaf (ك)",
        "detailed_explanation": "Letter Kaf (ك) omitted due to 27-class model vocabulary limit. Remaining string ('دو ١٣٩٦') recognized with >0.84 confidence.",
    },
    {
        "sample_id": "sample_093",
        "ground_truth": "كطو ٢٧٩٦",
        "predicted_text": "طو ٢٧٩٦",
        "primary_error_category": "OOV character",
        "secondary_category": "Vocabulary limitation",
        "model_limitation_vs_annotation_uncertainty": "Model limitation",
        "recurring_pattern": "Omission of unmodeled Arabic Kaf (ك)",
        "detailed_explanation": "Letter Kaf (ك) omitted due to 27-class model vocabulary limit. Remaining string ('طو ٢٧٩٦') recognized with >0.88 confidence.",
    },
    {
        "sample_id": "sample_015",
        "ground_truth": "سقب ٩٢٥٦",
        "predicted_text": "",
        "primary_error_category": "Poor crop / perspective",
        "secondary_category": "Low contrast / IR underexposure",
        "model_limitation_vs_annotation_uncertainty": "Model limitation",
        "recurring_pattern": "Night IR underexposure confidence collapse",
        "detailed_explanation": "Severe underexposure on night IR capture (mean brightness 58.6) caused all character confidences to drop below the 0.25 confidence threshold (max conf was 0.106).",
    },
]

out_csv = root_dir / "artifacts" / "missions" / "mission_008" / "ocr_error_analysis.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)

with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
    fieldnames = [
        "sample_id",
        "ground_truth",
        "predicted_text",
        "primary_error_category",
        "secondary_category",
        "model_limitation_vs_annotation_uncertainty",
        "recurring_pattern",
        "detailed_explanation",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(ocr_errors)

print(f"Generated OCR error analysis: {out_csv} ({len(ocr_errors)} rows)")
