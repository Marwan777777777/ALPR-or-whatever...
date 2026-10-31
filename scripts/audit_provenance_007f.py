#!/usr/bin/env python3
"""Mission 007F: Ground Truth Provenance and Independence Audit.

Audits the provenance of all 100 gate triage annotations between:
- completed_annotations.csv (original 007A/007B human annotation)
- completed_annotations_007e.csv (finalized 007E annotation)

Outputs:
- artifacts/triage/mission_007f/row_level_provenance.csv
- artifacts/triage/mission_007f/provenance_audit.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.stdout.reconfigure(encoding="utf-8")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# Documented visual revisions applied in create_annotations_007e.py
REVISION_DETAILS: Dict[int, Dict[str, str]] = {
    1: {
        "rationale": "Independent visual review of crop_001.jpg confirms RTL reading order is Seen (س), Qaf (ق), Jeem (ج) -> 'سقج'. Original label transposed middle and left letters ('سجق').",
        "evidence": "create_annotations_007e.py:L53-63; crop_001.jpg visual inspection",
    },
    14: {
        "rationale": "Visual review of frame_014.jpg confirms side view of passing motorcycle with no plate visible anywhere. Transcription rules require no invented transcriptions for no_plate status.",
        "evidence": "create_annotations_007e.py:L65-78; frame_014.jpg visual inspection",
    },
    18: {
        "rationale": "Visual review of frame_018.jpg confirms square tricycle plate with letters 'كمد'. Contains documented OOV letter Kaf (ك, U+0643) outside the 27 model classes.",
        "evidence": "create_annotations_007e.py:L80-90; frame_018.jpg visual inspection",
    },
    23: {
        "rationale": "Independent visual review of crop_023.jpg confirms RTL reading order is Waw (و), Ain (ع), Raa (ر) -> 'وعر'. Original label 'صجر' was completely mismatched.",
        "evidence": "create_annotations_007e.py:L92-102; crop_023.jpg visual inspection",
    },
    30: {
        "rationale": "Visual review of frame_030.jpg confirms square tricycle plate with letters 'كمد'. Contains documented OOV letter Kaf (ك, U+0643).",
        "evidence": "create_annotations_007e.py:L104-114; frame_030.jpg visual inspection",
    },
    32: {
        "rationale": "Visual review of frame_032.jpg confirms square tricycle plate with letters 'كمد' and digits '٧٦١٢'. Contains documented OOV letter Kaf (ك, U+0643).",
        "evidence": "create_annotations_007e.py:L116-126; frame_032.jpg visual inspection",
    },
    37: {
        "rationale": "Independent visual review of crop_037.jpg confirms RTL reading order is Seen (س), Qaf (ق), Jeem (ج) -> 'سقج'. Original label transposed middle and left letters ('سجق').",
        "evidence": "create_annotations_007e.py:L128-138; crop_037.jpg visual inspection",
    },
    38: {
        "rationale": "Visual review of frame_038.jpg confirms square tricycle plate with letters 'كمد' and digits '٥٨١٧' at night. Populated missing status='readable', lighting='night_ir', and unsupported_chars='yes'.",
        "evidence": "create_annotations_007e.py:L140-152; frame_038.jpg visual inspection",
    },
    55: {
        "rationale": "Visual review of crop_055.jpg confirms rectangular plate with letters 'كدم'. Contains documented OOV letter Kaf (ك, U+0643).",
        "evidence": "create_annotations_007e.py:L154-164; crop_055.jpg visual inspection",
    },
    90: {
        "rationale": "Visual review of crop_090.jpg confirms rectangular plate with letters 'كدو'. Contains documented OOV letter Kaf (ك, U+0643).",
        "evidence": "create_annotations_007e.py:L166-176; crop_090.jpg visual inspection",
    },
    93: {
        "rationale": "Visual review of crop_093.jpg confirms rectangular plate with letters 'كطو'. Contains documented OOV letter Kaf (ك, U+0643).",
        "evidence": "create_annotations_007e.py:L178-188; crop_093.jpg visual inspection",
    },
}


def audit_provenance(root_dir: Path) -> Dict[str, Any]:
    triage_dir = root_dir / "artifacts" / "triage"
    out_dir = triage_dir / "mission_007f"
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_csv_p = triage_dir / "completed_annotations.csv"
    final_csv_p = triage_dir / "completed_annotations_007e.csv"

    if not orig_csv_p.exists() or not final_csv_p.exists():
        raise FileNotFoundError("Missing original or final annotation CSV.")

    orig_hash = compute_sha256(orig_csv_p)
    final_hash = compute_sha256(final_csv_p)

    with open(orig_csv_p, "r", encoding="utf-8-sig") as f:
        orig_rows = list(csv.DictReader(f))

    with open(final_csv_p, "r", encoding="utf-8-sig") as f:
        final_rows = list(csv.DictReader(f))

    if len(orig_rows) != 100 or len(final_rows) != 100:
        raise ValueError(f"Expected 100 rows each, got {len(orig_rows)} and {len(final_rows)}.")

    provenance_rows: List[Dict[str, Any]] = []
    category_counts = Counter()
    independence_counts = Counter()

    for idx in range(1, 101):
        sample_id = f"sample_{idx:03d}"
        r_orig = orig_rows[idx - 1]
        r_final = final_rows[idx - 1]

        orig_annot = f"{r_orig.get('true_letters', '')} {r_orig.get('true_digits', '')}".strip()
        final_annot = f"{r_final.get('true_letters', '')} {r_final.get('true_digits', '')}".strip()

        # Check all schema fields for differences
        diff_fields = {}
        for k in r_final.keys():
            v_orig = r_orig.get(k, "")
            v_final = r_final.get(k, "")
            if v_orig != v_final:
                diff_fields[k] = {"orig": v_orig, "final": v_final}

        changed = len(diff_fields) > 0

        if idx in REVISION_DETAILS:
            category = "visually_rechecked"
            evidence = REVISION_DETAILS[idx]["evidence"]
            independence_status = "partially_independent"
            explanation = REVISION_DETAILS[idx]["rationale"]
        elif changed:
            category = "visually_rechecked"
            evidence = "Differing fields in create_annotations_007e.py"
            independence_status = "partially_independent"
            explanation = f"Fields modified: {list(diff_fields.keys())}"
        else:
            category = "copied_from_prior_annotation"
            evidence = f"completed_annotations.csv (SHA-256: {orig_hash[:12]}...) row {idx} byte-identical"
            independence_status = "not_independent"
            explanation = (
                "Copied verbatim from completed_annotations.csv without de novo blind transcription. "
                "Annotator in 007A had unblinded access to legacy OCR filenames and model predictions."
            )

        category_counts[category] += 1
        independence_counts[independence_status] += 1

        provenance_rows.append({
            "sample_id": sample_id,
            "original_annotation": orig_annot,
            "final_annotation": final_annot,
            "changed": "true" if changed else "false",
            "provenance_category": category,
            "evidence_available": evidence,
            "independence_status": independence_status,
            "explanation": explanation,
        })

    # Write row_level_provenance.csv
    provenance_csv_p = out_dir / "row_level_provenance.csv"
    with open(provenance_csv_p, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "sample_id",
            "original_annotation",
            "final_annotation",
            "changed",
            "provenance_category",
            "evidence_available",
            "independence_status",
            "explanation",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(provenance_rows)

    # Build provenance audit JSON
    audit_summary = {
        "metadata": {
            "mission": "MISSION 007F",
            "target": "Ground Truth Independence & Provenance Audit",
            "original_csv": "artifacts/triage/completed_annotations.csv",
            "original_csv_sha256": orig_hash,
            "final_csv": "artifacts/triage/completed_annotations_007e.csv",
            "final_csv_sha256": final_hash,
            "total_samples": 100,
        },
        "provenance_breakdown": {
            "new_blind_transcription": category_counts["new_blind_transcription"],
            "copied_from_prior_annotation": category_counts["copied_from_prior_annotation"],
            "visually_rechecked": category_counts["visually_rechecked"],
            "unknown": category_counts["unknown"],
        },
        "independence_status_breakdown": {
            "fully_independent": independence_counts["fully_independent"],
            "partially_independent": independence_counts["partially_independent"],
            "not_independent": independence_counts["not_independent"],
        },
        "independence_criteria_status": "BLOCKED",
        "independence_verdict": {
            "status": "BLOCKED",
            "claim_independent_ground_truth": False,
            "summary": (
                "Complete independent transcription of all 100 samples from scratch was NOT performed. "
                "Exactly 89 samples were copied verbatim from the unblinded completed_annotations.csv, "
                "and 11 samples were visually rechecked / repaired in scripts/create_annotations_007e.py. "
                "While independent_labeling_interface.html was authored in 007E, no de novo 100-sample "
                "transcription run was conducted or exported from it. Ground truth independence item is marked BLOCKED."
            ),
        },
        "annotator_bias_exposure": {
            "original_annotation_session_007a": {
                "filenames_contained_legacy_anpr": True,
                "legacy_ocr_visible": True,
                "model_predictions_present_in_sheet": True,
                "confidence_scores_visible": True,
            },
            "recheck_session_007e": {
                "targeted_crops_inspected": 11,
                "full_blind_retranscription_completed": False,
            },
        },
        "rechecked_rows_inventory": [
            {"row": idx, "sample_id": f"sample_{idx:03d}", **REVISION_DETAILS[idx]}
            for idx in sorted(REVISION_DETAILS.keys())
        ],
    }

    audit_json_p = out_dir / "provenance_audit.json"
    audit_json_p.write_text(json.dumps(audit_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Generated: {provenance_csv_p}")
    print(f"Generated: {audit_json_p}")
    print(f"Categories: {dict(category_counts)}")
    print(f"Independence: {dict(independence_counts)}")

    return audit_summary


if __name__ == "__main__":
    from collections import Counter
    root = Path(__file__).resolve().parent.parent
    audit_provenance(root)
