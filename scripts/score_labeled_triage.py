#!/usr/bin/env python3
"""Score Labeled Gate CCTV Triage Sheet (Mission 007A).

Comprehensive, auditable scorer for gate footage plate annotations:
1. Validates CSV schema and characters strictly against 27 EALPR classes (+ '?').
2. Evaluates Exact Match accuracy among readable plates.
3. Computes Character Error Rate (Macro and Micro CER).
4. Computes All-In Accuracy (counting unreadable and missed detections as failures).
5. Computes Plate Detection Recall (across 100 sample including no-detection frames).
6. Classifies OCR errors into substitutions, insertions, and deletions via Levenshtein alignment.
7. Evaluates Legacy ANPR agreement (Human vs Legacy, Human vs Model, Model vs Legacy).
8. Computes Cluster-Aware 95% Confidence Intervals via transit cluster bootstrapping.
9. Computes Wilson 95% Confidence Intervals.
10. Emits human-readable summary, machine-readable JSON, and SHA-256 hash of the scored CSV.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Canonical 27 EALPR character classes
EALPR_LETTERS = ["أ", "ب", "ج", "د", "ر", "س", "ص", "ط", "ع", "ف", "ق", "ل", "م", "ن", "ھ", "و", "ى"]
EALPR_DIGITS = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"]
VALID_CHARACTERS: Set[str] = set(EALPR_LETTERS) | set(EALPR_DIGITS) | {"?"}
VALID_STATUSES: Set[str] = {"readable", "partially_readable", "unreadable", "no_plate", "multiple_plates"}

ASCII_TO_EASTERN_DIGITS = {
    "0": "٠", "1": "١", "2": "٢", "3": "٣", "4": "٤",
    "5": "٥", "6": "٦", "7": "٧", "8": "٨", "9": "٩",
}

UNICODE_NORMALIZATION = {
    "ه": "ھ",  # U+0647 -> U+06BE
    "ي": "ى",  # U+064A -> U+0649
    "ة": "ھ",
    "إ": "أ",
    "آ": "أ",
}


def normalize_plate_string(text: str) -> str:
    """Normalizes characters to the canonical 27 EALPR classes and strips whitespace."""
    out = []
    for ch in text.strip():
        if ch in (" ", "\t", "\r", "\n", "-"):
            continue
        ch = UNICODE_NORMALIZATION.get(ch, ch)
        ch = ASCII_TO_EASTERN_DIGITS.get(ch, ch)
        out.append(ch)
    return "".join(out)


def parse_timestamp(filename: str) -> Optional[datetime.datetime]:
    m = re.match(r"^(\d{8})_(\d{6})", filename)
    if m:
        try:
            return datetime.datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
        except Exception:
            return None
    return None


def extract_cluster_id(filename: str, time_bucket_sec: float = 30.0) -> str:
    """Groups frames of the same vehicle arrival into a time bucket cluster."""
    ts = parse_timestamp(filename)
    if ts:
        epoch_bucket = int(ts.timestamp() // time_bucket_sec)
        return f"time_bucket_{epoch_bucket}"
    parts = filename.split("_")
    return parts[0] if parts else filename


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def align_and_classify_errors(gt: str, pred: str) -> Dict[str, Any]:
    """Levenshtein alignment to classify edits into substitutions, insertions, and deletions."""
    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # substitution
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j],      # deletion
                )

    # Backtrack to count operations
    i, j = m, n
    substitutions = 0
    insertions = 0
    deletions = 0
    edits = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and gt[i - 1] == pred[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            edits.append(("substitution", f"expected '{gt[i - 1]}', got '{pred[j - 1]}'"))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            edits.append(("insertion", f"extra '{pred[j - 1]}'"))
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            edits.append(("deletion", f"missing '{gt[i - 1]}'"))
            i -= 1
        else:
            break

    edits.reverse()
    return {
        "edit_distance": dp[m][n],
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "edits": edits,
    }


def validate_row_schema(idx: int, r: Dict[str, str], strict: bool = True) -> Tuple[bool, List[str]]:
    """Validates schema, statuses, and character classes strictly."""
    errors = []
    fn = r.get("file", "").strip()
    status = r.get("status", "").strip().lower()
    true_letters = r.get("true_letters", "").strip()
    true_digits = r.get("true_digits", "").strip()

    if not fn:
        errors.append(f"Row {idx}: missing 'file' column value.")

    if not status:
        errors.append(f"Row {idx} ({fn}): missing 'status'. Must be one of {sorted(VALID_STATUSES)}.")
    elif status not in VALID_STATUSES:
        errors.append(f"Row {idx} ({fn}): invalid status '{status}'. Must be one of {sorted(VALID_STATUSES)}.")

    # Validate character classes in true_letters and true_digits
    for field_name, val in [("true_letters", true_letters), ("true_digits", true_digits)]:
        for ch in val:
            if ch in (" ", "\t", "-"):
                continue
            if ch not in VALID_CHARACTERS:
                cp = f"U+{ord(ch):04X}"
                errors.append(
                    f"Row {idx} ({fn}): character '{ch}' ({cp}) in '{field_name}' is outside the 27 EALPR classes + '?'."
                )

    # Status-specific semantic rules
    if status == "readable":
        if not true_letters:
            errors.append(f"Row {idx} ({fn}): status is 'readable' but 'true_letters' is empty.")
        if not true_digits:
            errors.append(f"Row {idx} ({fn}): status is 'readable' but 'true_digits' is empty.")
        if "?" in true_letters or "?" in true_digits:
            errors.append(f"Row {idx} ({fn}): status is 'readable' but contains '?' (uncertain characters). Use 'partially_readable'.")

    return (len(errors) == 0, errors)


def score_labeled_csv(
    csv_path: Path,
    json_output_path: Optional[Path] = None,
    bootstrap_draws: int = 10000,
    seed: int = 42,
    strict: bool = True,
) -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    csv_hash = compute_file_sha256(csv_path)

    # Read CSV with utf-8-sig to automatically handle UTF-8 BOM
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file {csv_path} contains no data rows.")

    validation_errors = []
    seen_files = set()

    for idx, r in enumerate(rows, start=1):
        fn = r.get("file", "").strip()
        if fn in seen_files:
            validation_errors.append(f"Row {idx}: duplicate file '{fn}'.")
        seen_files.add(fn)

        ok, errs = validate_row_schema(idx, r, strict=strict)
        if not ok:
            validation_errors.extend(errs)

    if validation_errors:
        err_msg = "\n".join(validation_errors[:20])
        if len(validation_errors) > 20:
            err_msg += f"\n... and {len(validation_errors) - 20} more errors."
        raise ValueError(f"CSV Schema Validation Failed with {len(validation_errors)} error(s):\n{err_msg}")

    # Processing records
    status_counts = Counter()
    evaluated_readable_records = []
    all_in_records = []
    excluded_samples = []

    # Detection tracking
    total_samples = len(rows)
    true_plates_visible = 0
    detected_plates_count = 0
    missed_detection_count = 0
    wrong_crop_count = 0
    unsupported_chars_count = 0

    # Error taxonomy
    total_substitutions = 0
    total_insertions = 0
    total_deletions = 0
    cross_vehicle_associations = 0

    # Legacy comparisons
    legacy_comparable = 0
    legacy_exact_matches = 0
    model_legacy_exact_matches = 0
    latin_legacy_count = 0
    latin_legacy_samples = []

    for idx, r in enumerate(rows, start=1):
        fn = r["file"].strip()
        status = r["status"].strip().lower()
        status_counts[status] += 1

        crop_is_plate = r.get("crop_is_plate", "yes").strip().lower()
        unsupported_chars = r.get("unsupported_chars", "no").strip().lower() in ("yes", "true", "1")
        pred_raw = r.get("predicted_text", "").strip()
        legacy_raw = r.get("legacy_text", "").strip()
        letters_raw = r.get("true_letters", "").strip()
        digits_raw = r.get("true_digits", "").strip()
        visible = r.get("plate_visible_to_eye", "").strip().lower()
        lighting = r.get("lighting", "").strip().lower() or "unspecified"
        notes = r.get("notes", "").strip()

        cluster_id = extract_cluster_id(fn)
        has_detection = bool(pred_raw)

        # Detection ground truth evaluation
        is_plate_visible = visible in ("yes", "partial") or status in ("readable", "partially_readable", "unreadable", "multiple_plates")
        if is_plate_visible:
            true_plates_visible += 1

        # Check for wrong crop (detector hallucination / false crop)
        is_wrong_crop = (crop_is_plate == "no")
        if is_wrong_crop:
            wrong_crop_count += 1
            # If a plate was actually visible on the truck, the detector missed it by cropping the wrong region
            if is_plate_visible:
                missed_detection_count += 1
            excluded_samples.append({
                "index": idx,
                "file": fn,
                "cluster_id": cluster_id,
                "status": status,
                "reason": "Wrong crop (crop_is_plate=no): detection error on non-plate region, excluded from OCR scoring.",
                "notes": notes,
            })
            if is_plate_visible:
                all_in_records.append({"cluster_id": cluster_id, "success": False})
            continue

        if has_detection:
            detected_plates_count += 1
        elif is_plate_visible:
            missed_detection_count += 1

        # Check for unsupported character classes (e.g. Latin letters)
        if unsupported_chars:
            unsupported_chars_count += 1
            excluded_samples.append({
                "index": idx,
                "file": fn,
                "cluster_id": cluster_id,
                "status": status,
                "reason": "Unsupported character classes present (e.g. Latin letters): excluded from 27-class OCR evaluation.",
                "notes": notes,
            })
            if is_plate_visible:
                all_in_records.append({"cluster_id": cluster_id, "success": False})
            continue

        # Readable Plate Evaluation
        if status == "readable":
            gt_full = f"{letters_raw} {digits_raw}"
            gt_norm = normalize_plate_string(gt_full)
            pred_norm = normalize_plate_string(pred_raw)

            align = align_and_classify_errors(gt_norm, pred_norm)
            is_exact = (pred_norm == gt_norm and len(gt_norm) > 0)
            gt_len = max(1, len(gt_norm))
            cer = align["edit_distance"] / gt_len

            total_substitutions += align["substitutions"]
            total_insertions += align["insertions"]
            total_deletions += align["deletions"]

            rec = {
                "index": idx,
                "file": fn,
                "cluster_id": cluster_id,
                "status": status,
                "gt_text": gt_full,
                "pred_text": pred_raw,
                "gt_norm": gt_norm,
                "pred_norm": pred_norm,
                "is_exact": is_exact,
                "edit_distance": align["edit_distance"],
                "cer": cer,
                "substitutions": align["substitutions"],
                "insertions": align["insertions"],
                "deletions": align["deletions"],
                "lighting": lighting,
                "visible": visible,
                "notes": notes,
            }
            evaluated_readable_records.append(rec)
            all_in_records.append({"cluster_id": cluster_id, "success": is_exact})

            # Legacy comparison
            if legacy_raw:
                # Check for Latin-letter transliterated legacy string
                if re.search(r"[a-zA-Z]", legacy_raw):
                    latin_legacy_count += 1
                    latin_legacy_samples.append({"file": fn, "legacy_text": legacy_raw, "gt_text": gt_full})
                else:
                    legacy_norm = normalize_plate_string(legacy_raw)
                    legacy_comparable += 1
                    if legacy_norm == gt_norm:
                        legacy_exact_matches += 1
                    if legacy_norm == pred_norm:
                        model_legacy_exact_matches += 1

        else:
            # Excluded from readable-only OCR scoring
            excluded_samples.append({
                "index": idx,
                "file": fn,
                "cluster_id": cluster_id,
                "status": status,
                "reason": f"Status '{status}' - cannot establish confident ground truth plate text.",
                "notes": notes,
            })
            # For all-in evaluation: unreadable, partially_readable, and missed detections count as failure
            if is_plate_visible:
                all_in_records.append({"cluster_id": cluster_id, "success": False})

    # 1. Detection Recall
    det_recall_pct = round((detected_plates_count / true_plates_visible * 100.0), 2) if true_plates_visible else 0.0

    # 2. Readable Plate Metrics
    n_readable = len(evaluated_readable_records)
    if n_readable > 0:
        n_exact_readable = sum(1 for r in evaluated_readable_records if r["is_exact"])
        readable_acc_pct = round(n_exact_readable / n_readable * 100.0, 2)
        macro_cer = round(float(np.mean([r["cer"] for r in evaluated_readable_records])), 4)
        total_edit_distance = sum(r["edit_distance"] for r in evaluated_readable_records)
        total_gt_chars = sum(len(r["gt_norm"]) for r in evaluated_readable_records)
        micro_cer = round(total_edit_distance / max(1, total_gt_chars), 4)
    else:
        n_exact_readable = 0
        readable_acc_pct = 0.0
        macro_cer = 0.0
        total_edit_distance = 0
        total_gt_chars = 0
        micro_cer = 0.0

    # 3. All-In Accuracy (counting unreadable and missed plates as failures)
    n_all_in = len(all_in_records)
    n_exact_all_in = sum(1 for r in all_in_records if r["success"])
    all_in_acc_pct = round(n_exact_all_in / n_all_in * 100.0, 2) if n_all_in else 0.0

    # 4. Cluster Bootstrap 95% Confidence Interval (Readable)
    cluster_readable_groups = defaultdict(list)
    for r in evaluated_readable_records:
        cluster_readable_groups[r["cluster_id"]].append(r)

    cluster_readable_list = list(cluster_readable_groups.values())
    n_readable_clusters = len(cluster_readable_list)

    rng = np.random.RandomState(seed)
    boot_rates = []
    if n_readable_clusters > 0 and n_readable > 0:
        for _ in range(bootstrap_draws):
            sample_indices = rng.choice(n_readable_clusters, size=n_readable_clusters, replace=True)
            sample_plates = [p for i in sample_indices for p in cluster_readable_list[i]]
            if sample_plates:
                rate = sum(1 for p in sample_plates if p["is_exact"]) / len(sample_plates)
                boot_rates.append(rate * 100.0)

    if boot_rates:
        ci_low_readable = round(float(np.percentile(boot_rates, 2.5)), 2)
        ci_high_readable = round(float(np.percentile(boot_rates, 97.5)), 2)
    else:
        ci_low_readable = ci_high_readable = readable_acc_pct

    # Plain Wilson Interval (Readable)
    if n_readable > 0:
        z = 1.959963984540054
        p_hat = n_exact_readable / n_readable
        denom = 1.0 + z**2 / n_readable
        center = (p_hat + z**2 / (2 * n_readable)) / denom
        margin = (z * math.sqrt(p_hat * (1 - p_hat) / n_readable + z**2 / (4 * n_readable**2))) / denom
        wilson_ci_readable = [round(max(0.0, center - margin) * 100.0, 2), round(min(1.0, center + margin) * 100.0, 2)]
    else:
        wilson_ci_readable = [0.0, 0.0]

    # Cluster Bootstrap 95% CI (All-In)
    cluster_all_in_groups = defaultdict(list)
    for r in all_in_records:
        cluster_all_in_groups[r["cluster_id"]].append(r)

    cluster_all_in_list = list(cluster_all_in_groups.values())
    n_all_in_clusters = len(cluster_all_in_list)

    boot_all_in_rates = []
    if n_all_in_clusters > 0 and n_all_in > 0:
        for _ in range(bootstrap_draws):
            sample_indices = rng.choice(n_all_in_clusters, size=n_all_in_clusters, replace=True)
            sample_plates = [p for i in sample_indices for p in cluster_all_in_list[i]]
            if sample_plates:
                rate = sum(1 for p in sample_plates if p["success"]) / len(sample_plates)
                boot_all_in_rates.append(rate * 100.0)

    if boot_all_in_rates:
        ci_low_all_in = round(float(np.percentile(boot_all_in_rates, 2.5)), 2)
        ci_high_all_in = round(float(np.percentile(boot_all_in_rates, 97.5)), 2)
    else:
        ci_low_all_in = ci_high_all_in = all_in_acc_pct

    # 5. Legacy ANPR Comparison Rates
    legacy_acc_pct = round(legacy_exact_matches / legacy_comparable * 100.0, 2) if legacy_comparable else 0.0
    model_legacy_acc_pct = round(model_legacy_exact_matches / legacy_comparable * 100.0, 2) if legacy_comparable else 0.0

    # 6. Wrong Predictions
    wrong_records = [
        {
            "index": r["index"],
            "file": r["file"],
            "cluster_id": r["cluster_id"],
            "gt_text": r["gt_text"],
            "pred_text": r["pred_text"],
            "edit_distance": r["edit_distance"],
            "cer": round(r["cer"], 4),
            "substitutions": r["substitutions"],
            "insertions": r["insertions"],
            "deletions": r["deletions"],
            "lighting": r["lighting"],
            "notes": r["notes"],
        }
        for r in evaluated_readable_records if not r["is_exact"]
    ]

    report = {
        "csv_path": str(csv_path),
        "sha256_hash": csv_hash,
        "sample_summary": {
            "total_frames": total_samples,
            "status_breakdown": dict(status_counts),
            "readable_plates": n_readable,
            "partially_readable": status_counts.get("partially_readable", 0),
            "unreadable_plates": status_counts.get("unreadable", 0),
            "no_plate_frames": status_counts.get("no_plate", 0),
            "multiple_plates_frames": status_counts.get("multiple_plates", 0),
            "wrong_crop_detections": wrong_crop_count,
            "unsupported_characters_plates": unsupported_chars_count,
        },
        "detection_evaluation": {
            "true_plates_visible": true_plates_visible,
            "detected_plates": detected_plates_count,
            "missed_detections": missed_detection_count,
            "wrong_crop_detections": wrong_crop_count,
            "detection_recall_pct": det_recall_pct,
            "numerator": detected_plates_count,
            "denominator": true_plates_visible,
        },
        "readable_plate_accuracy": {
            "exact_matches": n_exact_readable,
            "total_readable": n_readable,
            "accuracy_pct": readable_acc_pct,
            "numerator": n_exact_readable,
            "denominator": n_readable,
            "wilson_ci_95": wilson_ci_readable,
            "cluster_bootstrap_ci_95": [ci_low_readable, ci_high_readable],
            "total_clusters": n_readable_clusters,
        },
        "all_in_accuracy": {
            "exact_matches": n_exact_all_in,
            "total_eval": n_all_in,
            "accuracy_pct": all_in_acc_pct,
            "numerator": n_exact_all_in,
            "denominator": n_all_in,
            "cluster_bootstrap_ci_95": [ci_low_all_in, ci_high_all_in],
            "total_clusters": n_all_in_clusters,
            "definition": "Exact matches / (Readable + Partially Readable + Unreadable + Missed Detections)",
        },
        "character_error_rates": {
            "macro_cer": macro_cer,
            "micro_cer": micro_cer,
            "total_edit_distance": total_edit_distance,
            "total_gt_characters": total_gt_chars,
        },
        "error_taxonomy": {
            "substitutions": total_substitutions,
            "insertions": total_insertions,
            "deletions": total_deletions,
            "cross_vehicle_association_mistakes": cross_vehicle_associations,
            "detection_misses": missed_detection_count,
            "wrong_crop_detections": wrong_crop_count,
            "unsupported_characters_plates": unsupported_chars_count,
        },
        "legacy_anpr_comparison": {
            "comparable_arabic_samples": legacy_comparable,
            "legacy_vs_human_exact_matches": legacy_exact_matches,
            "legacy_vs_human_accuracy_pct": legacy_acc_pct,
            "model_vs_human_exact_matches": n_exact_readable,
            "model_vs_human_accuracy_pct": readable_acc_pct,
            "model_vs_legacy_exact_matches": model_legacy_exact_matches,
            "model_vs_legacy_agreement_pct": model_legacy_acc_pct,
            "latin_transliterated_legacy_count": latin_legacy_count,
            "latin_transliterated_samples": latin_legacy_samples,
        },
        "wrong_plates_list": wrong_records,
        "excluded_samples": excluded_samples,
    }

    if json_output_path:
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(description="Score labeled gate footage triage sheet (Mission 007A).")
    parser.add_argument("csv_path", type=Path, help="Path to completed_annotations.csv")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to save machine-readable JSON")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap random seed (default: 42)")
    parser.add_argument("--draws", type=int, default=10000, help="Bootstrap iterations (default: 10000)")
    args = parser.parse_args()

    report = score_labeled_csv(
        args.csv_path,
        json_output_path=args.json_out,
        bootstrap_draws=args.draws,
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print("MISSION 007A: LABELED GATE CCTV EVALUATION REPORT")
    print("=" * 70)
    print(f"File SHA-256:          {report['sha256_hash']}")
    print(f"Total Evaluated Rows:  {report['sample_summary']['total_frames']}")
    print(f"Status Breakdown:      {report['sample_summary']['status_breakdown']}")

    print("\n[1. Detection Evaluation]")
    det = report["detection_evaluation"]
    print(f"  True Visible Plates:   {det['true_plates_visible']}")
    print(f"  Detected Plates:       {det['detected_plates']}")
    print(f"  Missed Detections:     {det['missed_detections']}")
    print(f"  Wrong Crop Hallucinations: {det['wrong_crop_detections']}")
    print(f"  Detection Recall:      {det['detection_recall_pct']}% ({det['numerator']}/{det['denominator']})")

    print("\n[2. OCR Accuracy (Readable Plates Only)]")
    rec = report["readable_plate_accuracy"]
    print(f"  Exact Match Accuracy:  {rec['accuracy_pct']}% ({rec['numerator']}/{rec['denominator']})")
    print(f"  Plain Wilson 95% CI:   [{rec['wilson_ci_95'][0]}%, {rec['wilson_ci_95'][1]}%]")
    print(f"  Cluster Bootstrap CI:  [{rec['cluster_bootstrap_ci_95'][0]}%, {rec['cluster_bootstrap_ci_95'][1]}%] ({rec['total_clusters']} clusters)")
    print(f"  Macro CER:             {report['character_error_rates']['macro_cer']}")
    print(f"  Micro CER:             {report['character_error_rates']['micro_cer']}")

    print("\n[3. All-In Accuracy (Counting Unreadable & Missed as Failures)]")
    ai = report["all_in_accuracy"]
    print(f"  All-In Accuracy:       {ai['accuracy_pct']}% ({ai['numerator']}/{ai['denominator']})")
    print(f"  Cluster Bootstrap CI:  [{ai['cluster_bootstrap_ci_95'][0]}%, {ai['cluster_bootstrap_ci_95'][1]}%] ({ai['total_clusters']} clusters)")

    print("\n[4. Error Taxonomy]")
    err = report["error_taxonomy"]
    print(f"  Substitutions:         {err['substitutions']}")
    print(f"  Insertions:            {err['insertions']}")
    print(f"  Deletions:             {err['deletions']}")
    print(f"  Detection Misses:      {err['detection_misses']}")
    print(f"  Wrong Crop Errors:     {err['wrong_crop_detections']}")
    print(f"  Unsupported Char Plates: {err['unsupported_characters_plates']}")

    print("\n[5. Legacy ANPR Comparison]")
    leg = report["legacy_anpr_comparison"]
    print(f"  Comparable Arabic:     {leg['comparable_arabic_samples']}")
    print(f"  Legacy vs Human Acc:   {leg['legacy_vs_human_accuracy_pct']}% ({leg['legacy_vs_human_exact_matches']}/{leg['comparable_arabic_samples'] if leg['comparable_arabic_samples'] else 1})")
    print(f"  Model vs Human Acc:    {leg['model_vs_human_accuracy_pct']}% ({leg['model_vs_human_exact_matches']}/{leg['comparable_arabic_samples'] if leg['comparable_arabic_samples'] else 1})")
    print(f"  Model vs Legacy Agr:   {leg['model_vs_legacy_agreement_pct']}% ({leg['model_vs_legacy_exact_matches']}/{leg['comparable_arabic_samples'] if leg['comparable_arabic_samples'] else 1})")
    print(f"  Latin Transliterated:  {leg['latin_transliterated_legacy_count']} (reported separately)")

    if report["wrong_plates_list"]:
        print(f"\n--- Wrong Plates ({len(report['wrong_plates_list'])}) ---")
        for w in report["wrong_plates_list"]:
            print(f"  Row {w['index']}: GT='{w['gt_text']}' | PRED='{w['pred_text']}' -> EditDist={w['edit_distance']} (CER={w['cer']:.2f})")

    if report["excluded_samples"]:
        print(f"\n--- Excluded Samples ({len(report['excluded_samples'])}) ---")
        for ex in report["excluded_samples"][:10]:
            print(f"  Row {ex['index']} ({ex['status']}): {ex['file']} - {ex['reason']}")
        if len(report["excluded_samples"]) > 10:
            print(f"  ... and {len(report['excluded_samples']) - 10} more.")


if __name__ == "__main__":
    main()
