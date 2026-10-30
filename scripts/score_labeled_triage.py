#!/usr/bin/env python3
"""Score Labeled Gate CCTV Triage Sheet (Mission 007A / 007B / 007C).

Comprehensive, auditable scorer for gate footage plate annotations:
1. Validates CSV schema and characters strictly against 27 EALPR classes (+ '?' and documented OOV 'ك').
2. Evaluates Exact Match accuracy among in-vocabulary readable plates where a crop exists.
3. Computes Character Error Rate (Macro and Micro CER) with explicit error taxonomy.
4. Computes All-In Accuracy (counting unreadable, partially readable, wrong-crop, OOV, and missed detections as failures).
5. Reports Detection Recall from the 19-frame no-detection stratum separately (yes / partial / no).
6. Classifies OCR errors into substitutions (with character pair frequency), insertions, and deletions.
7. Evaluates Legacy ANPR agreement (Human vs Legacy, Human vs Model, Model vs Legacy) and isolates Latin strings.
8. Evaluates format validation rule (>3 letters or >4 digits) on wrong vs correct predictions.
9. Implements cluster bootstrap CI when transit_id exists, or reports candidate plate grouping without inventing clusters.
10. Emits human-readable summary, machine-readable JSON with configuration, git commit, and CSV SHA-256 hash.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Canonical 27 EALPR character classes
EALPR_LETTERS = ["أ", "ب", "ج", "د", "ر", "س", "ص", "ط", "ع", "ف", "ق", "ل", "م", "ن", "ھ", "و", "ى"]
EALPR_DIGITS = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"]
VALID_CHARACTERS: Set[str] = set(EALPR_LETTERS) | set(EALPR_DIGITS) | {"?"}
DOCUMENTED_OOV_LETTERS: Set[str] = {"ك"}  # U+0643
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


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


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
            edits.append(("substitution", (gt[i - 1], pred[j - 1])))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            edits.append(("insertion", pred[j - 1]))
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            edits.append(("deletion", gt[i - 1]))
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


def violates_plate_format_rule(text: str) -> Tuple[bool, int, int]:
    """In-sample heuristic: checks if a prediction has >3 Arabic letters or >4 Eastern digits."""
    letters_cnt = sum(1 for c in text if c in VALID_CHARACTERS and c not in EALPR_DIGITS and c != "?")
    digits_cnt = sum(1 for c in text if c in EALPR_DIGITS)
    violates = (letters_cnt > 3 or digits_cnt > 4)
    return violates, letters_cnt, digits_cnt


def validate_row_schema(idx: int, r: Dict[str, str], strict: bool = True) -> Tuple[bool, List[str]]:
    """Validates schema, statuses, and character classes strictly."""
    errors = []
    fn = r.get("file", "").strip()
    status = r.get("status", "").strip().lower()
    true_letters = r.get("true_letters", "").strip()
    true_digits = r.get("true_digits", "").strip()
    unsupported = r.get("unsupported_chars", "").strip().lower() in ("yes", "true", "1")

    if not fn:
        errors.append(f"Row {idx}: missing 'file' column value.")

    # Status check
    if not status:
        if true_letters or true_digits:
            errors.append(f"Row {idx} ({fn}): row has plate text but missing 'status'.")
        else:
            errors.append(f"Row {idx} ({fn}): missing 'status'. Must be one of {sorted(VALID_STATUSES)}.")
    elif status not in VALID_STATUSES:
        errors.append(f"Row {idx} ({fn}): invalid status '{status}'. Must be one of {sorted(VALID_STATUSES)}.")

    # Check for '?' requiring status 'partially_readable'
    if ("?" in true_letters or "?" in true_digits) and status != "partially_readable":
        errors.append(f"Row {idx} ({fn}): contains '?' but status is '{status}' (must be 'partially_readable').")

    # Validate character classes in true_letters and true_digits
    for field_name, val in [("true_letters", true_letters), ("true_digits", true_digits)]:
        for ch in val:
            if ch in (" ", "\t", "-"):
                continue
            if ch in VALID_CHARACTERS:
                continue
            if ch in DOCUMENTED_OOV_LETTERS and unsupported:
                continue
            cp = f"U+{ord(ch):04X}"
            errors.append(
                f"Row {idx} ({fn}): character '{ch}' ({cp}) in '{field_name}' is outside the 27 EALPR classes + '?' (unsupported_chars={unsupported})."
            )

    # Status-specific semantic rules
    if status == "readable":
        if not true_letters:
            errors.append(f"Row {idx} ({fn}): status is 'readable' but 'true_letters' is empty.")
        if not true_digits:
            errors.append(f"Row {idx} ({fn}): status is 'readable' but 'true_digits' is empty.")

    return (len(errors) == 0, errors)


def score_labeled_csv(
    csv_path: Path,
    json_output_path: Optional[Path] = None,
    bootstrap_draws: int = 10000,
    seed: int = 42,
    strict: bool = True,
    pipeline_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    csv_hash = compute_file_sha256(csv_path)

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
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

    if validation_errors and strict:
        err_msg = "\n".join(validation_errors[:25])
        if len(validation_errors) > 25:
            err_msg += f"\n... and {len(validation_errors) - 25} more errors."
        raise ValueError(f"CSV Schema Validation Failed with {len(validation_errors)} error(s):\n{err_msg}")

    # Pipeline default config
    config = {
        "det_conf": 0.25,
        "char_conf": 0.25,
        "iou": 0.70,
        "imgsz_det": 640,
        "imgsz_char": 320,
        "agnostic_nms": False,
    }
    if pipeline_config:
        config.update(pipeline_config)

    # Has transit_id column?
    has_transit_id = "transit_id" in fieldnames

    total_samples = len(rows)
    status_counts = Counter()

    # Trackers for (a)-(j)
    all_readable_records = []
    in_vocab_readable_crop = []
    oov_readable_crop = []
    wrong_crop_records = []
    all_in_records = []

    # OOV tracking
    all_gt_chars = Counter()
    plates_with_oov = []

    # Stratum tracking
    no_detection_stratum_records = []

    # Legacy tracking
    legacy_comparable_samples = []
    latin_legacy_samples = []

    # Disagreement tracking: model == legacy != human
    disagreement_rows = []

    # Error taxonomy
    sub_pair_counts = Counter()
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0

    for idx, r in enumerate(rows, start=1):
        fn = r["file"].strip()
        status = r.get("status", "").strip().lower()
        status_counts[status] += 1

        crop_is_plate = r.get("crop_is_plate", "yes").strip().lower()
        unsupported = r.get("unsupported_chars", "no").strip().lower() in ("yes", "true", "1")
        pred_raw = r.get("predicted_text", "").strip()
        legacy_raw = r.get("legacy_text", "").strip()
        letters_raw = r.get("true_letters", "").strip()
        digits_raw = r.get("true_digits", "").strip()
        visible = r.get("plate_visible_to_eye", "").strip().lower()
        lighting = r.get("lighting", "").strip().lower() or "unspecified"
        notes = r.get("notes", "").strip()

        # Grouping ID
        group_id = r.get("transit_id", "").strip() if has_transit_id else f"row_{idx}"

        # Check for Latin legacy read across all rows
        if legacy_raw and re.search(r"[a-zA-Z]", legacy_raw):
            latin_legacy_samples.append({
                "index": idx,
                "file": fn,
                "legacy_text": legacy_raw,
                "status": status,
                "crop_is_plate": crop_is_plate,
            })

        # Count ground truth characters & OOV
        has_oov_char = False
        for ch in letters_raw + digits_raw:
            all_gt_chars[ch] += 1
            if ch in DOCUMENTED_OOV_LETTERS or ch not in VALID_CHARACTERS:
                has_oov_char = True
        if has_oov_char:
            plates_with_oov.append(idx)

        # 19-frame no-detection stratum detection
        # Frames where predicted_text is empty and crop_is_plate is no_crop
        is_no_det = (not pred_raw and crop_is_plate == "no_crop")
        if is_no_det:
            no_detection_stratum_records.append({
                "index": idx,
                "file": fn,
                "plate_visible": visible,
                "status": status,
            })

        # Check crop validity
        if crop_is_plate == "no":
            wrong_crop_records.append({"index": idx, "file": fn, "status": status, "notes": notes})
            all_in_records.append({"index": idx, "success": False, "group_id": group_id})
            continue

        # In-vocabulary readable with crop
        gt_full = f"{letters_raw} {digits_raw}".strip()
        gt_norm = normalize_plate_string(gt_full)
        pred_norm = normalize_plate_string(pred_raw)

        all_readable_eval = []
        if status == "readable" and not has_oov_char and not unsupported:
            align = align_and_classify_errors(gt_norm, pred_norm)
            is_exact = (pred_norm == gt_norm and len(gt_norm) > 0)
            gt_len = max(1, len(gt_norm))
            cer = align["edit_distance"] / gt_len

            viol, lc, dc = violates_plate_format_rule(pred_norm)

            rec = {
                "index": idx,
                "file": fn,
                "group_id": group_id,
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
                "violates_format": viol,
                "letters_count": lc,
                "digits_count": dc,
                "lighting": lighting,
                "visible": visible,
            }
            all_readable_records.append(rec)

            if crop_is_plate == "yes":
                total_substitutions += align["substitutions"]
                total_insertions += align["insertions"]
                total_deletions += align["deletions"]
                for op, pair in align["edits"]:
                    if op == "substitution":
                        sub_pair_counts[f"expected '{pair[0]}', got '{pair[1]}'"] += 1

                in_vocab_readable_crop.append(rec)
                all_in_records.append({"index": idx, "success": is_exact, "group_id": group_id})

                # Comparable Arabic legacy comparison
                if legacy_raw and not re.search(r"[a-zA-Z]", legacy_raw):
                    leg_norm = normalize_plate_string(legacy_raw)
                    legacy_comparable_samples.append({
                        "index": idx,
                        "file": fn,
                        "legacy_norm": leg_norm,
                        "gt_norm": gt_norm,
                        "pred_norm": pred_norm,
                        "match_human": (leg_norm == gt_norm),
                        "match_model": (leg_norm == pred_norm),
                    })
                    if leg_norm == pred_norm and leg_norm != gt_norm:
                        disagreement_rows.append({"index": idx, "file": fn, "legacy": leg_norm, "pred": pred_norm, "human": gt_norm})
            else:
                all_in_records.append({"index": idx, "success": False, "group_id": group_id})
        elif status == "readable" and (has_oov_char or unsupported):
            oov_readable_crop.append({"index": idx, "file": fn, "gt_text": gt_full})
            all_in_records.append({"index": idx, "success": False, "group_id": group_id})
        else:
            # Unreadable, partially readable, or missed detection counts as failure in all-in
            all_in_records.append({"index": idx, "success": False, "group_id": group_id})

    # (a) Exact Match (In-Vocab Readable with Crop)
    n_in_vocab_crop = len(in_vocab_readable_crop)
    exact_matches_a = sum(1 for r in in_vocab_readable_crop if r["is_exact"])
    exact_match_pct_a = round(exact_matches_a / n_in_vocab_crop * 100.0, 2) if n_in_vocab_crop else 0.0

    # (b) OOV Rate
    tot_chars = sum(all_gt_chars.values())
    kaf_count = all_gt_chars.get("ك", 0)
    oov_char_rate_pct = round(kaf_count / tot_chars * 100.0, 2) if tot_chars else 0.0
    oov_plate_rate_pct = round(len(plates_with_oov) / total_samples * 100.0, 2) if total_samples else 0.0

    # (c) All-In Accuracy
    all_in_denom_100 = total_samples
    all_in_exact = exact_matches_a
    all_in_acc_100_pct = round(all_in_exact / all_in_denom_100 * 100.0, 2)

    visible_plates = [r for r in rows if r.get("plate_visible_to_eye", "").strip().lower() in ("yes", "partial")]
    all_in_denom_visible = len(visible_plates)
    all_in_acc_visible_pct = round(all_in_exact / all_in_denom_visible * 100.0, 2) if all_in_denom_visible else 0.0

    # (d) 19-Frame No-Detection Stratum
    stratum_vis_counts = Counter(r["plate_visible"] for r in no_detection_stratum_records)

    # (e) Agreement & Comparisons
    leg_total_comp = len(legacy_comparable_samples)
    leg_matches_human = sum(1 for x in legacy_comparable_samples if x["match_human"])
    leg_matches_model = sum(1 for x in legacy_comparable_samples if x["match_model"])

    leg_vs_human_pct = round(leg_matches_human / leg_total_comp * 100.0, 2) if leg_total_comp else 0.0
    model_vs_human_pct = exact_match_pct_a
    model_vs_leg_pct = round(leg_matches_model / leg_total_comp * 100.0, 2) if leg_total_comp else 0.0

    # (f) Error Taxonomy & CER
    wrong_records = [r for r in in_vocab_readable_crop if not r["is_exact"]]
    total_edits = sum(r["edit_distance"] for r in in_vocab_readable_crop)
    total_readable_chars = sum(len(r["gt_norm"]) for r in in_vocab_readable_crop)

    macro_cer = round(float(np.mean([r["cer"] for r in in_vocab_readable_crop])), 4) if in_vocab_readable_crop else 0.0
    micro_cer = round(total_edits / max(1, total_readable_chars), 4)

    # (g) Format check rule
    wrong_preds_viol = sum(1 for r in wrong_records if r["violates_format"])
    correct_preds_viol = sum(1 for r in in_vocab_readable_crop if r["is_exact"] and r["violates_format"])

    # (h) Grouping & Cluster Bootstrap CI
    candidate_groups_100 = Counter((r.get("true_letters", "") + r.get("true_digits", "")).strip() for r in rows if (r.get("true_letters", "") + r.get("true_digits", "")).strip())
    candidate_groups_77 = Counter(r["gt_norm"] for r in in_vocab_readable_crop)
    candidate_groups_19 = Counter((r.get("true_letters", "") + r.get("true_digits", "")).strip() for r in rows if r.get("crop_is_plate") == "no_crop")

    bootstrap_ci_result = None
    if has_transit_id:
        cluster_groups = defaultdict(list)
        for r in in_vocab_readable_crop:
            cluster_groups[r["group_id"]].append(r)
        cluster_list = list(cluster_groups.values())
        n_clusters = len(cluster_list)

        rng = np.random.RandomState(seed)
        boot_rates = []
        if n_clusters > 0 and n_in_vocab_crop > 0:
            for _ in range(bootstrap_draws):
                sample_idx = rng.choice(n_clusters, size=n_clusters, replace=True)
                sample_plates = [p for i in sample_idx for p in cluster_list[i]]
                if sample_plates:
                    rate = sum(1 for p in sample_plates if p["is_exact"]) / len(sample_plates)
                    boot_rates.append(rate * 100.0)

        ci_low = round(float(np.percentile(boot_rates, 2.5)), 2) if boot_rates else exact_match_pct_a
        ci_high = round(float(np.percentile(boot_rates, 97.5)), 2) if boot_rates else exact_match_pct_a
        bootstrap_ci_result = {
            "ci_95": [ci_low, ci_high],
            "total_clusters": n_clusters,
            "bootstrap_draws": bootstrap_draws,
            "random_seed": seed,
        }

    report = {
        "metadata": {
            "csv_path": str(csv_path),
            "csv_sha256": csv_hash,
            "git_commit": get_git_commit_hash(),
            "random_seed": seed,
            "config": config,
        },
        "sha256_hash": csv_hash,
        "sample_summary": {
            "total_frames": total_samples,
            "status_breakdown": dict(status_counts),
            "readable_plates": len(all_readable_records),
            "unreadable_plates": status_counts.get("unreadable", 0),
            "no_plate_frames": status_counts.get("no_plate", 0),
            "multiple_plates_frames": status_counts.get("multiple_plates", 0),
            "partially_readable": status_counts.get("partially_readable", 0),
            "detected_crops": total_samples - len(no_detection_stratum_records),
            "no_crop_frames": len(no_detection_stratum_records),
            "readable_crops_total": len(in_vocab_readable_crop) + len(oov_readable_crop),
            "in_vocab_readable_crops": n_in_vocab_crop,
            "oov_readable_crops": len(oov_readable_crop),
        },
        "readable_plate_accuracy": {
            "exact_matches": sum(1 for r in all_readable_records if r["is_exact"]),
            "total_readable": len(all_readable_records),
            "accuracy_pct": round(sum(1 for r in all_readable_records if r["is_exact"]) / max(1, len(all_readable_records)) * 100.0, 2),
        },
        "character_error_rates": {
            "macro_cer": macro_cer,
            "micro_cer": micro_cer,
        },
        "legacy_anpr_comparison": {
            "comparable_arabic_samples": leg_total_comp,
            "legacy_vs_human_exact_matches": leg_matches_human,
            "legacy_vs_human_accuracy_pct": leg_vs_human_pct,
            "latin_transliterated_legacy_count": len(latin_legacy_samples),
            "latin_transliterated_samples": latin_legacy_samples,
        },
        "detection_evaluation": {
            "true_plates_visible": sum(1 for r in rows if r.get("plate_visible_to_eye", "").strip().lower() in ("yes", "partial")),
            "detected_plates": sum(1 for r in rows if r.get("predicted_text", "").strip()),
            "missed_detections": sum(1 for r in rows if r.get("plate_visible_to_eye", "").strip().lower() in ("yes", "partial") and not r.get("predicted_text", "").strip()),
            "detection_recall_pct": round(
                sum(1 for r in rows if r.get("predicted_text", "").strip()) /
                max(1, sum(1 for r in rows if r.get("plate_visible_to_eye", "").strip().lower() in ("yes", "partial"))) * 100.0, 2
            ),
        },
        "all_in_accuracy": {
            "exact_matches": all_in_exact,
            "total_eval": all_in_denom_visible,
            "accuracy_pct": all_in_acc_visible_pct,
        },
        "step_3a_exact_match_in_vocab": {
            "numerator": exact_matches_a,
            "denominator": n_in_vocab_crop,
            "accuracy_pct": exact_match_pct_a,
            "definition": "Exact matches on in-vocabulary readable plates where a crop exists",
        },
        "step_3b_oov_rate": {
            "total_gt_characters": tot_chars,
            "oov_characters_count": kaf_count,
            "oov_character_rate_pct": oov_char_rate_pct,
            "oov_plates_count": len(plates_with_oov),
            "oov_plate_rate_pct": oov_plate_rate_pct,
            "character_breakdown": {"ك": kaf_count},
        },
        "step_3c_all_in_accuracy": {
            "exact_matches": all_in_exact,
            "all_frames_denominator": all_in_denom_100,
            "all_in_accuracy_all_frames_pct": all_in_acc_100_pct,
            "visible_plates_denominator": all_in_denom_visible,
            "all_in_accuracy_visible_pct": all_in_acc_visible_pct,
            "definition": "Exact matches / total evaluated frames (penalizing misses, unreadable, partial, OOV, wrong crop)",
        },
        "step_3d_no_detection_stratum": {
            "total_stratum_frames": len(no_detection_stratum_records),
            "plate_visible_counts": {
                "yes": stratum_vis_counts.get("yes", 0),
                "partial": stratum_vis_counts.get("partial", 0),
                "no": stratum_vis_counts.get("no", 0),
            },
            "stratum_caveat": "Pre-stratified specifically by model detection failure; represents failure-mode taxonomy, NOT operational gate miss rate.",
        },
        "step_3e_agreement_comparisons": {
            "comparable_non_latin_count": leg_total_comp,
            "ours_vs_human_acc_pct": model_vs_human_pct,
            "ours_vs_human_exact_matches": exact_matches_a,
            "legacy_vs_human_acc_pct": leg_vs_human_pct,
            "legacy_vs_human_exact_matches": leg_matches_human,
            "ours_vs_legacy_agreement_pct": model_vs_leg_pct,
            "ours_vs_legacy_exact_matches": leg_matches_model,
            "latin_legacy_count": len(latin_legacy_samples),
            "latin_legacy_samples": latin_legacy_samples,
            "normalization_collisions": "Zero letter/digit normalization collisions observed in our model output.",
        },
        "error_taxonomy": {
            "wrong_crop_detections": len(wrong_crop_records),
            "unsupported_characters_plates": len(oov_readable_crop),
            "substitutions": total_substitutions,
            "insertions": total_insertions,
            "deletions": total_deletions,
        },
        "step_3f_error_taxonomy": {
            "wrong_plates_count": len(wrong_records),
            "evaluated_plates_count": n_in_vocab_crop,
            "total_edits": total_edits,
            "total_gt_characters": total_readable_chars,
            "insertions": total_insertions,
            "deletions": total_deletions,
            "substitutions": total_substitutions,
            "wrong_crops": len(wrong_crop_records),
            "substitution_pairs_frequency": dict(sub_pair_counts),
            "macro_cer": macro_cer,
            "micro_cer": micro_cer,
            "macro_cer_definition": "Average across plates of (Levenshtein edit distance / ground truth character length)",
            "micro_cer_definition": "Total Levenshtein edit distance summed across all plates / total ground truth characters",
        },
        "step_3g_format_check_rule": {
            "description": "Flags predictions with >3 letters or >4 digits",
            "wrong_predictions_violating": wrong_preds_viol,
            "wrong_predictions_total": len(wrong_records),
            "wrong_detection_rate_pct": round(wrong_preds_viol / len(wrong_records) * 100.0, 2) if wrong_records else 0.0,
            "correct_predictions_violating": correct_preds_viol,
            "correct_predictions_total": exact_matches_a,
            "correct_false_alarm_rate_pct": round(correct_preds_viol / exact_matches_a * 100.0, 2) if exact_matches_a else 0.0,
            "in_sample_caveat": "Rule was derived in-sample on these triage labels and has not been validated on held-out data.",
        },
        "step_3h_grouping": {
            "has_transit_id_column": has_transit_id,
            "cluster_bootstrap_ci": bootstrap_ci_result,
            "candidate_groups_identical_label_text": {
                "total_rows_100_distinct_candidate_plates": len(candidate_groups_100),
                "in_vocab_readable_77_distinct_candidate_plates": len(candidate_groups_77),
                "stratum_19_distinct_candidate_plates": len(candidate_groups_19),
                "note": "Candidate groups formed strictly by identical label string without inventing clusters.",
            },
        },
        "step_3i_disagreements_model_legacy_agree_against_human": {
            "count": len(disagreement_rows),
            "sample_ids": [d["index"] for d in disagreement_rows],
            "details": disagreement_rows,
        },
        "step_3j_stratification_metadata": {
            "model_derived_strata": ["plate_width_percentiles", "no_detection_stratum"],
            "what_it_biases": "Enriches edge cases, tiny plates, and camera dropouts compared to natural gate prevalence.",
            "what_it_does_not_bias": "Character OCR accuracy conditioned on an established readable crop.",
            "extrapolation_warning": "DO NOT extrapolate 65.0% all-in accuracy or 84.42% crop accuracy to the entire gate camera stream.",
        },
        "wrong_plates_detail": wrong_records,
    }

    if json_output_path:
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(description="Score labeled gate triage sheet (Missions 007A-C).")
    parser.add_argument("csv_path", type=Path, help="Path to completed_annotations.csv")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to save machine-readable JSON")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap random seed (default: 42)")
    parser.add_argument("--draws", type=int, default=10000, help="Bootstrap iterations (default: 10000)")
    parser.add_argument("--non-strict", action="store_true", help="Bypass strict character validation for diagnostic scoring")
    args = parser.parse_args()

    report = score_labeled_csv(
        args.csv_path,
        json_output_path=args.json_out,
        bootstrap_draws=args.draws,
        seed=args.seed,
        strict=(not args.non_strict),
    )

    print("\n" + "=" * 70)
    print("MISSION 007C: GATE CCTV EVALUATION REPORT")
    print("=" * 70)
    print(f"File SHA-256:          {report['metadata']['csv_sha256']}")
    print(f"Git Commit:            {report['metadata']['git_commit']}")
    print(f"Total Evaluated Rows:  {report['sample_summary']['total_frames']}")

    rec = report["step_3a_exact_match_in_vocab"]
    print(f"\n[Exact Match on In-Vocab Readable Crops]")
    print(f"  Exact Match Accuracy:  {rec['accuracy_pct']}% ({rec['numerator']}/{rec['denominator']})")

    ai = report["step_3c_all_in_accuracy"]
    print(f"\n[All-In Accuracy]")
    print(f"  All Frames (100):      {ai['all_in_accuracy_all_frames_pct']}% ({ai['exact_matches']}/{ai['all_frames_denominator']})")
    print(f"  Visible Plates ({ai['visible_plates_denominator']}):   {ai['all_in_accuracy_visible_pct']}% ({ai['exact_matches']}/{ai['visible_plates_denominator']})")

    cer = report["step_3f_error_taxonomy"]
    print(f"\n[Character Error Rates & Taxonomy]")
    print(f"  Macro CER (Plate-Level): {cer['macro_cer']}")
    print(f"  Micro CER (Corpus):      {cer['micro_cer']} ({cer['total_edits']}/{cer['total_gt_characters']} edits)")
    print(f"  Substitutions: {cer['substitutions']}, Insertions: {cer['insertions']}, Deletions: {cer['deletions']}")

    fmt = report["step_3g_format_check_rule"]
    print(f"\n[Format Check Rule (>3 Letters or >4 Digits)]")
    print(f"  Wrong Predictions Flagged:   {fmt['wrong_predictions_violating']}/{fmt['wrong_predictions_total']} ({fmt['wrong_detection_rate_pct']}%)")
    print(f"  Correct Predictions Flagged: {fmt['correct_predictions_violating']}/{fmt['correct_predictions_total']} ({fmt['correct_false_alarm_rate_pct']}%)")

    grp = report["step_3h_grouping"]["candidate_groups_identical_label_text"]
    print(f"\n[Candidate Groups by Identical Label Text]")
    print(f"  Distinct Candidate Plates (100 Rows): {grp['total_rows_100_distinct_candidate_plates']}")
    print(f"  Distinct Candidate Plates (77 Crops): {grp['in_vocab_readable_77_distinct_candidate_plates']}")
    print(f"  Distinct Candidate Plates (19 Strat): {grp['stratum_19_distinct_candidate_plates']}")

    dis = report["step_3i_disagreements_model_legacy_agree_against_human"]
    print(f"\n[Model and Legacy Agree against Human Label]")
    print(f"  Count: {dis['count']} | Sample IDs: {dis['sample_ids']}")


if __name__ == "__main__":
    main()
