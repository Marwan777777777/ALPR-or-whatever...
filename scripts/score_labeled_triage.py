#!/usr/bin/env python3
"""Score Labeled Gate Footage Triage Sheet (Mission 007 Step 4).

Reads filled label_sheet.csv:
- Evaluates Exact Match accuracy
- Computes Macro CER and Micro CER
- Evaluates Exact Match split by lighting condition
- Computes Cluster-Aware 95% Confidence Interval (grouped by 30-second vehicle time bucket)
- Lists all discordant predictions with per-plate Levenshtein edit distance
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.audit_end_to_end_and_ocr import levenshtein_distance


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
        # 30-second epoch bucket
        epoch_bucket = int(ts.timestamp() // time_bucket_sec)
        return f"time_bucket_{epoch_bucket}"
    # Fallback to filename prefix
    parts = filename.split("_")
    return parts[0] if parts else filename


def score_labeled_csv(csv_path: Path, bootstrap_draws: int = 10000, seed: int = 42) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    evaluated_records = []
    skipped_empty = 0

    for idx, r in enumerate(rows, 1):
        fn = r.get("file", "").strip()
        pred_raw = r.get("predicted_text", "").strip()
        gt_raw = r.get("true_text", "").strip()
        visible = r.get("plate_visible_to_eye", "").strip().lower()
        lighting = r.get("lighting", "").strip().lower()
        notes = r.get("notes", "").strip()

        # If true_text is empty, skip or warn
        if not gt_raw:
            skipped_empty += 1
            continue

        # Normalize whitespace
        pred_norm = pred_raw.replace(" ", "")
        gt_norm = gt_raw.replace(" ", "")

        dist = levenshtein_distance(gt_norm, pred_norm)
        is_exact = (pred_norm == gt_norm and len(gt_norm) > 0)
        gt_len = max(1, len(gt_norm))
        plate_cer = dist / gt_len

        cluster_id = extract_cluster_id(fn)

        evaluated_records.append({
            "index": idx,
            "file": fn,
            "cluster_id": cluster_id,
            "gt_text": gt_raw,
            "pred_text": pred_raw,
            "gt_norm": gt_norm,
            "pred_norm": pred_norm,
            "is_exact": is_exact,
            "edit_distance": dist,
            "cer": plate_cer,
            "lighting": lighting or "unspecified",
            "visible": visible,
            "notes": notes,
        })

    total_eval = len(evaluated_records)
    if total_eval == 0:
        return {
            "error": "No labeled rows with non-empty 'true_text' found.",
            "total_rows_in_file": len(rows),
            "skipped_empty": skipped_empty,
        }

    # 1. Exact Match
    n_exact = sum(1 for r in evaluated_records if r["is_exact"])
    exact_acc_pct = round(n_exact / total_eval * 100.0, 2)

    # 2. CER (Macro and Micro)
    # Macro CER: mean of individual plate CERs
    macro_cer = round(float(np.mean([r["cer"] for r in evaluated_records])), 4)
    # Micro CER: total edit distance across all plates divided by total ground truth characters
    total_edit_distance = sum(r["edit_distance"] for r in evaluated_records)
    total_gt_chars = sum(len(r["gt_norm"]) for r in evaluated_records)
    micro_cer = round(total_edit_distance / max(1, total_gt_chars), 4)

    # 3. Exact Match split by Lighting
    lighting_groups = defaultdict(list)
    for r in evaluated_records:
        lighting_groups[r["lighting"]].append(r)

    lighting_breakdown = {}
    for l_key, l_records in lighting_groups.items():
        l_exact = sum(1 for r in l_records if r["is_exact"])
        lighting_breakdown[l_key] = {
            "count": len(l_records),
            "exact_matches": l_exact,
            "exact_match_pct": round(l_exact / len(l_records) * 100.0, 2),
            "mean_cer": round(float(np.mean([r["cer"] for r in l_records])), 4),
        }

    # 4. Cluster-Aware Confidence Interval
    cluster_groups = defaultdict(list)
    for r in evaluated_records:
        cluster_groups[r["cluster_id"]].append(r)

    cluster_list = list(cluster_groups.values())
    n_clusters = len(cluster_list)

    rng = np.random.RandomState(seed)
    boot_rates = []
    for _ in range(bootstrap_draws):
        sample_indices = rng.choice(n_clusters, size=n_clusters, replace=True)
        sample_plates = [p for i in sample_indices for p in cluster_list[i]]
        if sample_plates:
            rate = sum(1 for p in sample_plates if p["is_exact"]) / len(sample_plates)
            boot_rates.append(rate * 100.0)

    if boot_rates:
        ci_low = round(float(np.percentile(boot_rates, 2.5)), 2)
        ci_high = round(float(np.percentile(boot_rates, 97.5)), 2)
    else:
        ci_low = ci_high = exact_acc_pct

    # Plain Wilson Interval
    z = 1.959963984540054
    p_hat = n_exact / total_eval
    denom = 1.0 + z**2 / total_eval
    center = (p_hat + z**2 / (2 * total_eval)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / total_eval + z**2 / (4 * total_eval**2))) / denom
    wilson_ci = [round(max(0.0, center - margin) * 100.0, 2), round(min(1.0, center + margin) * 100.0, 2)]

    # 5. List of wrong predictions with edit distances
    wrong_records = [
        {
            "index": r["index"],
            "file": r["file"],
            "gt_text": r["gt_text"],
            "pred_text": r["pred_text"],
            "edit_distance": r["edit_distance"],
            "cer": round(r["cer"], 4),
            "lighting": r["lighting"],
            "cluster_id": r["cluster_id"],
        }
        for r in evaluated_records if not r["is_exact"]
    ]

    report = {
        "csv_path": str(csv_path),
        "total_labeled_plates": total_eval,
        "skipped_empty_rows": skipped_empty,
        "exact_match": {
            "matches": n_exact,
            "total": total_eval,
            "accuracy_pct": exact_acc_pct,
        },
        "character_error_rates": {
            "macro_cer": macro_cer,
            "micro_cer": micro_cer,
            "total_edit_distance": total_edit_distance,
            "total_gt_characters": total_gt_chars,
            "definitions": {
                "macro_cer": "Arithmetic mean of individual per-plate CERs (dist / len(gt))",
                "micro_cer": "Sum of all edit distances across all plates / Sum of all ground truth characters",
            },
        },
        "exact_match_by_lighting": lighting_breakdown,
        "confidence_intervals_95": {
            "total_clusters": n_clusters,
            "cluster_grouping_rule": "30-second arrival epoch bucket from filename timestamp",
            "cluster_bootstrap_ci_pct": [ci_low, ci_high],
            "wilson_ci_pct": wilson_ci,
            "bootstrap_draws": bootstrap_draws,
            "random_seed": seed,
        },
        "wrong_plates_count": len(wrong_records),
        "wrong_plates_list": wrong_records,
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Score labeled Egyptian ALPR triage sheet.")
    parser.add_argument("csv_path", type=Path, help="Path to filled label_sheet.csv")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap random seed (default: 42)")
    parser.add_argument("--draws", type=int, default=10000, help="Bootstrap iterations (default: 10000)")
    args = parser.parse_args()

    report = score_labeled_csv(args.csv_path, bootstrap_draws=args.draws, seed=args.seed)

    print("\n" + "=" * 60)
    print("LABELED TRIAGE EVALUATION REPORT")
    print("=" * 60)
    if "error" in report:
        print(f"ERROR: {report['error']}")
        return

    print(f"Total Evaluated Plates: {report['total_labeled_plates']} (Skipped empty: {report['skipped_empty_rows']})")
    print(f"Exact Match Accuracy:   {report['exact_match']['accuracy_pct']}% ({report['exact_match']['matches']}/{report['exact_match']['total']})")
    print(f"Macro CER:              {report['character_error_rates']['macro_cer']}")
    print(f"Micro CER:              {report['character_error_rates']['micro_cer']}")
    print(f"Plain Wilson 95% CI:    [{report['confidence_intervals_95']['wilson_ci_pct'][0]}%, {report['confidence_intervals_95']['wilson_ci_pct'][1]}%]")
    print(f"Cluster Bootstrap 95%:  [{report['confidence_intervals_95']['cluster_bootstrap_ci_pct'][0]}%, {report['confidence_intervals_95']['cluster_bootstrap_ci_pct'][1]}%] ({report['confidence_intervals_95']['total_clusters']} clusters)")

    print("\n--- Exact Match Breakdown by Lighting ---")
    for l_key, l_data in report["exact_match_by_lighting"].items():
        print(f"  {l_key:<16}: {l_data['exact_match_pct']:>6.2f}% ({l_data['exact_matches']}/{l_data['count']}) | Macro CER: {l_data['mean_cer']}")

    print(f"\n--- Wrong Plates ({len(report['wrong_plates_list'])}) ---")
    for w in report["wrong_plates_list"]:
        print(f"  Row {w['index']}: GT='{w['gt_text']}' | PRED='{w['pred_text']}' -> EditDist={w['edit_distance']} (CER={w['cer']:.2f}, {w['lighting']})")


if __name__ == "__main__":
    main()
