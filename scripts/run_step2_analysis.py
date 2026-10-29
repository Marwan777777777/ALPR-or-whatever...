import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.core.domain import BoundingBox
from vehicle_intelligence.ocr.reading_order import (
    CharacterBox,
    EgyptianPlateReadingOrder,
)
from scripts.audit_end_to_end_and_ocr import (
    parse_character_boxes_from_yolo,
    parse_gt_characters_from_file,
    levenshtein_distance,
)
from scripts.tune_confidence_and_coverage import compute_plate_confidence


def evaluate_split_e2e(
    split_name: str,
    det_model: YOLO,
    char_model: YOLO,
    agnostic_nms: bool,
    conf_thresh: float = 0.25,
) -> List[dict]:
    ealpr_root = Path(r"C:\Users\MARWAN\plate_project\images_raw\EALPR-master")
    manifest = json.loads(Path("artifacts/splits/cleaned_split_manifest.json").read_text(encoding="utf-8"))
    stems = manifest[f"{split_name}_vehicles"]

    v_dir = ealpr_root / "EALPR Vechicles dataset" / "Vehicles"
    vl_dir = ealpr_root / "EALPR Vechicles dataset" / "Vehicles Labeling"
    cl_dir = ealpr_root / "EALPR- LP characters dataset" / "Characters Labeling"

    results = []

    for stem in stems:
        img_path = v_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = v_dir / f"{stem}.jpeg"
        if not img_path.exists():
            continue

        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]

        # 1. Ground truth boxes
        v_lbl = vl_dir / f"{stem}.txt"
        gt_boxes = []
        if v_lbl.exists():
            for line in v_lbl.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    xc, yc, bw, bh = [float(p) for p in parts[1:5]]
                    gt_boxes.append(BoundingBox(
                        x1=(xc - bw / 2.0) * w,
                        y1=(yc - bh / 2.0) * h,
                        x2=(xc + bw / 2.0) * w,
                        y2=(yc + bh / 2.0) * h,
                    ))

        # 2. GT characters
        c_files = sorted(list(cl_dir.glob(f"{stem}_license_plate_*.txt")))
        gt_char_map = {}
        for c_file in c_files:
            try:
                pidx = int(c_file.stem.split("_license_plate_")[-1]) - 1
            except ValueError:
                pidx = 0
            char_boxes = parse_gt_characters_from_file(c_file)
            txt, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(char_boxes)
            gt_char_map[pidx] = txt.replace(" ", "")

        # 3. Predict plates
        res = det_model.predict(source=str(img_path), conf=conf_thresh, imgsz=640, device=0, verbose=False)[0]
        detected_crops = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w, bx2), min(h, by2)
                if bx2 > bx1 and by2 > by1:
                    detected_crops.append({
                        "box": BoundingBox(x1=bx1, y1=by1, x2=bx2, y2=by2),
                        "crop": im[by1:by2, bx1:bx2],
                        "conf": float(b.conf[0]),
                    })

        # Match detected crops to GT boxes
        matched_gt = [False] * len(gt_boxes)
        detected_crops.sort(key=lambda c: c["conf"], reverse=True)
        gt_to_pred = {}
        for p_idx, p_crop in enumerate(detected_crops):
            best_iou = 0.0
            best_g = -1
            for g_idx, g_box in enumerate(gt_boxes):
                if not matched_gt[g_idx]:
                    iou = p_crop["box"].iou(g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_g = g_idx
            if best_iou >= 0.50 and best_g >= 0:
                matched_gt[best_g] = True
                gt_to_pred[best_g] = p_crop

        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx not in gt_char_map:
                continue
            gt_text = gt_char_map[g_idx]
            p_crop = gt_to_pred.get(g_idx)

            if p_crop is not None:
                c_res = char_model.predict(
                    source=p_crop["crop"],
                    conf=conf_thresh,
                    iou=0.70,
                    agnostic_nms=agnostic_nms,
                    imgsz=320,
                    verbose=False,
                )[0]
                pred_chars = parse_character_boxes_from_yolo(c_res, conf_thresh=conf_thresh)
                pred_full, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(pred_chars)
                pred_text = pred_full.replace(" ", "")

                dist = levenshtein_distance(gt_text, pred_text)
                is_exact = (pred_text == gt_text and len(gt_text) > 0)
                char_confs = [c.confidence for c in pred_chars if c.confidence is not None]
                plate_conf = compute_plate_confidence(p_crop["conf"], char_confs)

                results.append({
                    "vehicle": stem,
                    "plate_idx": g_idx + 1,
                    "gt_text": gt_text,
                    "pred_text": pred_text,
                    "is_exact": is_exact,
                    "edit_distance": dist,
                    "plate_conf": plate_conf,
                    "det_conf": p_crop["conf"],
                    "status": "DETECTED",
                })
            else:
                dist = len(gt_text)
                results.append({
                    "vehicle": stem,
                    "plate_idx": g_idx + 1,
                    "gt_text": gt_text,
                    "pred_text": "",
                    "is_exact": False,
                    "edit_distance": dist,
                    "plate_conf": 0.0,
                    "det_conf": 0.0,
                    "status": "MISSED_DETECTION",
                })

    return results


def wilson_interval(n_success: int, n_total: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n_total == 0:
        return 0.0, 0.0
    z = 1.959963984540054  # 95% standard normal quantile
    p_hat = n_success / n_total
    denom = 1.0 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2))) / denom
    return round(max(0.0, center - margin) * 100.0, 2), round(min(1.0, center + margin) * 100.0, 2)


def main():
    print("Loading models for Step 2...")
    det_model = YOLO("artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt")
    char_model = YOLO("artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt")

    # Part 2(a): Val E2E exact match with agnostic_nms=False vs True
    print("\n--- Part 2(a): Val E2E with agnostic_nms OFF vs ON ---")
    val_off = evaluate_split_e2e("val", det_model, char_model, agnostic_nms=False)
    val_on = evaluate_split_e2e("val", det_model, char_model, agnostic_nms=True)

    n_val = len(val_off)
    exact_off = sum(1 for r in val_off if r["is_exact"])
    exact_on = sum(1 for r in val_on if r["is_exact"])
    ed_off = sum(r["edit_distance"] for r in val_off)
    ed_on = sum(r["edit_distance"] for r in val_on)
    tot_chars = sum(len(r["gt_text"]) for r in val_off)

    print(f"Val Eligible Plates: {n_val}")
    print(f"agnostic_nms=False (default): Exact Match = {exact_off}/{n_val} ({exact_off/n_val*100:.2f}%), CER = {ed_off/tot_chars:.4f} (total edit dist={ed_off})")
    print(f"agnostic_nms=True:           Exact Match = {exact_on}/{n_val} ({exact_on/n_val*100:.2f}%), CER = {ed_on/tot_chars:.4f} (total edit dist={ed_on})")

    # Part 2(c): Per-plate edit distances for the 25 wrong plates on test
    print("\n--- Part 2(c): Per-Plate Edit Distances for 25 Wrong Test Plates ---")
    test_results = evaluate_split_e2e("test", det_model, char_model, agnostic_nms=False)
    wrong_test = [r for r in test_results if not r["is_exact"]]
    total_test_ed = sum(r["edit_distance"] for r in test_results)
    print(f"Total wrong test plates: {len(wrong_test)}, Total edit distance across wrong plates: {sum(r['edit_distance'] for r in wrong_test)}")
    for idx, r in enumerate(wrong_test, 1):
        print(f"  {idx:02d}. Vehicle={r['vehicle']}, Plate={r['plate_idx']}: GT='{r['gt_text']}' | PRED='{r['pred_text']}' -> EditDist={r['edit_distance']} ({r['status']})")

    # Part 2(b): Cluster Bootstrap for Test E2E Exact Match
    print("\n--- Part 2(b): 95% CIs for Test E2E Exact Match (Cluster Bootstrap vs Wilson) ---")
    # Define clusters in test.
    # Group test results by cluster. If vehicle clustering:
    # We group by vehicle or by the connected components.
    # Let's map each test result to its vehicle or cluster:
    # Notice the user prompt: "resample the 161 test clusters, 10,000 draws, seed 42"
    # Let's see how the 161 clusters are structured:
    # If 161 test clusters partition the 302 test plates, each cluster has a list of results.
    # Let's group by vehicle prefix or the 161 clusters:
    clusters_file = Path("artifacts/splits/test_clusters.json")
    if clusters_file.exists():
        raw_clusters = json.loads(clusters_file.read_text(encoding="utf-8"))
    else:
        raw_clusters = [[v] for v in set(r["vehicle"] for r in test_results)]

    # Assign each test result to its cluster
    vehicle_to_cluster = {}
    for c_idx, c in enumerate(raw_clusters):
        for v in c:
            vehicle_to_cluster[v] = c_idx

    cluster_plates = defaultdict(list)
    for r in test_results:
        c_idx = vehicle_to_cluster.get(r["vehicle"], r["vehicle"])
        cluster_plates[c_idx].append(r)

    active_clusters = list(cluster_plates.values())
    print(f"Number of active test clusters containing eligible plates: {len(active_clusters)}")

    # Run Cluster Bootstrap: 10,000 draws, seed 42
    rng = np.random.RandomState(42)
    B = 10000
    n_c = len(active_clusters)
    boot_accs = []
    for _ in range(B):
        idx_sample = rng.choice(n_c, size=n_c, replace=True)
        sample_plates = [p for i in idx_sample for p in active_clusters[i]]
        if len(sample_plates) > 0:
            acc = sum(1 for p in sample_plates if p["is_exact"]) / len(sample_plates)
            boot_accs.append(acc * 100.0)

    boot_ci_low = np.percentile(boot_accs, 2.5)
    boot_ci_high = np.percentile(boot_accs, 97.5)

    n_test_exact = sum(1 for r in test_results if r["is_exact"])
    n_test_total = len(test_results)
    wilson_low, wilson_high = wilson_interval(n_test_exact, n_test_total)

    print(f"Test E2E Exact Match: {n_test_exact}/{n_test_total} ({n_test_exact/n_test_total*100:.2f}%)")
    print(f"  Plain Wilson 95% CI:             [{wilson_low:.2f}%, {wilson_high:.2f}%]")
    print(f"  Cluster Bootstrap 95% CI:        [{boot_ci_low:.2f}%, {boot_ci_high:.2f}%] (10,000 draws, seed 42)")

    # Part 2(d): Coverage-vs-Accuracy Table (thresholds 0.50 to 0.95 in steps of 0.05)
    print("\n--- Part 2(d): Coverage-vs-Accuracy Table (Val & Test) ---")
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    print(f"{'Threshold':<10} | {'Val Cov (%)':<12} | {'Val Acc (%)':<12} | {'Test Cov (%)':<12} | {'Test Acc (%)':<12}")
    print("-" * 65)

    coverage_table = []
    for t in thresholds:
        # Val
        val_acc_plates = [r for r in val_off if r["plate_conf"] >= t]
        val_cov = len(val_acc_plates) / len(val_off) * 100.0 if len(val_off) else 0.0
        val_acc = sum(1 for r in val_acc_plates if r["is_exact"]) / len(val_acc_plates) * 100.0 if len(val_acc_plates) else 0.0

        # Test
        test_acc_plates = [r for r in test_results if r["plate_conf"] >= t]
        test_cov = len(test_acc_plates) / len(test_results) * 100.0 if len(test_results) else 0.0
        test_acc = sum(1 for r in test_acc_plates if r["is_exact"]) / len(test_acc_plates) * 100.0 if len(test_acc_plates) else 0.0

        print(f"{t:<10.2f} | {val_cov:<12.2f} | {val_acc:<12.2f} | {test_cov:<12.2f} | {test_acc:<12.2f}")
        coverage_table.append({
            "threshold": t,
            "val_coverage_pct": round(val_cov, 2),
            "val_accuracy_pct": round(val_acc, 2),
            "val_accepted_count": len(val_acc_plates),
            "test_coverage_pct": round(test_cov, 2),
            "test_accuracy_pct": round(test_acc, 2),
            "test_accepted_count": len(test_acc_plates),
        })

    # Save summary
    out_file = Path("artifacts/evaluation_ocr/step2_fixes_summary.json")
    out_file.write_text(json.dumps({
        "val_agnostic_nms_comparison": {
            "val_eligible_plates": n_val,
            "agnostic_nms_false": {
                "exact_matches": exact_off,
                "exact_match_pct": round(exact_off / n_val * 100.0, 2),
                "total_edit_distance": ed_off,
                "mean_cer": round(ed_off / tot_chars, 4),
            },
            "agnostic_nms_true": {
                "exact_matches": exact_on,
                "exact_match_pct": round(exact_on / n_val * 100.0, 2),
                "total_edit_distance": ed_on,
                "mean_cer": round(ed_on / tot_chars, 4),
            },
        },
        "wrong_test_plates_25": wrong_test,
        "ci_comparison": {
            "exact_matches": n_test_exact,
            "total_plates": n_test_total,
            "exact_match_pct": round(n_test_exact / n_test_total * 100.0, 2),
            "wilson_95_ci": [wilson_low, wilson_high],
            "cluster_bootstrap_95_ci": [round(boot_ci_low, 2), round(boot_ci_high, 2)],
        },
        "coverage_vs_accuracy_table": coverage_table,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved Step 2 results to {out_file}")


if __name__ == "__main__":
    main()
