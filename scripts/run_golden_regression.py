"""Run golden image regression evaluation and output structured JSON report."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import cv2
import torch

from vehicle_intelligence.pipeline.alpr_pipeline import EndToEndALPRPipeline


def run_golden_regression(
    output_path: Path = Path("artifacts/missions/mission_014/golden_image_regression.json"),
    device: str | None = None,
) -> bool:
    golden_dir = Path("tests/golden")
    expectations_file = golden_dir / "expectations.json"

    if not expectations_file.exists():
        print(f"Error: {expectations_file} not found.")
        return False

    detector_weights = Path("artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt")
    char_weights = Path("artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt")

    if not detector_weights.exists() or not char_weights.exists():
        print("Error: Required model checkpoints are missing.")
        return False

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Initializing ALPR Pipeline on device: {device}")
    pipeline = EndToEndALPRPipeline(
        detector_weights=detector_weights,
        char_weights=char_weights,
        device=device,
    )

    expectations = json.loads(expectations_file.read_text(encoding="utf-8"))
    print(f"Loaded {len(expectations)} golden samples from {expectations_file}")

    results = []
    perfect_matches = 0
    total_samples = len(expectations)

    t0 = time.time()
    for item in expectations:
        filename = item["filename"]
        img_path = golden_dir / "images" / filename
        if not img_path.exists():
            print(f"Missing golden image: {img_path}")
            return False

        im = cv2.imread(str(img_path))
        if im is None:
            print(f"Failed to read image: {img_path}")
            return False

        preds = pipeline.read_plate_image(im, source_id=filename)
        detected = len(preds) > 0
        pred_text = preds[0].plate_text if detected else ""
        pred_conf = preds[0].plate_confidence if detected else 0.0
        pred_digits = preds[0].digits if detected else ""
        pred_letters = preds[0].letters if detected else ""

        expected_text = item["expected_plate_text"]
        text_match = pred_text.replace(" ", "") == expected_text.replace(" ", "")
        if text_match:
            perfect_matches += 1

        results.append({
            "filename": filename,
            "detected": detected,
            "expected_text": expected_text,
            "predicted_text": pred_text,
            "text_match": text_match,
            "expected_confidence": item.get("plate_confidence", 0.0),
            "predicted_confidence": round(pred_conf, 4),
            "predicted_digits": pred_digits,
            "predicted_letters": pred_letters,
            "char_count": len(preds[0].per_character_confidences) if detected else 0,
        })

    elapsed_s = time.time() - t0
    accuracy = (perfect_matches / total_samples) * 100.0 if total_samples > 0 else 0.0

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": device,
        "detector_weights": str(detector_weights),
        "char_weights": str(char_weights),
        "total_samples": total_samples,
        "perfect_matches": perfect_matches,
        "accuracy_percent": round(accuracy, 2),
        "elapsed_seconds": round(elapsed_s, 3),
        "avg_ms_per_image": round((elapsed_s / total_samples) * 1000, 2),
        "all_passed": perfect_matches == total_samples,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGolden regression completed: {perfect_matches}/{total_samples} passed ({accuracy:.1f}%) in {elapsed_s:.2f}s")
    print(f"Report saved to: {output_path}")

    return perfect_matches == total_samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Golden Image Regression")
    parser.add_argument("--device", type=str, default=None, help="Target device (cpu, cuda:0, etc.)")
    parser.add_argument("--output", type=str, default="artifacts/missions/mission_014/golden_image_regression.json", help="Output path")
    args = parser.parse_args()

    success = run_golden_regression(output_path=Path(args.output), device=args.device)
    sys.exit(0 if success else 1)
