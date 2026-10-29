"""Regression test asserting pipeline accuracy on 20 representative golden test images (Mission 6 Part D)."""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import pytest

from vehicle_intelligence.pipeline.alpr_pipeline import EndToEndALPRPipeline


@pytest.fixture(scope="module")
def alpr_pipeline():
    detector_weights = Path("artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt")
    char_weights = Path("artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt")

    if not detector_weights.exists() or not char_weights.exists():
        pytest.skip("Model weights not available in artifacts directory.")

    return EndToEndALPRPipeline(
        detector_weights=detector_weights,
        char_weights=char_weights,
        device="cuda:0",
    )


def test_golden_dataset_regression(alpr_pipeline: EndToEndALPRPipeline):
    golden_dir = Path("tests/golden")
    expectations_file = golden_dir / "expectations.json"

    if not expectations_file.exists():
        pytest.skip("Golden expectations file missing.")

    expectations = json.loads(expectations_file.read_text(encoding="utf-8"))
    assert len(expectations) == 20, f"Expected 20 golden samples, found {len(expectations)}"

    failures = []
    for item in expectations:
        img_path = golden_dir / "images" / item["filename"]
        assert img_path.exists(), f"Missing golden image file: {img_path}"

        im = cv2.imread(str(img_path))
        assert im is not None, f"Failed to read golden image: {img_path}"

        results = alpr_pipeline.read_plate_image(im, source_id=item["filename"])
        assert len(results) > 0, f"No license plate detected on golden image: {item['filename']}"

        pred_text = results[0].plate_text.replace(" ", "")
        expected_text = item["expected_plate_text"].replace(" ", "")

        if pred_text != expected_text:
            failures.append({
                "filename": item["filename"],
                "expected": expected_text,
                "predicted": pred_text,
            })

    assert len(failures) == 0, f"Golden test regression detected! Failures: {failures}"
