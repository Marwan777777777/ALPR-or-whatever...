"""Tests for Egyptian plate OCR evaluation, Levenshtein distance, CER, and EMA."""

import pytest
from vehicle_intelligence.evaluation.ocr_eval import (
    EgyptianPlateTextNormalizer,
    GroundTruthPlateText,
    PlateOCREvaluator,
    PredictedPlateText,
)


def test_plate_normalizer():
    # Normalizes Eastern digits '١ ٢ ٣' to '1 2 3' and strips internal spaces
    raw = "س  ع  د  ١ ٢ ٣"
    norm = EgyptianPlateTextNormalizer.normalize_text(raw, convert_eastern_digits=True)
    assert norm == "سعد123"

    # Letter variations (أ -> ا, ى -> ي)
    raw_var = "أ ب ى"
    norm_var = EgyptianPlateTextNormalizer.normalize_text(raw_var)
    assert norm_var == "ابي"


def test_levenshtein_distance():
    assert PlateOCREvaluator.levenshtein_distance("", "") == 0
    assert PlateOCREvaluator.levenshtein_distance("kitten", "sitting") == 3
    assert PlateOCREvaluator.levenshtein_distance("سعد123", "سعد123") == 0
    assert PlateOCREvaluator.levenshtein_distance("سعد123", "سعد124") == 1
    assert PlateOCREvaluator.levenshtein_distance("سعد123", "") == 6


def test_ocr_evaluator_exact_and_partial_matches():
    gts = [
        GroundTruthPlateText(plate_id="p1", raw_text="س ع د ١ ٢ ٣"),
        GroundTruthPlateText(plate_id="p2", raw_text="ط ن ر ٤ ٥ ٦"),
        GroundTruthPlateText(plate_id="p3", raw_text="ب ج د ٧ ٨ ٩"),
    ]
    preds = [
        # p1: exact match
        PredictedPlateText(plate_id="p1", predicted_text="سعد 123", confidence=0.95),
        # p2: 1 character substitution ('طنر457' instead of 'طنر456')
        PredictedPlateText(plate_id="p2", predicted_text="طنر 457", confidence=0.88),
        # p3: unreadable prediction
        PredictedPlateText(plate_id="p3", predicted_text=None, is_unreadable=True),
    ]

    evaluator = PlateOCREvaluator(normalize_eastern_digits=True)
    metrics = evaluator.evaluate(gts, preds)

    assert metrics.total_evaluated_plates == 3
    assert metrics.exact_matches_count == 1
    assert pytest.approx(metrics.exact_match_accuracy, 0.01) == 0.3333
    assert metrics.unreadable_predictions_count == 1
    assert metrics.missing_predictions_count == 0

    # Check per-plate scores
    p1_score = metrics.per_plate_scores["p1"]
    assert p1_score.exact_match is True
    assert p1_score.cer == 0.0
    assert p1_score.normalized_edit_distance == 1.0

    p2_score = metrics.per_plate_scores["p2"]
    assert p2_score.exact_match is False
    assert p2_score.edit_distance == 1
    assert pytest.approx(p2_score.cer, 0.01) == 0.1667 # 1 / 6

    p3_score = metrics.per_plate_scores["p3"]
    assert p3_score.exact_match is False
    assert p3_score.cer == 1.0
    assert p3_score.normalized_edit_distance == 0.0


def test_ocr_evaluator_missing_predictions():
    gts = [GroundTruthPlateText(plate_id="p1", raw_text="س ع د 123")]
    preds = [] # missing prediction

    evaluator = PlateOCREvaluator()
    metrics = evaluator.evaluate(gts, preds)

    assert metrics.missing_predictions_count == 1
    assert metrics.exact_matches_count == 0
    assert metrics.exact_match_accuracy == 0.0
    assert metrics.mean_character_error_rate == 1.0
