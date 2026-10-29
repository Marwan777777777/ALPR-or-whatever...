"""Egyptian license plate OCR evaluation contract: CER, Exact Match Accuracy, and Edit Distance."""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from vehicle_intelligence.ocr.mock_ocr import EASTERN_ARABIC_DIGITS_MAP


class GroundTruthPlateText(BaseModel):
    """Ground truth text annotation for a license plate."""
    plate_id: str
    raw_text: str
    arabic_letters: Optional[str] = None
    arabic_digits: Optional[str] = None
    is_valid: bool = True


class PredictedPlateText(BaseModel):
    """Model predicted text for a license plate crop."""
    plate_id: str
    predicted_text: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_unreadable: bool = False
    needs_review: bool = False


class PlateMatchScore(BaseModel):
    """Evaluation result for an individual plate."""
    plate_id: str
    ground_truth: str
    predicted: str
    exact_match: bool
    edit_distance: int
    cer: float
    normalized_edit_distance: float
    is_unreadable_prediction: bool


class OCRAggregateMetrics(BaseModel):
    """Aggregate OCR metrics across all evaluated plates."""
    total_evaluated_plates: int
    exact_matches_count: int
    exact_match_accuracy: float = Field(..., description="Percentage of plates with 100% exact character match")
    mean_character_error_rate: float = Field(..., description="Mean CER = edit_distance / gt_length")
    mean_normalized_edit_distance: float = Field(..., description="Mean 1 - (dist / max(len_gt, len_pred))")
    unreadable_predictions_count: int
    missing_predictions_count: int
    per_plate_scores: Dict[str, PlateMatchScore] = Field(default_factory=dict)


class EgyptianPlateTextNormalizer:
    """Normalizes Arabic letters and numbers for consistent comparison."""

    # Normalization map for common Arabic OCR ambiguities
    ARABIC_LETTER_NORMALIZATION = {
        "أ": "ا", "إ": "ا", "آ": "ا",
        "ى": "ي",
        "ھ": "ه",
        "ة": "ه",
    }

    @classmethod
    def normalize_text(cls, text: Optional[str], convert_eastern_digits: bool = True) -> str:
        if not text:
            return ""

        # Normalize unicode and strip combining marks (including hamza, shaddah, tanween)
        decomposed = unicodedata.normalize("NFKD", text.strip())
        stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")

        # Replace letter variants
        for src, dst in cls.ARABIC_LETTER_NORMALIZATION.items():
            stripped = stripped.replace(src, dst)

        # Optionally convert Eastern Arabic numerals (١, ٢, ...) to Western digits (1, 2, ...)
        if convert_eastern_digits:
            for east, west in EASTERN_ARABIC_DIGITS_MAP.items():
                stripped = stripped.replace(east, west)

        # Remove extra whitespace and spaces inside plate code
        tokens = stripped.split()
        return "".join(tokens)


class PlateOCREvaluator:
    """Evaluates predicted plate text against ground truth."""

    def __init__(self, normalize_eastern_digits: bool = True):
        self.normalize_digits = normalize_eastern_digits

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Computes standard character-level Levenshtein edit distance."""
        if s1 == s2:
            return 0
        if not s1:
            return len(s2)
        if not s2:
            return len(s1)

        dp = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)
        for i in range(len(s1) + 1):
            dp[i, 0] = i
        for j in range(len(s2) + 1):
            dp[0, j] = j

        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i, j] = min(
                    dp[i - 1, j] + 1,      # deletion
                    dp[i, j - 1] + 1,      # insertion
                    dp[i - 1, j - 1] + cost  # substitution
                )

        return int(dp[len(s1), len(s2)])

    def evaluate_pair(
        self,
        gt_raw: str,
        pred_raw: Optional[str],
        plate_id: str,
        is_unreadable: bool = False
    ) -> PlateMatchScore:
        gt_norm = EgyptianPlateTextNormalizer.normalize_text(gt_raw, self.normalize_digits)
        pred_norm = EgyptianPlateTextNormalizer.normalize_text(pred_raw, self.normalize_digits) if not is_unreadable else ""

        if is_unreadable or not pred_norm:
            dist = len(gt_norm)
            exact = (len(gt_norm) == 0 and not is_unreadable)
            cer = 1.0 if len(gt_norm) > 0 else 0.0
            ned = 0.0 if len(gt_norm) > 0 else 1.0
        else:
            dist = self.levenshtein_distance(gt_norm, pred_norm)
            exact = (gt_norm == pred_norm)
            cer = dist / max(1, len(gt_norm))
            max_len = max(len(gt_norm), len(pred_norm))
            ned = max(0.0, 1.0 - (dist / max(1, max_len)))

        return PlateMatchScore(
            plate_id=plate_id,
            ground_truth=gt_norm,
            predicted=pred_norm,
            exact_match=exact,
            edit_distance=dist,
            cer=round(cer, 4),
            normalized_edit_distance=round(ned, 4),
            is_unreadable_prediction=is_unreadable,
        )

    def evaluate(
        self,
        ground_truths: List[GroundTruthPlateText],
        predictions: List[PredictedPlateText]
    ) -> OCRAggregateMetrics:
        gt_map = {gt.plate_id: gt for gt in ground_truths}
        pred_map = {p.plate_id: p for p in predictions}

        all_ids = sorted(list(gt_map.keys()))
        per_plate: Dict[str, PlateMatchScore] = {}

        exact_matches = 0
        cer_list: List[float] = []
        ned_list: List[float] = []
        unreadable_count = 0
        missing_count = 0

        for pid in all_ids:
            gt = gt_map[pid]
            pred = pred_map.get(pid)

            if pred is None:
                missing_count += 1
                score = self.evaluate_pair(
                    gt_raw=gt.raw_text,
                    pred_raw=None,
                    plate_id=pid,
                    is_unreadable=True
                )
            else:
                if pred.is_unreadable:
                    unreadable_count += 1
                score = self.evaluate_pair(
                    gt_raw=gt.raw_text,
                    pred_raw=pred.predicted_text,
                    plate_id=pid,
                    is_unreadable=pred.is_unreadable
                )

            per_plate[pid] = score
            if score.exact_match:
                exact_matches += 1
            cer_list.append(score.cer)
            ned_list.append(score.normalized_edit_distance)

        total_plates = len(all_ids)
        ema = (exact_matches / total_plates) if total_plates > 0 else 0.0
        mean_cer = float(np.mean(cer_list)) if cer_list else 0.0
        mean_ned = float(np.mean(ned_list)) if ned_list else 0.0

        return OCRAggregateMetrics(
            total_evaluated_plates=total_plates,
            exact_matches_count=exact_matches,
            exact_match_accuracy=round(ema, 4),
            mean_character_error_rate=round(mean_cer, 4),
            mean_normalized_edit_distance=round(mean_ned, 4),
            unreadable_predictions_count=unreadable_count,
            missing_predictions_count=missing_count,
            per_plate_scores=per_plate,
        )
