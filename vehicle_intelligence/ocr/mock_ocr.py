"""Pluggable OCR interfaces and deterministic mock OCR engine for Egyptian plates."""

from __future__ import annotations

import re
from typing import Optional
import numpy as np

from vehicle_intelligence.core.domain import PlateTextResult
from vehicle_intelligence.core.interfaces import BasePlateOCR


# Standard mapping of Egyptian Arabic plate letters to Latin transliteration
ARABIC_TO_LATIN_PLATE_MAP = {
    "أ": "A", "ب": "B", "ج": "G", "د": "D", "ر": "R", "س": "S", "ص": "C",
    "ط": "T", "ع": "E", "ف": "F", "ق": "Q", "ك": "K", "ل": "L", "م": "M",
    "ن": "N", "ه": "H", "و": "W", "ي": "Y"
}

# Eastern Arabic numerals to Western Arabic digits
EASTERN_ARABIC_DIGITS_MAP = {
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9"
}


class MockPlateOCR(BasePlateOCR):
    """Deterministic OCR engine for Egyptian license plates.

    Generates structured Arabic letters, numbers, and Latin transcriptions,
    with configurable confidence and review flags.
    """

    def __init__(
        self,
        default_letters: str = "س ع د",
        default_digits: str = "١ ٢ ٣ ٤",
        confidence: float = 0.91,
        force_needs_review: bool = False
    ):
        self._letters = default_letters
        self._digits = default_digits
        self._confidence = confidence
        self._force_needs_review = force_needs_review

    @property
    def engine_name(self) -> str:
        return "mock_egyptian_ocr_engine_v1"

    def read(self, plate_crop: np.ndarray) -> PlateTextResult:
        if plate_crop is None or plate_crop.size == 0:
            return PlateTextResult(
                raw_text="",
                confidence=0.0,
                needs_review=True,
                review_reasons=["Empty or corrupted plate crop provided"]
            )

        # Transliterate Arabic letters
        letters_tokens = [c for c in self._letters if c in ARABIC_TO_LATIN_PLATE_MAP]
        latin_letters = " ".join([ARABIC_TO_LATIN_PLATE_MAP[c] for c in letters_tokens])

        # Normalize digits
        normalized_digits = "".join([EASTERN_ARABIC_DIGITS_MAP.get(c, c) for c in self._digits if c.isdigit() or c in EASTERN_ARABIC_DIGITS_MAP])

        latin_transcription = f"{latin_letters} {normalized_digits}".strip()
        raw_text = f"{self._letters} {self._digits}".strip()

        needs_review = self._force_needs_review or (self._confidence < 0.75)
        review_reasons = []
        if needs_review:
            review_reasons.append("Confidence below acceptance threshold or flagged for human review")

        return PlateTextResult(
            raw_text=raw_text,
            arabic_letters=self._letters,
            arabic_digits=self._digits,
            latin_transcription=latin_transcription,
            confidence=self._confidence,
            needs_review=needs_review,
            review_reasons=review_reasons
        )
