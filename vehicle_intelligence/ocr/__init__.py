"""Egyptian License Plate OCR modules and transliteration utilities."""

from vehicle_intelligence.ocr.mock_ocr import (
    ARABIC_TO_LATIN_PLATE_MAP,
    EASTERN_ARABIC_DIGITS_MAP,
    MockPlateOCR,
)

__all__ = [
    "ARABIC_TO_LATIN_PLATE_MAP",
    "EASTERN_ARABIC_DIGITS_MAP",
    "MockPlateOCR",
]
