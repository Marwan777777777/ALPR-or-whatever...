"""Egyptian License Plate Reading Order and String Reconstruction.

Defines the explicit spatial mapping from 2D character bounding boxes to canonical plate strings.

Layout Rules for Standard Egyptian Plates:
1. Digits partition (Left side, x < ~0.52):
   - Classes 17-26 (Eastern Arabic numerals ٠-٩).
   - Ordered Left-to-Right (ascending x-center) matching standard mathematical digit significance:
     x_1 < x_2 < ... < x_m -> "٧١٢٦"
2. Letters partition (Right side, x > ~0.52):
   - Classes 0-16 (Arabic letters).
   - Ordered Right-to-Left (descending x-center) matching standard Arabic reading direction:
     x_1 > x_2 > ... > x_k -> "س ط ن"
3. Combined Canonical String:
   - "{letters} {digits}" (e.g. "س ط ن ٧١٢٦") or "{digits} {letters}".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from vehicle_intelligence.data.audit import EALPR_CHARACTER_CLASSES
from vehicle_intelligence.ocr.mock_ocr import EASTERN_ARABIC_DIGITS_MAP


@dataclass
class CharacterBox:
    """A detected or ground-truth character bounding box on a plate crop."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float = 1.0

    @property
    def is_digit(self) -> bool:
        return 17 <= self.class_id <= 26

    @property
    def is_letter(self) -> bool:
        return 0 <= self.class_id <= 16

    @property
    def character(self) -> str:
        return EALPR_CHARACTER_CLASSES.get(self.class_id, "?")

    @property
    def western_digit(self) -> str:
        char = self.character
        return EASTERN_ARABIC_DIGITS_MAP.get(char, char)


class EgyptianPlateReadingOrder:
    """Explicit plate reading order resolver for Egyptian license plates."""

    @classmethod
    def sort_digits_ltr(cls, digit_boxes: List[CharacterBox]) -> List[CharacterBox]:
        """Sorts digits strictly from Left to Right (ascending x_center)."""
        return sorted(digit_boxes, key=lambda b: b.x_center)

    @classmethod
    def sort_letters_rtl(cls, letter_boxes: List[CharacterBox]) -> List[CharacterBox]:
        """Sorts Arabic letters strictly from Right to Left (descending x_center)."""
        return sorted(letter_boxes, key=lambda b: b.x_center, reverse=True)

    @classmethod
    def sort_letters_ltr(cls, letter_boxes: List[CharacterBox]) -> List[CharacterBox]:
        """Sorts Arabic letters Left to Right (ascending x_center)."""
        return sorted(letter_boxes, key=lambda b: b.x_center)

    @classmethod
    def reconstruct_plate(
        cls,
        character_boxes: List[CharacterBox],
        letters_rtl: bool = True,
        prefix_letters: bool = True,
        convert_to_western_digits: bool = False,
    ) -> Tuple[str, str, str]:
        """Reconstructs the full plate text from unordered character boxes.

        Args:
            character_boxes: List of character boxes on the plate crop.
            letters_rtl: If True, orders letters Right-to-Left (Arabic standard).
            prefix_letters: If True, formats as '{letters} {digits}', else '{digits} {letters}'.
            convert_to_western_digits: If True, converts Eastern Arabic numerals to 0-9.

        Returns:
            Tuple of (full_plate_string, letters_string, digits_string).
        """
        if not character_boxes:
            return "", "", ""

        # Separate digits and letters
        digits = [b for b in character_boxes if b.is_digit]
        letters = [b for b in character_boxes if b.is_letter]

        # Sort digits LTR
        sorted_digits = cls.sort_digits_ltr(digits)
        if convert_to_western_digits:
            digits_str = "".join([b.western_digit for b in sorted_digits])
        else:
            digits_str = "".join([b.character for b in sorted_digits])

        # Sort letters
        if letters_rtl:
            sorted_letters = cls.sort_letters_rtl(letters)
        else:
            sorted_letters = cls.sort_letters_ltr(letters)
        letters_str = "".join([b.character for b in sorted_letters])

        # Format combined string
        if letters_str and digits_str:
            if prefix_letters:
                full_text = f"{letters_str} {digits_str}"
            else:
                full_text = f"{digits_str} {letters_str}"
        else:
            full_text = letters_str or digits_str

        return full_text, letters_str, digits_str
