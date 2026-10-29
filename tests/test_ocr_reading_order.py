"""Unit tests for Egyptian Plate Reading Order and spatial string reconstruction."""

import pytest
from vehicle_intelligence.ocr.reading_order import (
    CharacterBox,
    EgyptianPlateReadingOrder,
)


def test_reading_order_standard_4digits_3letters():
    """Test ground-truth plate 0001: 4 digits ('٧', '١', '٢', '٦') and 3 letters ('ن', 'ط', 'س')."""
    # Create boxes in scrambled/unordered sequence
    boxes = [
        CharacterBox(class_id=5, x_center=0.866, y_center=0.63, width=0.14, height=0.26),  # 'س'
        CharacterBox(class_id=19, x_center=0.354, y_center=0.60, width=0.09, height=0.35), # '٢'
        CharacterBox(class_id=13, x_center=0.593, y_center=0.62, width=0.11, height=0.28), # 'ن'
        CharacterBox(class_id=24, x_center=0.159, y_center=0.58, width=0.11, height=0.31), # '٧'
        CharacterBox(class_id=7, x_center=0.728, y_center=0.62, width=0.12, height=0.32),  # 'ط'
        CharacterBox(class_id=23, x_center=0.459, y_center=0.61, width=0.09, height=0.34), # '٦'
        CharacterBox(class_id=18, x_center=0.256, y_center=0.60, width=0.06, height=0.32), # '١'
    ]

    # Reconstruct with RTL letters (standard Arabic: س then ط then ن)
    full_text, letters, digits = EgyptianPlateReadingOrder.reconstruct_plate(
        boxes, letters_rtl=True, prefix_letters=True
    )
    assert digits == "٧١٢٦"
    assert letters == "س ط ن".replace(" ", "")  # "سطن"
    assert full_text == "سطن ٧١٢٦"

    # Reconstruct with Western digits conversion
    full_text_w, letters_w, digits_w = EgyptianPlateReadingOrder.reconstruct_plate(
        boxes, letters_rtl=True, prefix_letters=True, convert_to_western_digits=True
    )
    assert digits_w == "7126"
    assert full_text_w == "سطن 7126"


def test_reading_order_letters_ltr_vs_rtl():
    """Verifies that letters_rtl=False respects strictly ascending x order."""
    boxes = [
        CharacterBox(class_id=13, x_center=0.593, y_center=0.62, width=0.11, height=0.28), # 'ن'
        CharacterBox(class_id=7, x_center=0.728, y_center=0.62, width=0.12, height=0.32),  # 'ط'
        CharacterBox(class_id=5, x_center=0.866, y_center=0.63, width=0.14, height=0.26),  # 'س'
    ]

    # RTL: descending x -> س then ط then ن
    _, letters_rtl, _ = EgyptianPlateReadingOrder.reconstruct_plate(boxes, letters_rtl=True)
    assert letters_rtl == "سطن"

    # LTR: ascending x -> ن then ط then س
    _, letters_ltr, _ = EgyptianPlateReadingOrder.reconstruct_plate(boxes, letters_rtl=False)
    assert letters_ltr == "نطس"


def test_reading_order_empty_or_partial_plate():
    """Tests empty inputs and digits-only or letters-only edge cases."""
    assert EgyptianPlateReadingOrder.reconstruct_plate([]) == ("", "", "")

    # Digits only
    d_boxes = [
        CharacterBox(class_id=18, x_center=0.2, y_center=0.5, width=0.1, height=0.3),  # '١'
        CharacterBox(class_id=20, x_center=0.4, y_center=0.5, width=0.1, height=0.3),  # '٣'
    ]
    full_d, l_d, d_d = EgyptianPlateReadingOrder.reconstruct_plate(d_boxes)
    assert full_d == "١٣"
    assert l_d == ""
    assert d_d == "١٣"
