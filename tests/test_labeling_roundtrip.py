"""Tests for Mission 007A: Labeling Round-Trip, UTF-8 BOM, Character Classes, and Scorer.

Covers:
- Byte-exact round-trip for Arabic letters (U+0623..U+0649) and Eastern Arabic digits (U+0660..U+0669)
- Character class enforcement (fail on characters outside 27-class set + '?')
- Wrong crop handling (crop_is_plate=no counted as detection error, not OCR error)
- Unsupported characters handling (unsupported_chars=yes counted separately, not OCR error)
- Legacy comparison normalization (ASCII digits -> Eastern Arabic, ه -> ھ, ي -> ى) and Latin strings
- Comprehensive synthetic fixtures covering:
  * Correct transcription
  * Character substitution
  * Insertion and deletion
  * Unreadable plate
  * Missed detection
  * Multiple plates in one frame
  * Correlated frames from same vehicle transit
  * Malformed CSV rows and invalid statuses
"""

import csv
import io
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest
from scripts.score_labeled_triage import (
    EALPR_LETTERS,
    EALPR_DIGITS,
    VALID_CHARACTERS,
    score_labeled_csv,
    normalize_plate_string,
    align_and_classify_errors,
)


def test_arabic_unicode_codepoints_documented():
    """Verify and document the 27 character classes and specific codepoints."""
    assert len(EALPR_LETTERS) == 17
    assert len(EALPR_DIGITS) == 10
    assert len(VALID_CHARACTERS) == 28  # 17 letters + 10 digits + '?'

    # Verify specific EALPR codepoints
    assert ord("ھ") == 0x06BE  # Heh Doachashmee (class 14)
    assert ord("ى") == 0x0649  # Alef Maksura (class 16)
    assert ord("أ") == 0x0623  # Alef with Hamza Above (class 0)
    assert ord("٠") == 0x0660  # Eastern Arabic Digit Zero (class 17)
    assert ord("٩") == 0x0669  # Eastern Arabic Digit Nine (class 26)


def test_byte_exact_roundtrip_with_bom(tmp_path: Path):
    """Write all 17 letters and 10 digits with UTF-8 BOM, re-import, and assert byte-exact equality."""
    csv_file = tmp_path / "completed_annotations.csv"

    # All 17 letters in reading order
    letters_str = "".join(EALPR_LETTERS)
    digits_str = "".join(EALPR_DIGITS)

    content = "file,legacy_text,predicted_text,crop_is_plate,unsupported_chars,status,true_letters,true_digits,plate_visible_to_eye,lighting,notes\n"
    # Row with all letters and digits
    row = f'20261015_010000_000_Gate_test_L90_C90.jpg,test,"{letters_str} {digits_str}",yes,no,readable,"{letters_str}","{digits_str}",yes,normal,all_chars\n'
    full_text = content + row

    # Write as UTF-8 with BOM (\ufeff)
    csv_file.write_text("\ufeff" + full_text, encoding="utf-8")

    report = score_labeled_csv(csv_file, bootstrap_draws=100)

    assert report["sample_summary"]["readable_plates"] == 1
    assert report["readable_plate_accuracy"]["exact_matches"] == 1
    assert report["readable_plate_accuracy"]["accuracy_pct"] == 100.0
    assert report["character_error_rates"]["macro_cer"] == 0.0

    # Read back raw bytes and assert UTF-8 BOM is present
    raw_bytes = csv_file.read_bytes()
    assert raw_bytes.startswith(b"\xef\xbb\xbf")

    # Assert letters and digits match byte-for-byte
    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = list(csv.DictReader(f))
        assert reader[0]["true_letters"] == letters_str
        assert reader[0]["true_digits"] == digits_str
        assert reader[0]["crop_is_plate"] == "yes"
        assert reader[0]["unsupported_chars"] == "no"


def test_invalid_character_outside_27_classes_fails(tmp_path: Path):
    """Scorer must reject characters outside the 27 classes with a clear error."""
    csv_file = tmp_path / "invalid_chars.csv"

    content = (
        "\ufefffile,legacy_text,predicted_text,crop_is_plate,unsupported_chars,status,true_letters,true_digits,plate_visible_to_eye,lighting,notes\n"
        '20261015_010000_000_Gate_test_L90_C90.jpg,test,سقج ٨٥٦٨,yes,no,readable,سقX,٨٥٦@,yes,normal,\n'
    )
    csv_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        score_labeled_csv(csv_file)

    err = str(excinfo.value)
    assert "outside the 27 EALPR classes" in err
    assert "'X'" in err or "'@'" in err


def test_wrong_crop_and_unsupported_chars_handling(tmp_path: Path):
    """Wrong crops (crop_is_plate=no) and unsupported chars must not be penalized as OCR errors."""
    csv_file = tmp_path / "edge_cases.csv"

    rows = [
        # Frame 1: Wrong crop (grille detected instead of plate) -> Detection error, excluded from OCR
        '20261015_010000_000_Gate_test1_L90_C90.jpg,سقج8568,سقج ٨٥٦٨,no,no,unreadable,,,yes,normal,detected_truck_grille',
        # Frame 2: Unsupported chars (e.g. Latin plate) -> Excluded from 27-class OCR evaluation
        '20261015_010001_000_Gate_test2_L90_C90.jpg,cd1234,سقج ٨٥٦٨,yes,yes,readable,سقج,١٢٣٤,yes,normal,diplomatic_latin_plate',
        # Frame 3: Normal readable plate
        '20261015_010002_000_Gate_test3_L90_C90.jpg,سقج8568,سقج ٨٥٦٨,yes,no,readable,سقج,٨٥٦٨,yes,normal,exact_match',
    ]
    header = "file,legacy_text,predicted_text,crop_is_plate,unsupported_chars,status,true_letters,true_digits,plate_visible_to_eye,lighting,notes\n"
    csv_file.write_text("\ufeff" + header + "\n".join(rows) + "\n", encoding="utf-8")

    report = score_labeled_csv(csv_file, bootstrap_draws=100)

    # Only Frame 3 is in readable 27-class OCR evaluation!
    assert report["sample_summary"]["readable_plates"] == 1
    assert report["readable_plate_accuracy"]["exact_matches"] == 1
    assert report["readable_plate_accuracy"]["accuracy_pct"] == 100.0

    # Frame 1 accounted as wrong crop error
    assert report["error_taxonomy"]["wrong_crop_detections"] == 1

    # Frame 2 accounted as unsupported characters
    assert report["error_taxonomy"]["unsupported_characters_plates"] == 1


def test_legacy_comparison_normalization_and_latin_strings(tmp_path: Path):
    """Legacy strings must normalize ASCII digits and Arabic letters; Latin strings reported separately."""
    csv_file = tmp_path / "legacy_test.csv"

    rows = [
        # Arabic legacy: سقج8568 -> normalizes ASCII 8568 to Eastern Arabic ٨٥٦٨
        '20261015_010000_000_Gate_test1_L90_C90.jpg,سقج8568,سقج ٨٥٦٨,yes,no,readable,سقج,٨٥٦٨,yes,normal,exact',
        # Arabic legacy with ه and ي: هفل458 -> normalizes to ھفل ٤٥٨
        '20261015_010001_000_Gate_test2_L90_C90.jpg,هفل458,ھفل ٤٥٨,yes,no,readable,ھفل,٤٥٨,yes,normal,norm_letters',
        # Latin legacy: wfc458 -> Latin transliterated string reported separately
        '20261015_010002_000_Gate_test3_L90_C90.jpg,wfc458,وفص ٤٥٨,yes,no,readable,وفص,٤٥٨,yes,normal,latin_translit',
    ]
    header = "file,legacy_text,predicted_text,crop_is_plate,unsupported_chars,status,true_letters,true_digits,plate_visible_to_eye,lighting,notes\n"
    csv_file.write_text("\ufeff" + header + "\n".join(rows) + "\n", encoding="utf-8")

    report = score_labeled_csv(csv_file, bootstrap_draws=100)

    leg = report["legacy_anpr_comparison"]
    # 2 comparable Arabic legacy strings, both matching perfectly after normalization
    assert leg["comparable_arabic_samples"] == 2
    assert leg["legacy_vs_human_exact_matches"] == 2
    assert leg["legacy_vs_human_accuracy_pct"] == 100.0

    # 1 Latin transliterated legacy string reported separately
    assert leg["latin_transliterated_legacy_count"] == 1
    assert leg["latin_transliterated_samples"][0]["legacy_text"] == "wfc458"


def test_invalid_status_fails(tmp_path: Path):
    """Scorer must reject invalid annotation status."""
    csv_file = tmp_path / "invalid_status.csv"

    content = (
        "\ufefffile,legacy_text,predicted_text,crop_is_plate,unsupported_chars,status,true_letters,true_digits,plate_visible_to_eye,lighting,notes\n"
        '20261015_010000_000_Gate_test_L90_C90.jpg,test,سقج ٨٥٦٨,yes,no,maybe_readable,سقج,٨٥٦٨,yes,normal,\n'
    )
    csv_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        score_labeled_csv(csv_file)

    assert "invalid status 'maybe_readable'" in str(excinfo.value)


def test_duplicate_file_id_fails(tmp_path: Path):
    """Scorer must fail on duplicate file entries."""
    csv_file = tmp_path / "duplicate_rows.csv"

    content = (
        "\ufefffile,legacy_text,predicted_text,crop_is_plate,unsupported_chars,status,true_letters,true_digits,plate_visible_to_eye,lighting,notes\n"
        '20261015_010000_000_Gate_test_L90_C90.jpg,test,سقج ٨٥٦٨,yes,no,readable,سقج,٨٥٦٨,yes,normal,\n'
        '20261015_010000_000_Gate_test_L90_C90.jpg,test,سقج ٨٥٦٨,yes,no,readable,سقج,٨٥٦٨,yes,normal,\n'
    )
    csv_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        score_labeled_csv(csv_file)

    assert "duplicate file" in str(excinfo.value)


def test_synthetic_fixtures_comprehensive(tmp_path: Path):
    """Synthetic test fixture covering:
    - correct transcription
    - one character substitution
    - insertion and deletion
    - unreadable plate
    - missed detection
    - multiple plates in one frame
    - correlated frames from the same transit
    - Eastern Arabic digits & Arabic Unicode
    """
    csv_file = tmp_path / "synthetic_fixture.csv"
    json_out = tmp_path / "synthetic_report.json"

    # Transit A has two frames 5 seconds apart (same transit cluster):
    # - Frame 1: Correct transcription
    # - Frame 2: Substitution (predicted ص instead of س)
    # Transit B: Insertion and Deletion (predicted extra char and missing digit)
    # Transit C: Unreadable plate (status: unreadable)
    # Transit D: Missed plate detection (visible=yes, model predicted nothing)
    # Transit E: Multiple plates in frame (status: multiple_plates)
    # Transit F: Partially readable with '?' (status: partially_readable)
    # Transit G: No plate in frame (status: no_plate, not visible)

    rows = [
        # Frame 1 (Transit A): Exact match
        '20261015_120000_100_TruckEntry_سقج8568_L90_C90.jpg,سقج8568,سقج ٨٥٦٨,yes,no,readable,سقج,٨٥٦٨,yes,grayscale_night,exact',
        # Frame 2 (Transit A): 1 Substitution (pred ص vs true س)
        '20261015_120005_200_TruckEntry_سقج8568_L90_C90.jpg,سقج8568,صقج ٨٥٦٨,yes,no,readable,سقج,٨٥٦٨,yes,grayscale_night,sub',
        # Frame 3 (Transit B): 1 Insertion and 1 Deletion
        '20261015_120100_300_TruckEntry_طيف7815_L90_C90.jpg,طيف7815,أطىف ٧٨١,yes,no,readable,طىف,٧٨١٥,yes,grayscale_night,ins_del',
        # Frame 4 (Transit C): Unreadable plate
        '20261015_120200_400_TruckEntry_blur_L30_C50.jpg,blur,,no_crop,no,unreadable,,,yes,grayscale_night,unreadable_motion_blur',
        # Frame 5 (Transit D): Missed detection (visible=yes, model predicted nothing)
        '20261015_120300_500_TruckEntry_miss_L60_C70.jpg,miss,,no_crop,no,readable,رقم,١٢٣٤,yes,grayscale_night,missed_detector',
        # Frame 6 (Transit E): Multiple plates in frame
        '20261015_120400_600_TruckEntry_multi_L80_C80.jpg,multi,,no_crop,no,multiple_plates,,,yes,grayscale_night,two_trucks',
        # Frame 7 (Transit F): Partially readable with uncertain char '?'
        '20261015_120500_700_TruckEntry_part_L70_C70.jpg,part,أب? ٥٦?,yes,no,partially_readable,أب?,٥٦?,partial,grayscale_night,occluded',
        # Frame 8 (Transit G): No plate in frame
        '20261015_120600_800_TruckEntry_empty_L10_C10.jpg,empty,,no_crop,no,no_plate,,,no,grayscale_night,empty_road',
    ]

    header = "file,legacy_text,predicted_text,crop_is_plate,unsupported_chars,status,true_letters,true_digits,plate_visible_to_eye,lighting,notes\n"
    csv_file.write_text("\ufeff" + header + "\n".join(rows) + "\n", encoding="utf-8")

    report = score_labeled_csv(csv_file, json_output_path=json_out, bootstrap_draws=500, seed=42)

    # 1. Verification of status counts
    assert report["sample_summary"]["total_frames"] == 8
    assert report["sample_summary"]["readable_plates"] == 4
    assert report["sample_summary"]["unreadable_plates"] == 1
    assert report["sample_summary"]["no_plate_frames"] == 1
    assert report["sample_summary"]["multiple_plates_frames"] == 1
    assert report["sample_summary"]["partially_readable"] == 1

    # 2. Verification of readable-only accuracy
    # 4 readable: Row 1 (exact), Row 2 (wrong: sub), Row 3 (wrong: ins/del), Row 5 (missed: pred empty)
    # Exact match: 1 / 4 = 25.0%
    assert report["readable_plate_accuracy"]["exact_matches"] == 1
    assert report["readable_plate_accuracy"]["total_readable"] == 4
    assert report["readable_plate_accuracy"]["accuracy_pct"] == 25.0

    # 3. Verification of Detection Recall
    # Visible plates: Row 1, 2, 3, 4, 5, 6, 7 (7 total)
    # Detected plates (has pred_text): Row 1, 2, 3, 7 (4 total)
    # Missed detections: Row 4, 5, 6 (3 total)
    det = report["detection_evaluation"]
    assert det["true_plates_visible"] == 7
    assert det["detected_plates"] == 4
    assert det["missed_detections"] == 3
    assert det["detection_recall_pct"] == round(4 / 7 * 100.0, 2)

    # 4. Verification of Error Taxonomy
    err = report["error_taxonomy"]
    assert err["substitutions"] == 1  # Row 2 (س vs ص)
    assert err["insertions"] >= 1     # Row 3 (أ)
    assert err["deletions"] >= 1      # Row 3 (٥) and Row 5 (all chars)

    # 5. Verification of All-In Accuracy
    # Visible plates that succeeded: only Row 1 (1 success out of 7 visible events)
    ai = report["all_in_accuracy"]
    assert ai["exact_matches"] == 1
    assert ai["total_eval"] == 7
    assert ai["accuracy_pct"] == round(1 / 7 * 100.0, 2)

    # 6. Verification of JSON output
    assert json_out.exists()
    assert report["sha256_hash"] is not None
