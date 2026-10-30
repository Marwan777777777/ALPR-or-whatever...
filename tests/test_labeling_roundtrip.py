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


def test_step3_metrics_synthetic_fixture(tmp_path: Path):
    """Synthetic test fixture testing Mission 007C Step 3 (a)-(j) metrics:
    - Step 3a: Exact match in-vocab readable crop
    - Step 3b: OOV rate (with documented OOV 'ك')
    - Step 3c: All-in accuracy (penalizing misses, unreadable, OOV, wrong crop)
    - Step 3d: No-detection stratum tracking
    - Step 3e: Agreement comparisons and Latin transliterated legacy strings
    - Step 3f: Error taxonomy and CER
    - Step 3g: Format check rule (>3 letters or >4 digits)
    - Step 3h: Transit ID cluster bootstrap CI and candidate groups
    """
    csv_file = tmp_path / "mock_step3_fixture.csv"
    json_out = tmp_path / "mock_step3_summary.json"

    # Synthetic mock rows with transit_id column
    # Transit 101: 2 frames of the same vehicle
    #   - Frame 1: Perfect match (crop=yes, readable)
    #   - Frame 2: 1 substitution error (crop=yes, readable)
    # Transit 102: Format violation prediction (>3 letters: 4 letters predicted)
    # Transit 103: Documented OOV letter 'ك' (crop=yes, unsupported_chars=yes)
    # Transit 104: Wrong crop (crop_is_plate=no)
    # Transit 105: No-detection stratum (crop_is_plate=no_crop, pred empty, visible=yes)
    # Transit 106: Latin legacy ANPR fallback (legacy=abc1234)
    rows = [
        # Frame 1: transit_101, exact match
        'mock_frame_01.jpg,طىف1234,طىف ١٢٣٤,yes,no,readable,طىف,١٢٣٤,yes,daylight,exact,transit_101',
        # Frame 2: transit_101, substitution error (ط vs ص)
        'mock_frame_02.jpg,طىف1234,صىف ١٢٣٤,yes,no,readable,طىف,١٢٣٤,yes,daylight,sub,transit_101',
        # Frame 3: transit_102, format violation (>3 letters: predicted 4 letters: أبجد)
        'mock_frame_03.jpg,أبج5678,أبجد ٥٦٧٨,yes,no,readable,أبج,٥٦٧٨,yes,daylight,fmt_err,transit_102',
        # Frame 4: transit_103, OOV letter ك (unsupported_chars=yes)
        'mock_frame_04.jpg,كفل9999,,yes,yes,readable,كفل,٩٩٩٩,yes,daylight,oov_plate,transit_103',
        # Frame 5: transit_104, wrong crop (crop_is_plate=no)
        'mock_frame_05.jpg,رقم1111,,no,no,unreadable,,,yes,daylight,wrong_crop,transit_104',
        # Frame 6: transit_105, 19-frame stratum style (no crop, missed detection)
        'mock_frame_06.jpg,رقم2222,,no_crop,no,readable,رقم,٢٢٢٢,yes,daylight,stratum_miss,transit_105',
        # Frame 7: transit_106, Latin legacy fallback string
        'mock_frame_07.jpg,abc1234,طىف ١٢٣٤,yes,no,readable,طىف,١٢٣٤,yes,night,latin_legacy,transit_106',
    ]

    header = "file,legacy_text,predicted_text,crop_is_plate,unsupported_chars,status,true_letters,true_digits,plate_visible_to_eye,lighting,notes,transit_id\n"
    csv_file.write_text("\ufeff" + header + "\n".join(rows) + "\n", encoding="utf-8")

    report = score_labeled_csv(csv_file, json_output_path=json_out, bootstrap_draws=200, seed=123)

    # 1. Step 3a: Exact match on in-vocab readable crop
    # In-vocab readable with crop=yes: Frames 1, 2, 3, 7 (4 total).
    # Exact matches: Frame 1 and Frame 7 (2 total).
    s3a = report["step_3a_exact_match_in_vocab"]
    assert s3a["numerator"] == 2
    assert s3a["denominator"] == 4
    assert s3a["accuracy_pct"] == 50.0

    # 2. Step 3b: OOV Rate (one 'ك')
    s3b = report["step_3b_oov_rate"]
    assert s3b["oov_characters_count"] == 1
    assert s3b["character_breakdown"] == {"ك": 1}
    assert s3b["oov_plates_count"] == 1

    # 3. Step 3c: All-in accuracy
    # Total frames: 7. Succeeded: 2 (Frame 1 and Frame 7).
    s3c = report["step_3c_all_in_accuracy"]
    assert s3c["all_frames_denominator"] == 7
    assert s3c["exact_matches"] == 2
    assert s3c["all_in_accuracy_all_frames_pct"] == round(2 / 7 * 100.0, 2)

    # 4. Step 3d: No-detection stratum
    s3d = report["step_3d_no_detection_stratum"]
    assert s3d["total_stratum_frames"] == 1
    assert s3d["plate_visible_counts"]["yes"] == 1

    # 5. Step 3e: Agreement and Latin legacy
    s3e = report["step_3e_agreement_comparisons"]
    assert s3e["latin_legacy_count"] == 1
    assert s3e["latin_legacy_samples"][0]["legacy_text"] == "abc1234"

    # 6. Step 3f: Error taxonomy
    s3f = report["step_3f_error_taxonomy"]
    assert s3f["wrong_plates_count"] == 2  # Frame 2 and Frame 3
    assert s3f["substitutions"] >= 1       # Frame 2 (ط vs ص)
    assert s3f["insertions"] >= 1          # Frame 3 (د)
    assert s3f["macro_cer"] > 0
    assert s3f["micro_cer"] > 0

    # 7. Step 3g: Format check rule
    s3g = report["step_3g_format_check_rule"]
    assert s3g["wrong_predictions_violating"] == 1  # Frame 3 flagged
    assert s3g["correct_predictions_violating"] == 0 # Zero false alarms on exact matches

    # 8. Step 3h: Cluster bootstrap CI
    s3h = report["step_3h_grouping"]
    assert s3h["has_transit_id_column"] is True
    assert s3h["cluster_bootstrap_ci"] is not None
    assert len(s3h["cluster_bootstrap_ci"]["ci_95"]) == 2

    # 9. JSON output metadata verification
    assert json_out.exists()
    meta = report["metadata"]
    assert meta["random_seed"] == 123
    assert meta["csv_sha256"] is not None
    assert meta["config"]["det_conf"] == 0.25
    assert meta["config"]["agnostic_nms"] is False

