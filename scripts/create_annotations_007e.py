#!/usr/bin/env python3
"""Generate finalized ground truth annotations: completed_annotations_007e.csv.

Mission 007E: Data Quality & Ground Truth Finalization.
Preserves completed_annotations.csv untouched.
Applies verified human revisions for:
- Rows 1, 37: RTL letter correction ('سقج' instead of transposed 'سجق')
- Row 23: RTL letter correction ('وعر' instead of mislabeled 'صجر')
- Row 14: Motorcycle no-plate correction (empty letters and digits instead of '???')
- Rows 18, 30, 32, 38, 55, 90, 93: Explicit unsupported_chars='yes' for OOV letter 'ك' (U+0643)
- Row 38: Populate missing status='readable' and lighting='night_ir'
All other 89 rows are preserved byte-for-byte.
Exports as UTF-8 with BOM (\ufeff).
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    root_dir = Path(__file__).resolve().parent.parent
    src_csv = root_dir / "artifacts" / "triage" / "completed_annotations.csv"
    out_csv = root_dir / "artifacts" / "triage" / "completed_annotations_007e.csv"

    if not src_csv.exists():
        raise FileNotFoundError(f"Source annotations not found: {src_csv}")

    src_hash = compute_sha256(src_csv)
    print(f"Source: {src_csv} (SHA-256: {src_hash})")

    with open(src_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    print(f"Loaded {len(rows)} source rows.")

    audit_log = []

    # Apply verified human revisions
    # Row 1 (index 0)
    old_row_1 = dict(rows[0])
    rows[0]["true_letters"] = "سقج"
    audit_log.append({
        "row": 1,
        "file": rows[0]["file"],
        "field": "true_letters",
        "old_value": old_row_1["true_letters"],
        "new_value": rows[0]["true_letters"],
        "rationale": "Independent visual review of crop_001.jpg confirms RTL reading order is Seen (س), Qaf (ق), Jeem (ج) -> 'سقج'. Original label transposed middle and left letters."
    })

    # Row 14 (index 13)
    old_row_14 = dict(rows[13])
    rows[13]["true_letters"] = ""
    rows[13]["true_digits"] = ""
    rows[13]["status"] = "no_plate"
    rows[13]["plate_visible_to_eye"] = "no"
    audit_log.append({
        "row": 14,
        "file": rows[13]["file"],
        "field": "true_letters, true_digits",
        "old_value": f"{old_row_14['true_letters']!r}, {old_row_14['true_digits']!r}",
        "new_value": "'' , ''",
        "rationale": "Visual review of frame_014.jpg confirms side view of passing motorcycle with no plate visible anywhere. Transcription rules require no invented transcriptions for no_plate status."
    })

    # Row 18 (index 17)
    old_row_18 = dict(rows[17])
    rows[17]["unsupported_chars"] = "yes"
    audit_log.append({
        "row": 18,
        "file": rows[17]["file"],
        "field": "unsupported_chars",
        "old_value": old_row_18["unsupported_chars"],
        "new_value": rows[17]["unsupported_chars"],
        "rationale": "Visual review of frame_018.jpg confirms square tricycle plate with letters 'كمد'. Contains documented OOV letter Kaf (ك, U+0643) outside the 27 model classes."
    })

    # Row 23 (index 22)
    old_row_23 = dict(rows[22])
    rows[22]["true_letters"] = "وعر"
    audit_log.append({
        "row": 23,
        "file": rows[22]["file"],
        "field": "true_letters",
        "old_value": old_row_23["true_letters"],
        "new_value": rows[22]["true_letters"],
        "rationale": "Independent visual review of crop_023.jpg confirms RTL reading order is Waw (و), Ain (ع), Raa (ر) -> 'وعر'. Original label 'صجر' was completely mismatched."
    })

    # Row 30 (index 29)
    old_row_30 = dict(rows[29])
    rows[29]["unsupported_chars"] = "yes"
    audit_log.append({
        "row": 30,
        "file": rows[29]["file"],
        "field": "unsupported_chars",
        "old_value": old_row_30["unsupported_chars"],
        "new_value": rows[29]["unsupported_chars"],
        "rationale": "Visual review of frame_030.jpg confirms square tricycle plate with letters 'كمد'. Contains documented OOV letter Kaf (ك, U+0643)."
    })

    # Row 32 (index 31)
    old_row_32 = dict(rows[31])
    rows[31]["unsupported_chars"] = "yes"
    audit_log.append({
        "row": 32,
        "file": rows[31]["file"],
        "field": "unsupported_chars",
        "old_value": old_row_32["unsupported_chars"],
        "new_value": rows[31]["unsupported_chars"],
        "rationale": "Visual review of frame_032.jpg confirms square tricycle plate with letters 'كمد' and digits '٧٦١٢'. Contains documented OOV letter Kaf (ك, U+0643)."
    })

    # Row 37 (index 36)
    old_row_37 = dict(rows[36])
    rows[36]["true_letters"] = "سقج"
    audit_log.append({
        "row": 37,
        "file": rows[36]["file"],
        "field": "true_letters",
        "old_value": old_row_37["true_letters"],
        "new_value": rows[36]["true_letters"],
        "rationale": "Independent visual review of crop_037.jpg confirms RTL reading order is Seen (س), Qaf (ق), Jeem (ج) -> 'سقج'. Original label transposed middle and left letters."
    })

    # Row 38 (index 37)
    old_row_38 = dict(rows[37])
    rows[37]["status"] = "readable"
    rows[37]["unsupported_chars"] = "yes"
    rows[37]["lighting"] = "night_ir"
    audit_log.append({
        "row": 38,
        "file": rows[37]["file"],
        "field": "status, unsupported_chars, lighting",
        "old_value": f"status={old_row_38['status']!r}, unsupp={old_row_38['unsupported_chars']!r}, light={old_row_38['lighting']!r}",
        "new_value": "status='readable', unsupported_chars='yes', lighting='night_ir'",
        "rationale": "Visual review of frame_038.jpg confirms square tricycle plate with letters 'كمد' and digits '٥٨١٧' at night. Populated missing status, lighting, and documented OOV flag."
    })

    # Row 55 (index 54)
    old_row_55 = dict(rows[54])
    rows[54]["unsupported_chars"] = "yes"
    audit_log.append({
        "row": 55,
        "file": rows[54]["file"],
        "field": "unsupported_chars",
        "old_value": old_row_55["unsupported_chars"],
        "new_value": rows[54]["unsupported_chars"],
        "rationale": "Visual review of crop_055.jpg confirms rectangular plate with letters 'كدم'. Contains documented OOV letter Kaf (ك, U+0643)."
    })

    # Row 90 (index 89)
    old_row_90 = dict(rows[89])
    rows[89]["unsupported_chars"] = "yes"
    audit_log.append({
        "row": 90,
        "file": rows[89]["file"],
        "field": "unsupported_chars",
        "old_value": old_row_90["unsupported_chars"],
        "new_value": rows[89]["unsupported_chars"],
        "rationale": "Visual review of crop_090.jpg confirms rectangular plate with letters 'كدو'. Contains documented OOV letter Kaf (ك, U+0643)."
    })

    # Row 93 (index 92)
    old_row_93 = dict(rows[92])
    rows[92]["unsupported_chars"] = "yes"
    audit_log.append({
        "row": 93,
        "file": rows[92]["file"],
        "field": "unsupported_chars",
        "old_value": old_row_93["unsupported_chars"],
        "new_value": rows[92]["unsupported_chars"],
        "rationale": "Visual review of crop_093.jpg confirms rectangular plate with letters 'كطو'. Contains documented OOV letter Kaf (ك, U+0643)."
    })

    # Write out completed_annotations_007e.csv with UTF-8 BOM
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    out_hash = compute_sha256(out_csv)
    print(f"\nGenerated: {out_csv}")
    print(f"SHA-256: {out_hash}")
    print(f"Total rows written: {len(rows)}")
    print(f"Total audit revisions: {len(audit_log)}")

    # Print audit log table
    print("\n=== AUDIT TRAIL OF REVISED ANNOTATIONS ===")
    for entry in audit_log:
        print(f"Row {entry['row']:02d} [{entry['field']}]:")
        print(f"  Old: {entry['old_value']}")
        print(f"  New: {entry['new_value']}")
        print(f"  Rationale: {entry['rationale']}\n")

if __name__ == "__main__":
    main()
