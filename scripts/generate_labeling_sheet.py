#!/usr/bin/env python3
"""Generate Stratified Labeling Sheet and Static HTML Interface (Mission 007A Step 0).

Features:
- Picks 100 non-duplicate gate images stratified across lighting and plate width (dHash > 3)
- Parses legacy ANPR text from filename into legacy_text column (hidden in HTML)
- Separate true_letters and true_digits fields entered via on-screen 27-class keypad (+ '?')
- Explicit status column: readable / partially_readable / unreadable / no_plate / multiple_plates
- Exports to a NEW file (completed_annotations.csv) as UTF-8 with BOM
- Confirmation modal and localStorage draft preservation
"""

from __future__ import annotations

import csv
import datetime
import html
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.data.splitter import compute_dhash, hamming_distance
from scripts.run_gate_triage import read_image_utf8

# The 27 canonical character classes in EALPR
EALPR_LETTERS = ["أ", "ب", "ج", "د", "ر", "س", "ص", "ط", "ع", "ف", "ق", "ل", "م", "ن", "ھ", "و", "ى"]
EALPR_DIGITS = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"]

FILENAME_PATTERN = re.compile(r"^(\d{8})_(\d{6})_(\d{3})_([^_]+)_(.*?)_L(\d+)_C(\d+)\.jpg$")


def parse_filename_metadata(filename: str) -> dict:
    m = FILENAME_PATTERN.match(filename)
    if m:
        dt = datetime.datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
        return {
            "timestamp": dt,
            "gate": m.group(4),
            "legacy_text": m.group(5),
            "light_score": int(m.group(6)),
            "conf_score": int(m.group(7)),
        }
    return {
        "timestamp": None,
        "gate": "",
        "legacy_text": "",
        "light_score": 0,
        "conf_score": 0,
    }


def generate_stratified_label_sheet(
    triage_csv_path: Path,
    gate_dir: Path,
    output_csv_path: Path,
    output_html_path: Path,
    target_count: int = 100,
) -> List[dict]:
    with open(triage_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    print(f"Loaded {len(all_rows)} triage rows.")

    image_data = []
    for r in all_rows:
        fn = r["file"]
        matches = list(gate_dir.glob(fn))
        if not matches:
            matches = list(gate_dir.rglob(fn))
        if not matches:
            continue
        p = matches[0]

        im = read_image_utf8(p)
        if im is None:
            continue
        im_pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        dh = compute_dhash(im_pil)

        brightness = float(r["mean_image_brightness"])
        is_gray = (r["is_grayscale"].strip().lower() == "true")
        plate_w = float(r["plate_w_px"])
        meta = parse_filename_metadata(fn)

        if is_gray:
            lighting = "grayscale_night"
        elif brightness < 65.0:
            lighting = "dark"
        else:
            lighting = "normal"

        if plate_w == 0:
            width_cat = "no_plate"
        elif plate_w < 100:
            width_cat = "small (<100px)"
        elif plate_w <= 160:
            width_cat = "medium (100-160px)"
        else:
            width_cat = "large (>160px)"

        image_data.append({
            "file": fn,
            "path": p,
            "dhash": dh,
            "legacy_text": meta["legacy_text"],
            "predicted_text": r["predicted_text"],
            "plate_confidence": float(r["plate_confidence"]),
            "plate_found": (r["plate_found"].strip().lower() == "true"),
            "plate_w_px": plate_w,
            "brightness": brightness,
            "lighting_cat": lighting,
            "width_cat": width_cat,
            "timestamp": meta["timestamp"],
        })

    # Group by strata
    strata = defaultdict(list)
    for item in image_data:
        strata[(item["lighting_cat"], item["width_cat"])].append(item)

    # Pick 100 images ensuring no near-duplicates (dHash > 3)
    picked: List[dict] = []
    picked_hashes: List[int] = []

    for k in strata:
        strata[k].sort(key=lambda x: (x["timestamp"].timestamp() if x["timestamp"] else 0, x["file"]))

    strata_keys = sorted(list(strata.keys()))
    round_idx = 0
    max_rounds = 300

    while len(picked) < target_count and round_idx < max_rounds:
        round_idx += 1
        added_in_round = 0
        for k in strata_keys:
            if len(picked) >= target_count:
                break
            items = strata[k]
            for it in items:
                if any(hamming_distance(it["dhash"], h) <= 3 for h in picked_hashes):
                    continue
                if it["timestamp"] and any(
                    p["timestamp"] and abs((it["timestamp"] - p["timestamp"]).total_seconds()) < 15.0
                    for p in picked
                ):
                    continue
                picked.append(it)
                picked_hashes.append(it["dhash"])
                items.remove(it)
                added_in_round += 1
                break
        if added_in_round == 0:
            for k in strata_keys:
                if len(picked) >= target_count:
                    break
                items = strata[k]
                for it in items:
                    if any(hamming_distance(it["dhash"], h) <= 3 for h in picked_hashes):
                        continue
                    picked.append(it)
                    picked_hashes.append(it["dhash"])
                    items.remove(it)
                    added_in_round += 1
                    break
            if added_in_round == 0:
                break

    print(f"Selected {len(picked)} stratified, non-duplicate images (target={target_count}).")

    # Write blank template label_sheet.csv (UTF-8 with BOM)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file",
            "legacy_text",
            "predicted_text",
            "crop_is_plate",
            "unsupported_chars",
            "status",
            "true_letters",
            "true_digits",
            "plate_visible_to_eye",
            "lighting",
            "notes",
        ])
        for it in picked:
            has_crop = bool(it.get("plate_w", 0) > 0 or it.get("plate_confidence", 0) > 0)
            writer.writerow([
                it["file"],
                it["legacy_text"],
                it["predicted_text"],
                "yes" if has_crop else "no_crop",
                "no",  # unsupported_chars default
                "",  # status
                "",  # true_letters
                "",  # true_digits
                "",  # plate_visible_to_eye
                "",  # lighting: blank by default
                "",  # notes
            ])

    print(f"Saved template labeling sheet to: {output_csv_path}")

    # Build HTML interface with opaque sample IDs and sanitized image links
    crops_dir = output_csv_path.parent / "crops"
    annotated_dir = output_csv_path.parent / "annotated"
    samples_dir = output_csv_path.parent / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    html_rows = []
    for idx, it in enumerate(picked, 1):
        fn = it["file"]
        stem = Path(fn).stem
        matching_crops = list(crops_dir.glob(f"*{stem}*_crop.jpg"))
        matching_ann = list(annotated_dir.glob(f"*{stem}*.jpg"))

        crop_opaque_rel = ""
        if matching_crops:
            crop_opaque_rel = f"samples/crop_{idx:03d}.jpg"
            shutil.copy2(matching_crops[0], output_csv_path.parent / crop_opaque_rel)

        ann_opaque_rel = ""
        if matching_ann:
            ann_opaque_rel = f"samples/frame_{idx:03d}.jpg"
            shutil.copy2(matching_ann[0], output_csv_path.parent / ann_opaque_rel)

        has_crop = bool(matching_crops)

        html_rows.append(f"""
        <tr class="sample-row" id="row-{idx}" data-idx="{idx}">
            <td class="idx-cell">{idx}</td>
            <td class="media-cell">
                <div class="media-container">
                    {f'<a href="{ann_opaque_rel}" target="_blank"><img src="{crop_opaque_rel}" alt="Crop #{idx:03d}" class="crop-img" title="Click to view full 4K frame"></a>' if crop_opaque_rel else '<div class="no-crop">No Crop</div>'}
                    {f'<a href="{ann_opaque_rel}" target="_blank"><img src="{ann_opaque_rel}" alt="Full Frame #{idx:03d}" class="full-img" title="Full 4K Frame"></a>' if ann_opaque_rel else ''}
                </div>
                <div class="file-name">Sample #{idx:03d}</div>
            </td>
            <td class="input-cell crop-plate-col">
                <select name="crop_is_plate_{idx}" id="crop_is_plate_{idx}" class="select-input" onchange="saveDraft()">
                    <option value="yes" {'selected' if has_crop else ''}>Yes</option>
                    <option value="no">No (Wrong Crop)</option>
                    <option value="no_crop" {'selected' if not has_crop else ''}>No Crop (N/A)</option>
                </select>
            </td>
            <td class="input-cell status-col">
                <select name="status_{idx}" id="status_{idx}" class="select-input status-select" onchange="onStatusChange({idx})">
                    <option value="" disabled selected>-- Select Status --</option>
                    <option value="readable">readable</option>
                    <option value="partially_readable">partially_readable</option>
                    <option value="unreadable">unreadable</option>
                    <option value="no_plate">no_plate</option>
                    <option value="multiple_plates">multiple_plates</option>
                </select>
            </td>
            <td class="input-cell letters-col">
                <input type="text" name="true_letters_{idx}" id="true_letters_{idx}"
                       placeholder="e.g. سقج" class="text-input arabic-font letter-input"
                       dir="rtl" onfocus="setActiveField(this, 'letters')" oninput="normalizeInput(this, 'letters')">
            </td>
            <td class="input-cell digits-col">
                <input type="text" name="true_digits_{idx}" id="true_digits_{idx}"
                       placeholder="e.g. ٨٥٦٨" class="text-input arabic-font digit-input"
                       dir="ltr" onfocus="setActiveField(this, 'digits')" oninput="normalizeInput(this, 'digits')">
            </td>
            <td class="input-cell unsupported-col">
                <label class="checkbox-label" title="Plate has characters outside the 27 classes (e.g. Latin letters)">
                    <input type="checkbox" name="unsupported_chars_{idx}" id="unsupported_chars_{idx}" onchange="saveDraft()"> Latin/Other
                </label>
            </td>
            <td class="input-cell visible-col">
                <select name="visible_{idx}" id="visible_{idx}" class="select-input" onchange="saveDraft()">
                    <option value="yes" selected>Yes</option>
                    <option value="partial">Partial</option>
                    <option value="no">No</option>
                </select>
            </td>
            <td class="input-cell lighting-col">
                <select name="lighting_{idx}" id="lighting_{idx}" class="select-input" onchange="saveDraft()">
                    <option value="" disabled selected>-- Select --</option>
                    <option value="daylight">Daylight</option>
                    <option value="night_ir">Night / IR</option>
                    <option value="dark">Dark</option>
                    <option value="glare">Glare / Washout</option>
                </select>
            </td>
            <td class="input-cell notes-col">
                <input type="text" name="notes_{idx}" id="notes_{idx}" placeholder="Optional notes" class="text-input" onchange="saveDraft()">
            </td>
        </tr>
        """)

    # Build letter buttons
    letter_buttons_html = "".join([
        f'<button type="button" class="key-btn letter-key" onclick="insertChar(\'{ch}\')">{ch}</button>'
        for ch in EALPR_LETTERS
    ])
    # Build digit buttons
    digit_buttons_html = "".join([
        f'<button type="button" class="key-btn digit-key" onclick="insertChar(\'{ch}\')">{ch}</button>'
        for ch in EALPR_DIGITS
    ])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mission 007A: Egyptian ALPR Gate Ground Truth Tool</title>
    <style>
        :root {{
            --bg-primary: #090d16;
            --bg-secondary: #131b2e;
            --bg-card: #1e293b;
            --accent: #38bdf8;
            --accent-hover: #0ea5e9;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border: #334155;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            padding: 20px 24px 180px 24px;
            line-height: 1.5;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 18px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 20px;
        }}
        h1 {{ font-size: 1.5rem; font-weight: 700; color: var(--accent); }}
        .header-desc {{ color: var(--text-secondary); font-size: 0.9rem; margin-top: 3px; }}
        .btn {{
            background: var(--accent);
            color: #090d16;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 700;
            cursor: pointer;
            font-size: 0.95rem;
            transition: all 0.2s ease;
            box-shadow: 0 2px 8px rgba(56, 189, 248, 0.25);
        }}
        .btn:hover {{ background: var(--accent-hover); transform: translateY(-1px); }}
        .btn-secondary {{
            background: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--border);
            margin-right: 10px;
        }}
        .btn-secondary:hover {{ background: #2d3748; }}
        .instructions {{
            background: rgba(30, 41, 59, 0.7);
            border-left: 4px solid var(--accent);
            padding: 12px 18px;
            margin-bottom: 20px;
            border-radius: 4px;
            font-size: 0.88rem;
        }}
        .table-wrap {{
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border);
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
        }}
        th {{
            background: #0f172a;
            padding: 12px 14px;
            text-align: left;
            font-weight: 600;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        td {{
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }}
        tr:hover td {{ background: rgba(56, 189, 248, 0.03); }}
        .idx-cell {{ font-weight: 700; color: var(--accent); text-align: center; width: 35px; }}
        .media-container {{ display: flex; gap: 8px; align-items: center; margin-bottom: 3px; }}
        .crop-img, .full-img {{
            height: 65px;
            border-radius: 4px;
            border: 1px solid var(--border);
            background: #000;
            object-fit: contain;
            cursor: pointer;
            transition: transform 0.15s ease;
        }}
        .crop-img:hover, .full-img:hover {{ transform: scale(1.05); }}
        .no-crop {{
            height: 65px;
            width: 110px;
            background: #334155;
            color: #94a3b8;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            border-radius: 4px;
        }}
        .file-name {{ font-family: monospace; font-size: 0.8rem; font-weight: 600; color: var(--accent); max-width: 260px; }}
        .text-input, .select-input {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: #fff;
            padding: 7px 10px;
            border-radius: 4px;
            font-size: 0.9rem;
            width: 100%;
        }}
        .arabic-font {{ font-size: 1.15rem; font-weight: 700; font-family: "Segoe UI", Tahoma, Arial, sans-serif; }}
        .letter-input {{ width: 105px; text-align: center; }}
        .digit-input {{ width: 105px; text-align: center; }}
        .status-select {{ width: 135px; }}
        .checkbox-label {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85rem;
            color: var(--text-primary);
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
        }}
        .checkbox-label input[type="checkbox"] {{
            width: 16px;
            height: 16px;
            cursor: pointer;
            accent-color: var(--accent);
        }}
        .text-input:focus, .select-input:focus {{ outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px rgba(56,189,248,0.25); }}

        /* Sticky On-Screen Keypad Dock */
        .keypad-dock {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #0f172a;
            border-top: 2px solid var(--accent);
            padding: 12px 24px;
            z-index: 100;
            box-shadow: 0 -4px 20px rgba(0,0,0,0.6);
        }}
        .keypad-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        .active-field-indicator {{
            color: var(--accent);
            font-weight: 600;
        }}
        .keypad-row {{
            display: flex;
            gap: 6px;
            margin-bottom: 6px;
            justify-content: center;
            flex-wrap: wrap;
        }}
        .key-btn {{
            background: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 5px;
            min-width: 44px;
            height: 38px;
            font-size: 1.15rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s ease;
            user-select: none;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .key-btn:hover {{
            background: var(--accent);
            color: #090d16;
            border-color: var(--accent);
            transform: translateY(-2px);
        }}
        .letter-key {{ border-color: #38bdf8; }}
        .digit-key {{ border-color: #f59e0b; color: #fbbf24; }}
        .special-key {{ background: #334155; min-width: 65px; font-size: 0.9rem; font-weight: 600; }}
        .uncertain-key {{ background: #991b1b; color: #fee2e2; border-color: #ef4444; }}

        /* Confirmation Modal */
        .modal-overlay {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.75);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 200;
        }}
        .modal-card {{
            background: var(--bg-secondary);
            border: 2px solid var(--success);
            border-radius: 8px;
            padding: 24px 32px;
            max-width: 500px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.8);
        }}
        .modal-title {{ font-size: 1.3rem; font-weight: 700; color: var(--success); margin-bottom: 10px; }}
        .modal-text {{ color: var(--text-primary); font-size: 0.95rem; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Mission 007A: Egyptian ALPR Gate CCTV Ground Truth Tool</h1>
            <div class="header-desc">100 Stratified Non-Duplicate Frames &bull; On-Screen 27-Class Keypad &bull; UTF-8 BOM Safe Export</div>
        </div>
        <div>
            <button class="btn btn-secondary" onclick="restoreDraft()">Restore Saved Draft</button>
            <button class="btn" onclick="exportCompletedCSV()">Export Completed Annotations</button>
        </div>
    </div>

    <div class="instructions">
        <strong>Annotation Protocol:</strong> Click on <strong>true_letters</strong> (Arabic reading order: right to left) or <strong>true_digits</strong> (left to right) to focus, then click keys on the on-screen keypad below or type directly. Set status to <code>readable</code>, <code>partially_readable</code> (use <code>?</code> for uncertain characters), <code>unreadable</code>, <code>no_plate</code>, or <code>multiple_plates</code>. Verify <strong>Crop is Plate?</strong> and check <strong>Latin/Other</strong> if plate has characters outside the 27 classes. Click <strong>Export Completed Annotations</strong> when finished to download <code>completed_annotations.csv</code>.
    </div>

    <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Visual Sample</th>
                    <th>Crop is Plate?</th>
                    <th>Status</th>
                    <th>Letters (RTL)</th>
                    <th>Digits (LTR)</th>
                    <th>Unsupported?</th>
                    <th>Visible?</th>
                    <th>Lighting</th>
                    <th>Notes</th>
                </tr>
            </thead>
            <tbody>
                {''.join(html_rows)}
            </tbody>
        </table>
    </div>

    <!-- On-Screen Keypad Dock -->
    <div class="keypad-dock">
        <div class="keypad-header">
            <div><strong>Active Target:</strong> <span id="activeFieldDisplay" class="active-field-indicator">None (click an input field above)</span></div>
            <div><span>Letters (RTL) &bull; Digits (LTR) &bull; 27 Classes + '?'</span></div>
        </div>
        <!-- Arabic Letters (17 classes) -->
        <div class="keypad-row">
            {letter_buttons_html}
        </div>
        <!-- Eastern Arabic Digits (10 classes) + Special -->
        <div class="keypad-row">
            {digit_buttons_html}
            <button type="button" class="key-btn uncertain-key" onclick="insertChar('?')">?</button>
            <button type="button" class="key-btn special-key" onclick="insertBackspace()">⌫ Del</button>
            <button type="button" class="key-btn special-key" onclick="clearActiveField()">Clear</button>
        </div>
    </div>

    <!-- Confirmation Modal -->
    <div class="modal-overlay" id="confirmModal">
        <div class="modal-card">
            <div class="modal-title">&#10004; Annotations Exported Successfully</div>
            <div class="modal-text" id="modalMessage">Exported 100 records with UTF-8 BOM to <code>completed_annotations.csv</code>. Original <code>label_sheet.csv</code> remains untouched.</div>
            <button class="btn" onclick="closeModal()">OK</button>
        </div>
    </div>

    <script>
        const initialData = {json.dumps([{"id": idx, "file": it["file"], "legacy_text": it["legacy_text"], "predicted_text": it["predicted_text"]} for idx, it in enumerate(picked, 1)])};
        let activeInput = null;

        function setActiveField(inputElem, fieldType) {{
            activeInput = inputElem;
            const rowIdx = inputElem.id.split("_").pop();
            const fieldName = (fieldType === "letters") ? "true_letters (Arabic Letters)" : "true_digits (Numerals)";
            document.getElementById("activeFieldDisplay").textContent = "Sample #" + String(rowIdx).padStart(3, '0') + " &rarr; " + fieldName;
        }}

        function insertChar(ch) {{
            if (!activeInput) {{
                alert("Please click on a true_letters or true_digits input field first.");
                return;
            }}
            activeInput.value += ch;
            activeInput.dispatchEvent(new Event("input"));
            activeInput.focus();
            saveDraft();
        }}

        function insertBackspace() {{
            if (!activeInput) return;
            activeInput.value = activeInput.value.slice(0, -1);
            activeInput.dispatchEvent(new Event("input"));
            activeInput.focus();
            saveDraft();
        }}

        function clearActiveField() {{
            if (!activeInput) return;
            activeInput.value = "";
            activeInput.dispatchEvent(new Event("input"));
            activeInput.focus();
            saveDraft();
        }}

        // Automatic normalization for physical keyboard typing
        function normalizeInput(inputElem, fieldType) {{
            let val = inputElem.value;
            if (fieldType === "letters") {{
                // Normalize standard ha U+0647 -> EALPR ha U+06BE
                val = val.replace(/\\u0647/g, "\\u06BE");
                // Normalize standard ya U+064A -> EALPR alef maksura U+0649
                val = val.replace(/\\u064A/g, "\\u0649");
            }} else if (fieldType === "digits") {{
                // Normalize ASCII digits 0-9 to Eastern Arabic digits
                const asciiToEastern = {{'0':'٠','1':'١','2':'٢','3':'٣','4':'٤','5':'٥','6':'٦','7':'٧','8':'٨','9':'٩'}};
                val = val.split('').map(c => asciiToEastern[c] || c).join('');
            }}
            inputElem.value = val;
            saveDraft();
        }}

        function onStatusChange(idx) {{
            const status = document.getElementById("status_" + idx).value;
            const letters = document.getElementById("true_letters_" + idx);
            const digits = document.getElementById("true_digits_" + idx);
            const vis = document.getElementById("visible_" + idx);

            if (status === "no_plate") {{
                letters.value = "";
                digits.value = "";
                vis.value = "no";
            }} else if (status === "unreadable") {{
                letters.value = "";
                digits.value = "";
            }}
            saveDraft();
        }}

        function saveDraft() {{
            const draft = {{}};
            initialData.forEach((row, i) => {{
                const idx = i + 1;
                draft["row_" + idx] = {{
                    crop_is_plate: document.getElementById("crop_is_plate_" + idx).value,
                    unsupported_chars: document.getElementById("unsupported_chars_" + idx).checked,
                    status: document.getElementById("status_" + idx).value,
                    letters: document.getElementById("true_letters_" + idx).value,
                    digits: document.getElementById("true_digits_" + idx).value,
                    visible: document.getElementById("visible_" + idx).value,
                    lighting: document.getElementById("lighting_" + idx).value,
                    notes: document.getElementById("notes_" + idx).value,
                }};
            }});
            try {{
                localStorage.setItem("egyptian_alpr_triage_draft_v2", JSON.stringify(draft));
            }} catch(e) {{}}
        }}

        function restoreDraft() {{
            try {{
                const saved = localStorage.getItem("egyptian_alpr_triage_draft_v2") || localStorage.getItem("egyptian_alpr_triage_draft");
                if (!saved) {{
                    alert("No saved draft found in browser storage.");
                    return;
                }}
                const draft = JSON.parse(saved);
                let restored = 0;
                initialData.forEach((row, i) => {{
                    const idx = i + 1;
                    const rData = draft["row_" + idx];
                    if (rData) {{
                        if (rData.crop_is_plate) document.getElementById("crop_is_plate_" + idx).value = rData.crop_is_plate;
                        if (typeof rData.unsupported_chars !== "undefined") document.getElementById("unsupported_chars_" + idx).checked = !!rData.unsupported_chars;
                        if (rData.status) document.getElementById("status_" + idx).value = rData.status;
                        if (rData.letters) document.getElementById("true_letters_" + idx).value = rData.letters;
                        if (rData.digits) document.getElementById("true_digits_" + idx).value = rData.digits;
                        if (rData.visible) document.getElementById("visible_" + idx).value = rData.visible;
                        if (rData.lighting) document.getElementById("lighting_" + idx).value = rData.lighting;
                        if (rData.notes) document.getElementById("notes_" + idx).value = rData.notes;
                        restored++;
                    }}
                }});
                alert("Successfully restored " + restored + " rows from saved draft!");
            }} catch(e) {{
                alert("Could not restore draft: " + e.message);
            }}
        }}

        function exportCompletedCSV() {{
            let csv = "file,legacy_text,predicted_text,crop_is_plate,unsupported_chars,status,true_letters,true_digits,plate_visible_to_eye,lighting,notes\\n";
            let filledCount = 0;

            initialData.forEach((row, i) => {{
                const idx = i + 1;
                const cropIsPlate = (document.getElementById("crop_is_plate_" + idx).value || "").trim();
                const unsupportedChars = document.getElementById("unsupported_chars_" + idx).checked ? "yes" : "no";
                const status = (document.getElementById("status_" + idx).value || "").trim();
                const letters = (document.getElementById("true_letters_" + idx).value || "").trim();
                const digits = (document.getElementById("true_digits_" + idx).value || "").trim();
                const visible = document.getElementById("visible_" + idx).value;
                const lighting = document.getElementById("lighting_" + idx).value;
                const notes = (document.getElementById("notes_" + idx).value || "").trim();

                if (status || letters || digits) filledCount++;

                const safeFile = '"' + row.file.replace(/"/g, '""') + '"';
                const safeLegacy = '"' + (row.legacy_text || "").replace(/"/g, '""') + '"';
                const safePred = '"' + (row.predicted_text || "").replace(/"/g, '""') + '"';
                const safeCrop = '"' + cropIsPlate.replace(/"/g, '""') + '"';
                const safeUnsupported = '"' + unsupportedChars.replace(/"/g, '""') + '"';
                const safeStatus = '"' + status.replace(/"/g, '""') + '"';
                const safeLetters = '"' + letters.replace(/"/g, '""') + '"';
                const safeDigits = '"' + digits.replace(/"/g, '""') + '"';
                const safeVis = '"' + visible.replace(/"/g, '""') + '"';
                const safeLight = '"' + lighting.replace(/"/g, '""') + '"';
                const safeNotes = '"' + notes.replace(/"/g, '""') + '"';

                csv += [safeFile, safeLegacy, safePred, safeCrop, safeUnsupported, safeStatus, safeLetters, safeDigits, safeVis, safeLight, safeNotes].join(",") + "\\n";
            }});

            // UTF-8 with BOM prefix
            const blob = new Blob(["\\uFEFF" + csv], {{ type: "text/csv;charset=utf-8;" }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "completed_annotations.csv";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);

            // Show Confirmation Modal
            document.getElementById("modalMessage").innerHTML = "Exported <strong>" + initialData.length + "</strong> records (" + filledCount + " annotated) to <code>completed_annotations.csv</code> with UTF-8 BOM.<br><br>The original <code>label_sheet.csv</code> remains completely untouched.<br>You can now run:<br><br><code>python scripts/score_labeled_triage.py artifacts/triage/completed_annotations.csv</code>";
            document.getElementById("confirmModal").style.display = "flex";
        }}

        function closeModal() {{
            document.getElementById("confirmModal").style.display = "none";
        }}

        // Auto-restore draft on page load if available
        window.addEventListener("DOMContentLoaded", () => {{
            const saved = localStorage.getItem("egyptian_alpr_triage_draft_v2") || localStorage.getItem("egyptian_alpr_triage_draft");
            if (saved) {{
                try {{
                    const draft = JSON.parse(saved);
                    initialData.forEach((row, i) => {{
                        const idx = i + 1;
                        const rData = draft["row_" + idx];
                        if (rData) {{
                            if (rData.crop_is_plate) document.getElementById("crop_is_plate_" + idx).value = rData.crop_is_plate;
                            if (typeof rData.unsupported_chars !== "undefined") document.getElementById("unsupported_chars_" + idx).checked = !!rData.unsupported_chars;
                            if (rData.status) document.getElementById("status_" + idx).value = rData.status;
                            if (rData.letters) document.getElementById("true_letters_" + idx).value = rData.letters;
                            if (rData.digits) document.getElementById("true_digits_" + idx).value = rData.digits;
                            if (rData.visible) document.getElementById("visible_" + idx).value = rData.visible;
                            if (rData.lighting) document.getElementById("lighting_" + idx).value = rData.lighting;
                            if (rData.notes) document.getElementById("notes_" + idx).value = rData.notes;
                        }}
                    }});
                }} catch(e) {{}}
            }}
        }});
    </script>
</body>
</html>
"""
    output_html_path.write_text(html_content, encoding="utf-8")
    print(f"Generated static labeling HTML interface to: {output_html_path}")

    return picked


if __name__ == "__main__":
    triage_csv = Path("artifacts/triage/triage.csv")
    gate_folder = Path(r"D:\TruckEntry")
    out_csv = Path("artifacts/triage/label_sheet.csv")
    out_html = Path("artifacts/triage/labeling_interface.html")
    generate_stratified_label_sheet(triage_csv, gate_folder, out_csv, out_html, target_count=100)
