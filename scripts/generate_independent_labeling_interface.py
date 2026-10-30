#!/usr/bin/env python3
"""Generate Independent, Blinded Human Labeling Interface (Mission 007E Step 2).

Creates a standalone static HTML labeling interface:
`artifacts/triage/independent_labeling_interface.html`

Guarantees of the independent annotation workflow:
1. Anonymous Presentation:
   - Evaluator sees ONLY anonymous sample IDs: 'Sample #001' .. 'Sample #100'
   - Full frame links point to 'samples/frame_XXX.jpg'
   - Crop links point to 'samples/crop_XXX.jpg' (when crop exists)
2. Blinded Evaluation (Zero Model / Legacy Bias):
   - Model predictions are completely excluded from DOM and JS memory.
   - Legacy OCR strings are completely excluded.
   - Character confidence scores are completely excluded.
   - Original camera filenames containing plate text are completely excluded.
3. Full Human Control & Vocabulary:
   - On-screen keypad supports all 28 Arabic letters (including 'ك'), Eastern digits, and '?'.
   - Physical typing supports any Unicode character without restrictive filtering.
   - Explicit fields for:
     * Transcription (Letters RTL, Digits LTR)
     * Status (readable, partially_readable, unreadable, no_plate, multiple_plates)
     * Crop assessment (yes, no, no_crop)
     * Visibility (yes, partial, no)
     * Confidence / Ambiguity (high, medium, ambiguous, uncertain_chars)
     * Unsupported / OOV checkbox
     * Lighting assessment
     * Notes
4. Client-side Draft & Export:
   - Auto-saves drafts to browser localStorage.
   - One-click export to `completed_annotations_007e.csv` with UTF-8 BOM.
"""

from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path

# All standard Arabic letters for on-screen keypad (including OOV 'ك')
ALL_ARABIC_LETTERS = [
    "أ", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر",
    "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف",
    "ق", "ك", "ل", "م", "ن", "ھ", "و", "ى", "ي", "ة"
]
EASTERN_DIGITS = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"]

def generate_independent_interface(
    triage_csv_path: Path,
    output_html_path: Path,
):
    with open(triage_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    print(f"Loaded {len(all_rows)} rows from {triage_csv_path}")

    # Build anonymous records without exposing original filename, legacy, or model predictions
    samples_data = []
    for idx, r in enumerate(all_rows, start=1):
        sample_id = f"sample_{idx:03d}"
        frame_rel = f"samples/frame_{idx:03d}.jpg"
        crop_rel = f"samples/crop_{idx:03d}.jpg"
        has_crop = (r.get("crop_is_plate", "").strip().lower() == "yes")

        # Anonymous sample record with ZERO plate text or metadata
        samples_data.append({
            "idx": idx,
            "sample_id": sample_id,
            "frame_path": frame_rel,
            "crop_path": crop_rel if has_crop else None,
            "has_crop": has_crop,
            "initial_crop_status": "yes" if has_crop else "no_crop",
        })

    rows_html = []
    for s in samples_data:
        idx = s["idx"]
        crop_html = ""
        if s["has_crop"]:
            crop_html = f'<a href="{s["crop_path"]}" target="_blank"><img src="{s["crop_path"]}" alt="Crop #{idx:03d}" class="crop-img" title="Click to inspect plate crop"></a>'
        else:
            crop_html = '<div class="no-crop-badge">No Detector Crop</div>'

        rows_html.append(f"""
        <tr class="sample-row" id="row-{idx}" data-idx="{idx}">
            <td class="idx-cell">{idx}</td>
            <td class="media-cell">
                <div class="media-container">
                    {crop_html}
                    <a href="{s["frame_path"]}" target="_blank"><img src="{s["frame_path"]}" alt="Frame #{idx:03d}" class="full-img" title="Click to view full 4K frame"></a>
                </div>
                <div class="sample-badge">Sample #{idx:03d}</div>
            </td>
            <td class="input-cell crop-plate-col">
                <select name="crop_is_plate_{idx}" id="crop_is_plate_{idx}" class="select-input" onchange="saveDraft()">
                    <option value="yes" {"selected" if s["has_crop"] else ""}>Yes (Valid Crop)</option>
                    <option value="no">No (Wrong Crop)</option>
                    <option value="no_crop" {"selected" if not s["has_crop"] else ""}>No Crop (Missed Det)</option>
                </select>
            </td>
            <td class="input-cell status-col">
                <select name="status_{idx}" id="status_{idx}" class="select-input status-select" onchange="onStatusChange({idx})">
                    <option value="" disabled selected>-- Status --</option>
                    <option value="readable">readable</option>
                    <option value="partially_readable">partially_readable</option>
                    <option value="unreadable">unreadable</option>
                    <option value="no_plate">no_plate</option>
                    <option value="multiple_plates">multiple_plates</option>
                </select>
            </td>
            <td class="input-cell letters-col">
                <input type="text" name="true_letters_{idx}" id="true_letters_{idx}"
                       placeholder="e.g. سقج / كمد" class="text-input arabic-font letter-input"
                       dir="rtl" onfocus="setActiveField(this, 'letters')" oninput="onTextInput(this)">
            </td>
            <td class="input-cell digits-col">
                <input type="text" name="true_digits_{idx}" id="true_digits_{idx}"
                       placeholder="e.g. ٨٥٦٨" class="text-input arabic-font digit-input"
                       dir="ltr" onfocus="setActiveField(this, 'digits')" oninput="onTextInput(this)">
            </td>
            <td class="input-cell unsupported-col text-center">
                <label class="checkbox-label" title="Plate has characters outside the 27 classes (e.g. ك, Latin letters)">
                    <input type="checkbox" name="unsupported_chars_{idx}" id="unsupported_chars_{idx}" onchange="saveDraft()"> OOV
                </label>
            </td>
            <td class="input-cell visible-col">
                <select name="visible_{idx}" id="visible_{idx}" class="select-input" onchange="saveDraft()">
                    <option value="yes" selected>Yes</option>
                    <option value="partial">Partial</option>
                    <option value="no">No</option>
                </select>
            </td>
            <td class="input-cell ambiguity-col">
                <select name="ambiguity_{idx}" id="ambiguity_{idx}" class="select-input" onchange="saveDraft()">
                    <option value="high" selected>High (Certain)</option>
                    <option value="medium">Medium</option>
                    <option value="ambiguous">Ambiguous</option>
                    <option value="uncertain_chars">Uncertain (?)</option>
                </select>
            </td>
            <td class="input-cell lighting-col">
                <select name="lighting_{idx}" id="lighting_{idx}" class="select-input" onchange="saveDraft()">
                    <option value="daylight" selected>Daylight</option>
                    <option value="night_ir">Night / IR</option>
                    <option value="dark">Dark</option>
                    <option value="glare">Glare / Washout</option>
                </select>
            </td>
            <td class="input-cell notes-col">
                <input type="text" name="notes_{idx}" id="notes_{idx}"
                       placeholder="Observations / notes" class="text-input notes-input" oninput="saveDraft()">
            </td>
        </tr>
        """)

    all_rows_html = "\n".join(rows_html)

    # Keypad HTML
    letters_buttons = "".join([f'<button type="button" class="key-btn letter-key" onclick="pressKey(\'{c}\')">{c}</button>' for c in ALL_ARABIC_LETTERS])
    digits_buttons = "".join([f'<button type="button" class="key-btn digit-key" onclick="pressKey(\'{d}\')">{d}</button>' for d in EASTERN_DIGITS])

    samples_json = json.dumps(samples_data, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mission 007E: Independent Gate Annotation Tool</title>
    <style>
        :root {{
            --bg-primary: #0b0f19;
            --bg-secondary: #121826;
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
            padding: 20px 24px 220px 24px;
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
        .rules-card {{
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            padding: 14px 18px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 0.88rem;
            line-height: 1.6;
        }}
        .rules-card strong {{ color: var(--accent); }}
        .table-wrap {{
            overflow-x: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--bg-secondary);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
        }}
        th {{
            background: #1e293b;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            padding: 10px 12px;
            text-align: left;
            border-bottom: 2px solid var(--border);
            white-space: nowrap;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid rgba(51, 65, 85, 0.5);
            vertical-align: middle;
        }}
        tr:hover {{ background: rgba(56, 189, 248, 0.03); }}
        .idx-cell {{ font-weight: 700; color: var(--text-secondary); text-align: center; width: 35px; }}
        .media-cell {{ width: 220px; min-width: 200px; }}
        .media-container {{ display: flex; gap: 8px; align-items: center; }}
        .crop-img {{
            width: 100px;
            height: 52px;
            object-fit: contain;
            background: #000;
            border: 1px solid var(--border);
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.15s ease;
        }}
        .crop-img:hover {{ transform: scale(1.08); border-color: var(--accent); }}
        .full-img {{
            width: 90px;
            height: 52px;
            object-fit: cover;
            border: 1px solid var(--border);
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.15s ease;
        }}
        .full-img:hover {{ transform: scale(1.08); border-color: var(--accent); }}
        .no-crop-badge {{
            width: 100px;
            height: 52px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(239, 68, 68, 0.1);
            border: 1px dashed var(--danger);
            color: #fca5a5;
            font-size: 0.75rem;
            border-radius: 4px;
            text-align: center;
            font-weight: 600;
        }}
        .sample-badge {{ font-size: 0.75rem; color: var(--text-secondary); margin-top: 4px; }}
        .select-input, .text-input {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 7px 10px;
            border-radius: 5px;
            font-size: 0.88rem;
            width: 100%;
            outline: none;
        }}
        .select-input:focus, .text-input:focus {{
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
        }}
        .arabic-font {{
            font-family: "Segoe UI", Tahoma, Arial, sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: 2px;
        }}
        .letter-input {{ width: 110px; text-align: center; }}
        .digit-input {{ width: 110px; text-align: center; }}
        .notes-input {{ width: 140px; }}
        .text-center {{ text-align: center; }}
        .checkbox-label {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 5px;
            font-size: 0.8rem;
            color: var(--text-secondary);
            cursor: pointer;
        }}

        /* Sticky Bottom Keypad */
        .keypad-bar {{
            position: fixed;
            bottom: 0; left: 0; right: 0;
            background: rgba(18, 24, 38, 0.96);
            backdrop-filter: blur(10px);
            border-top: 2px solid var(--border);
            padding: 12px 24px;
            z-index: 100;
            box-shadow: 0 -4px 20px rgba(0,0,0,0.5);
        }}
        .keypad-container {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .keypad-row {{
            display: flex;
            gap: 6px;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
        }}
        .key-btn {{
            background: #1e293b;
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 5px;
            min-width: 38px;
            height: 38px;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: "Segoe UI", Tahoma, sans-serif;
        }}
        .key-btn:hover {{
            background: var(--accent);
            color: #0b0f19;
            transform: translateY(-2px);
        }}
        .letter-key {{ border-color: #38bdf8; }}
        .digit-key {{ border-color: #f59e0b; color: #fbbf24; }}
        .special-key {{ background: #334155; min-width: 70px; font-size: 0.85rem; }}
        .uncertain-key {{ background: #991b1b; color: #fee2e2; border-color: #ef4444; }}

        /* Modal */
        .modal-overlay {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.8);
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
            max-width: 520px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.8);
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Mission 007E: Independent Gate Annotation Tool</h1>
            <div class="header-desc">100 Blinded Gate Samples &bull; Anonymous IDs &bull; Visible-Character Transcriptions &bull; UTF-8 BOM Safe Export</div>
        </div>
        <div>
            <button class="btn btn-secondary" onclick="restoreDraft()">Restore Saved Draft</button>
            <button class="btn" onclick="exportCompletedCSV()">Export Completed Annotations (007e)</button>
        </div>
    </div>

    <div class="rules-card">
        <strong>Transcription Protocol & Rules:</strong>
        <ul style="margin-left: 20px; margin-top: 5px;">
            <li><strong>Transcribe What is Visible:</strong> Write exact characters visible on plate. Do not silently normalize distinct characters (e.g. <code>ك</code> is preserved as <code>ك</code>, <code>ھ</code> as <code>ھ</code>).</li>
            <li><strong>Uncertainty:</strong> If a character is ambiguous, use <code>?</code> and set status to <code>partially_readable</code>. Do not guess.</li>
            <li><strong>No-Plate Frames:</strong> If no plate is visible, select status <code>no_plate</code> and leave letters and digits completely empty.</li>
            <li><strong>Detector Crops:</strong> Distinguish wrong crops (<code>no</code>) from missed detections where no crop exists (<code>no_crop</code>).</li>
            <li><strong>Out-of-Vocabulary:</strong> Check <strong>OOV</strong> for plates containing characters outside the 27 model classes (e.g. <code>ك</code>).</li>
        </ul>
    </div>

    <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Anonymous Media</th>
                    <th>Crop is Plate?</th>
                    <th>Status</th>
                    <th>Letters (RTL)</th>
                    <th>Digits (LTR)</th>
                    <th>OOV?</th>
                    <th>Visible?</th>
                    <th>Confidence</th>
                    <th>Lighting</th>
                    <th>Notes</th>
                </tr>
            </thead>
            <tbody>
                {all_rows_html}
            </tbody>
        </table>
    </div>

    <!-- Sticky Virtual Keypad -->
    <div class="keypad-bar">
        <div class="keypad-container">
            <div class="keypad-row">
                <span style="font-size:0.75rem; color:var(--accent); font-weight:700; margin-right:8px;">ARABIC LETTERS (RTL):</span>
                {letters_buttons}
                <button type="button" class="key-btn special-key" onclick="deleteChar()">Backspace</button>
                <button type="button" class="key-btn special-key" onclick="clearActiveField()">Clear</button>
            </div>
            <div class="keypad-row">
                <span style="font-size:0.75rem; color:#fbbf24; font-weight:700; margin-right:8px;">EASTERN DIGITS (LTR):</span>
                {digits_buttons}
                <button type="button" class="key-btn uncertain-key" onclick="pressKey('?')">? (Uncertain)</button>
            </div>
        </div>
    </div>

    <!-- Confirmation Modal -->
    <div id="confirmModal" class="modal-overlay">
        <div class="modal-card">
            <div style="font-size: 1.3rem; font-weight: 700; color: var(--success); margin-bottom: 12px;">Export Completed</div>
            <div id="modalMessage" style="color: var(--text-primary); font-size: 0.92rem; margin-bottom: 20px; line-height: 1.5;"></div>
            <button class="btn" onclick="closeModal()">Close</button>
        </div>
    </div>

    <script>
        const initialData = {samples_json};
        let activeInput = null;
        let activeFieldType = null;

        function setActiveField(elem, fieldType) {{
            activeInput = elem;
            activeFieldType = fieldType;
        }}

        function pressKey(char) {{
            if (!activeInput) return;
            const start = activeInput.selectionStart || activeInput.value.length;
            const end = activeInput.selectionEnd || activeInput.value.length;
            const val = activeInput.value;
            activeInput.value = val.substring(0, start) + char + val.substring(end);
            activeInput.selectionStart = activeInput.selectionEnd = start + char.length;
            activeInput.focus();
            saveDraft();
        }}

        function deleteChar() {{
            if (!activeInput || !activeInput.value) return;
            activeInput.value = activeInput.value.slice(0, -1);
            activeInput.focus();
            saveDraft();
        }}

        function clearActiveField() {{
            if (!activeInput) return;
            activeInput.value = "";
            activeInput.focus();
            saveDraft();
        }}

        function onTextInput(inputElem) {{
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
                    ambiguity: document.getElementById("ambiguity_" + idx).value,
                    lighting: document.getElementById("lighting_" + idx).value,
                    notes: document.getElementById("notes_" + idx).value,
                }};
            }});
            try {{
                localStorage.setItem("egyptian_alpr_007e_draft", JSON.stringify(draft));
            }} catch(e) {{}}
        }}

        function restoreDraft() {{
            try {{
                const saved = localStorage.getItem("egyptian_alpr_007e_draft");
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
                        if (rData.ambiguity) document.getElementById("ambiguity_" + idx).value = rData.ambiguity;
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
            let csv = "sample_id,crop_is_plate,unsupported_chars,status,true_letters,true_digits,plate_visible_to_eye,ambiguity,lighting,notes\\n";
            let filledCount = 0;

            initialData.forEach((row, i) => {{
                const idx = i + 1;
                const cropIsPlate = (document.getElementById("crop_is_plate_" + idx).value || "").trim();
                const unsupportedChars = document.getElementById("unsupported_chars_" + idx).checked ? "yes" : "no";
                const status = (document.getElementById("status_" + idx).value || "").trim();
                const letters = (document.getElementById("true_letters_" + idx).value || "").trim();
                const digits = (document.getElementById("true_digits_" + idx).value || "").trim();
                const visible = document.getElementById("visible_" + idx).value;
                const ambiguity = document.getElementById("ambiguity_" + idx).value;
                const lighting = document.getElementById("lighting_" + idx).value;
                const notes = (document.getElementById("notes_" + idx).value || "").trim();

                if (status || letters || digits) filledCount++;

                const safeId = '"' + row.sample_id + '"';
                const safeCrop = '"' + cropIsPlate.replace(/"/g, '""') + '"';
                const safeUnsupported = '"' + unsupportedChars.replace(/"/g, '""') + '"';
                const safeStatus = '"' + status.replace(/"/g, '""') + '"';
                const safeLetters = '"' + letters.replace(/"/g, '""') + '"';
                const safeDigits = '"' + digits.replace(/"/g, '""') + '"';
                const safeVis = '"' + visible.replace(/"/g, '""') + '"';
                const safeAmbiguity = '"' + ambiguity.replace(/"/g, '""') + '"';
                const safeLight = '"' + lighting.replace(/"/g, '""') + '"';
                const safeNotes = '"' + notes.replace(/"/g, '""') + '"';

                csv += [safeId, safeCrop, safeUnsupported, safeStatus, safeLetters, safeDigits, safeVis, safeAmbiguity, safeLight, safeNotes].join(",") + "\\n";
            }});

            const blob = new Blob(["\\uFEFF" + csv], {{ type: "text/csv;charset=utf-8;" }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "completed_annotations_007e.csv";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);

            document.getElementById("modalMessage").innerHTML = "Exported <strong>" + initialData.length + "</strong> records (" + filledCount + " annotated) to <code>completed_annotations_007e.csv</code> with UTF-8 BOM.<br><br>The original files remain untouched.<br>You can score this file using:<br><br><code>python scripts/score_labeled_triage.py artifacts/triage/completed_annotations_007e.csv</code>";
            document.getElementById("confirmModal").style.display = "flex";
        }}

        function closeModal() {{
            document.getElementById("confirmModal").style.display = "none";
        }}
    </script>
</body>
</html>
"""
    output_html_path.write_text(html_content, encoding="utf-8")
    print(f"Generated independent labeling interface to: {output_html_path}")


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent
    label_sheet_csv = root_dir / "artifacts" / "triage" / "label_sheet.csv"
    out_html = root_dir / "artifacts" / "triage" / "independent_labeling_interface.html"
    generate_independent_interface(label_sheet_csv, out_html)
