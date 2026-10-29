#!/usr/bin/env python3
"""Generate Stratified Labeling Sheet and Static HTML Interface (Mission 007 Step 4).

Picks 100 non-duplicate gate images stratified across lighting and plate width:
- dHash <= 3 near-duplicate filter against picked images
- Generates artifacts/triage/label_sheet.csv
- Generates artifacts/triage/labeling_interface.html
"""

from __future__ import annotations

import csv
import datetime
import html
import json
import math
import re
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


def parse_timestamp_from_filename(filename: str) -> Optional[datetime.datetime]:
    """Extracts timestamp from filename like '20261015_013942_111_TruckEntry_...jpg'."""
    m = re.match(r"^(\d{8})_(\d{6})", filename)
    if m:
        try:
            return datetime.datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
        except Exception:
            return None
    return None


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

    # Compute dHash for each image
    print("Computing dHash for near-duplicate rejection...")
    image_data = []
    for r in all_rows:
        fn = r["file"]
        # Find path in gate_dir
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

        # Determine lighting category
        if is_gray:
            lighting = "grayscale_night"
        elif brightness < 65.0:
            lighting = "dark_color"
        else:
            lighting = "normal"

        # Determine width bucket
        if plate_w == 0:
            width_cat = "no_plate"
        elif plate_w < 100:
            width_cat = "small (<100px)"
        elif plate_w <= 160:
            width_cat = "medium (100-160px)"
        else:
            width_cat = "large (>160px)"

        ts = parse_timestamp_from_filename(fn)

        image_data.append({
            "file": fn,
            "path": p,
            "dhash": dh,
            "predicted_text": r["predicted_text"],
            "plate_confidence": float(r["plate_confidence"]),
            "plate_found": (r["plate_found"].strip().lower() == "true"),
            "plate_w_px": plate_w,
            "brightness": brightness,
            "lighting_cat": lighting,
            "width_cat": width_cat,
            "timestamp": ts,
        })

    # Group by (lighting_cat, width_cat)
    strata = defaultdict(list)
    for item in image_data:
        strata[(item["lighting_cat"], item["width_cat"])].append(item)

    print(f"Total strata formed: {len(strata)}")
    for k, v in strata.items():
        print(f"  Strata {k}: {len(v)} images")

    # Pick 100 images ensuring no near-duplicates (dHash > 3) and spacing consecutive frames
    picked: List[dict] = []
    picked_hashes: List[int] = []

    # Sort each stratum to favor distinct vehicles (spread timestamps)
    for k in strata:
        strata[k].sort(key=lambda x: (x["timestamp"].timestamp() if x["timestamp"] else 0, x["file"]))

    # Iterative round-robin selection across strata
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
                # Check near-duplicate against already picked
                if any(hamming_distance(it["dhash"], h) <= 3 for h in picked_hashes):
                    continue
                # Check timestamp distance against last picked from same vehicle/time (< 15 seconds)
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
            # Relax time spacing if needed, but strictly maintain dHash > 3
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

    print(f"\nSelected {len(picked)} stratified, non-duplicate images (target={target_count}).")

    # Write label_sheet.csv
    # Columns: file, predicted_text, true_text (EMPTY), plate_visible_to_eye (EMPTY), lighting (EMPTY), notes (EMPTY)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "predicted_text", "true_text", "plate_visible_to_eye", "lighting", "notes"])
        for it in picked:
            writer.writerow([it["file"], it["predicted_text"], "", "", "", ""])

    print(f"Saved labeling sheet to: {output_csv_path}")

    # Build static HTML labeling interface
    # For each sample, find crop if exists, else annotated
    crops_dir = output_csv_path.parent / "crops"
    annotated_dir = output_csv_path.parent / "annotated"

    html_rows = []
    for idx, it in enumerate(picked, 1):
        fn = it["file"]
        stem = Path(fn).stem
        # Look for matching crop
        matching_crops = list(crops_dir.glob(f"*{stem}*_crop.jpg"))
        crop_rel = f"crops/{matching_crops[0].name}" if matching_crops else ""
        # Look for matching annotated
        matching_ann = list(annotated_dir.glob(f"*{stem}*.jpg"))
        ann_rel = f"annotated/{matching_ann[0].name}" if matching_ann else ""

        html_rows.append(f"""
        <tr class="sample-row" id="row-{idx}">
            <td class="idx-cell">{idx}</td>
            <td class="media-cell">
                <div class="media-container">
                    {f'<a href="{ann_rel}" target="_blank"><img src="{crop_rel}" alt="Crop" class="crop-img" title="Click for full annotated frame"></a>' if crop_rel else '<div class="no-crop">No Crop</div>'}
                    {f'<a href="{ann_rel}" target="_blank"><img src="{ann_rel}" alt="Full Frame" class="full-img" title="Full Frame"></a>' if ann_rel else ''}
                </div>
                <div class="file-name">{html.escape(fn)}</div>
                <div class="meta-tag">{it['lighting_cat']} | {it['width_cat']} | RawConf: {it['plate_confidence']:.2f}</div>
            </td>
            <td class="pred-cell">
                <span class="pred-tag">{html.escape(it['predicted_text']) if it['predicted_text'] else '<em>[None]</em>'}</span>
            </td>
            <td class="input-cell">
                <input type="text" name="true_text_{idx}" id="true_text_{idx}" placeholder="e.g. سقج ٨٥٦٨" class="text-input arabic-font" dir="rtl">
            </td>
            <td class="input-cell">
                <select name="visible_{idx}" id="visible_{idx}" class="select-input">
                    <option value="yes" selected>Yes</option>
                    <option value="partial">Partial</option>
                    <option value="no">No</option>
                </select>
            </td>
            <td class="input-cell">
                <select name="lighting_{idx}" id="lighting_{idx}" class="select-input">
                    <option value="normal" {'selected' if it['lighting_cat'] == 'normal' else ''}>Normal</option>
                    <option value="dark" {'selected' if it['lighting_cat'] == 'dark_color' else ''}>Dark</option>
                    <option value="grayscale_night" {'selected' if it['lighting_cat'] == 'grayscale_night' else ''}>Night / IR</option>
                </select>
            </td>
            <td class="input-cell">
                <input type="text" name="notes_{idx}" id="notes_{idx}" placeholder="Optional notes" class="text-input">
            </td>
        </tr>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Egyptian ALPR Gate Footage Ground Truth Labeling Sheet</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #334155;
            --accent: #38bdf8;
            --accent-hover: #0ea5e9;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border: #475569;
            --success: #22c55e;
            --warning: #f59e0b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }}
        h1 {{ font-size: 1.6rem; font-weight: 700; color: var(--accent); }}
        .header-desc {{ color: var(--text-secondary); font-size: 0.95rem; margin-top: 4px; }}
        .btn {{
            background: var(--accent);
            color: #0f172a;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            font-size: 0.95rem;
            transition: all 0.2s ease;
        }}
        .btn:hover {{ background: var(--accent-hover); }}
        .table-wrap {{
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border);
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        th {{
            background: #111827;
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }}
        tr:hover td {{ background: rgba(56, 189, 248, 0.04); }}
        .idx-cell {{ font-weight: 700; color: var(--accent); text-align: center; width: 40px; }}
        .media-container {{ display: flex; gap: 10px; align-items: center; margin-bottom: 4px; }}
        .crop-img {{
            height: 70px;
            border-radius: 4px;
            border: 1px solid var(--border);
            background: #000;
            object-fit: contain;
        }}
        .full-img {{
            height: 70px;
            border-radius: 4px;
            border: 1px solid var(--border);
            background: #000;
            object-fit: contain;
        }}
        .no-crop {{
            height: 70px;
            width: 120px;
            background: #475569;
            color: #cbd5e1;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            border-radius: 4px;
        }}
        .file-name {{ font-family: monospace; font-size: 0.78rem; color: var(--text-secondary); word-break: break-all; }}
        .meta-tag {{ font-size: 0.75rem; color: #38bdf8; margin-top: 2px; }}
        .pred-tag {{
            background: rgba(56, 189, 248, 0.15);
            color: #7dd3fc;
            padding: 6px 12px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 1rem;
            display: inline-block;
            font-family: Arial, sans-serif;
            direction: rtl;
        }}
        .text-input {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: #fff;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.95rem;
            width: 100%;
        }}
        .arabic-font {{ font-size: 1.1rem; font-weight: 600; }}
        .select-input {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: #fff;
            padding: 8px 10px;
            border-radius: 4px;
            font-size: 0.9rem;
            width: 100%;
        }}
        .text-input:focus, .select-input:focus {{ outline: none; border-color: var(--accent); }}
        .instructions {{
            background: rgba(30, 41, 59, 0.8);
            border-left: 4px solid var(--accent);
            padding: 12px 18px;
            margin-bottom: 20px;
            border-radius: 4px;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Egyptian ALPR Gate Footage Ground Truth Tool</h1>
            <div class="header-desc">100 Stratified Non-Duplicate Gate Images (dHash &gt; 3) &bull; TruckEntry Gate Footage</div>
        </div>
        <button class="btn" onclick="exportCSV()">Download Filled CSV</button>
    </div>

    <div class="instructions">
        <strong>Quick Labeling Guide:</strong> Type the Egyptian plate text into <em>true_text</em> in logical order (e.g. <code>سقج ٨٥٦٨</code> or Arabic letters then numerals). Click any thumbnail to view the full resolution frame. When finished, click <strong>Download Filled CSV</strong> and save it as <code>artifacts/triage/label_sheet.csv</code>.
    </div>

    <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Visual Sample (Crop &amp; Frame)</th>
                    <th>Model Prediction</th>
                    <th>Ground Truth (true_text)</th>
                    <th>Visible to Eye</th>
                    <th>Lighting</th>
                    <th>Notes</th>
                </tr>
            </thead>
            <tbody>
                {''.join(html_rows)}
            </tbody>
        </table>
    </div>

    <script>
        const initialData = {json.dumps([{"file": it["file"], "predicted_text": it["predicted_text"]} for it in picked])};

        function exportCSV() {{
            let csv = "file,predicted_text,true_text,plate_visible_to_eye,lighting,notes\\n";
            initialData.forEach((row, i) => {{
                const idx = i + 1;
                const trueText = (document.getElementById("true_text_" + idx).value || "").trim();
                const visible = document.getElementById("visible_" + idx).value;
                const lighting = document.getElementById("lighting_" + idx).value;
                const notes = (document.getElementById("notes_" + idx).value || "").trim();

                const safeFile = '"' + row.file.replace(/"/g, '""') + '"';
                const safePred = '"' + row.predicted_text.replace(/"/g, '""') + '"';
                const safeTrue = '"' + trueText.replace(/"/g, '""') + '"';
                const safeVis = '"' + visible.replace(/"/g, '""') + '"';
                const safeLight = '"' + lighting.replace(/"/g, '""') + '"';
                const safeNotes = '"' + notes.replace(/"/g, '""') + '"';

                csv += [safeFile, safePred, safeTrue, safeVis, safeLight, safeNotes].join(",") + "\\n";
            }});

            const blob = new Blob(["\\uFEFF" + csv], {{ type: "text/csv;charset=utf-8;" }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "label_sheet.csv";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }}
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
