#!/usr/bin/env python3
"""Mission 008: Plate Detection Miss Diagnostic Exporter.

Analyzes the 18 visible-plate misses from the 100-frame gate triage sample.
Outputs:
- artifacts/missions/mission_008/detection_miss_analysis.csv
"""

import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
root_dir = Path(__file__).resolve().parent.parent

miss_diagnoses = [
    {
        "sample_id": "sample_002",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "yes",
        "likely_failure_category": "Small or unusual plate",
        "human_locatable": "yes",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Plate mounted low on truck front chassis; downsampled from 4K to 640x640 reduces plate width below detector threshold. Fixable via multi-scale inference or ROI tile slicing.",
    },
    {
        "sample_id": "sample_004",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "partial",
        "likely_failure_category": "Plate outside frame",
        "human_locatable": "partial",
        "plausibly_fixable_by_detector_improvement": "no",
        "detailed_notes": "Truck positioned at extreme left frame boundary; left-side letters cropped out of camera field of view. Physical camera framing limitation, not detector fault.",
    },
    {
        "sample_id": "sample_006",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "partial",
        "likely_failure_category": "Oblique angle",
        "human_locatable": "yes",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Truck entering at steep oblique angle at extreme right border (x=3507). Plate width heavily compressed by perspective; detected at 1280 resolution with conf=0.33.",
    },
    {
        "sample_id": "sample_008",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "yes",
        "likely_failure_category": "Oblique angle",
        "human_locatable": "yes",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Steep turn entering gate at right border (x=3543). Detected with high confidence (conf=0.78) when inference resolution increased to 1280.",
    },
    {
        "sample_id": "sample_010",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "partial",
        "likely_failure_category": "Oblique angle",
        "human_locatable": "yes",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Persistent steep oblique yaw angle entering lane (x=3542). Successfully detected at 1280 resolution with conf=0.81.",
    },
    {
        "sample_id": "sample_012",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "yes",
        "likely_failure_category": "Low contrast / IR reflection",
        "human_locatable": "yes",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Heavy dust and road grime on lower truck bumper creating very low edge contrast against vehicle grille.",
    },
    {
        "sample_id": "sample_016",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "yes",
        "likely_failure_category": "Small or unusual plate",
        "human_locatable": "yes",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Truck entering further back in lane; plate width in 4K image is small (<120px) and lost during 640 downsampling.",
    },
    {
        "sample_id": "sample_018",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "yes",
        "likely_failure_category": "Small or unusual plate",
        "human_locatable": "yes",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Tricycle vehicle featuring square (two-line) plate mounted on non-standard side bracket. Standard detector bias toward rectangular car bumper plates misses this aspect ratio.",
    },
    {
        "sample_id": "sample_020",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "yes",
        "likely_failure_category": "Motion blur",
        "human_locatable": "yes",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Horizontal motion smearing across bumper area as vehicle transits quickly before gate arm.",
    },
    {
        "sample_id": "sample_022",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "yes",
        "likely_failure_category": "Plate occlusion",
        "human_locatable": "yes",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Plate mounted in deep recess under truck bumper overhang, casting a heavy shadow across top character half.",
    },
    {
        "sample_id": "sample_024",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "yes",
        "likely_failure_category": "Small or unusual plate",
        "human_locatable": "yes",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Plate width 155px in 4K frame; detected with conf=0.86 at 1280 resolution. Direct consequence of 640 downsampling artifact on 4K camera.",
    },
    {
        "sample_id": "sample_026",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "yes",
        "likely_failure_category": "Oblique angle",
        "human_locatable": "yes",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Truck turning sharply with ~60 deg horizontal yaw, compressing plate bounding box aspect ratio to 1.3.",
    },
    {
        "sample_id": "sample_028",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "yes",
        "likely_failure_category": "Motion blur",
        "human_locatable": "yes",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Noticeable vehicle motion blur and camera shutter lag softening plate border edges.",
    },
    {
        "sample_id": "sample_030",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "yes",
        "likely_failure_category": "Small or unusual plate",
        "human_locatable": "yes",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Square tricycle plate with OOV letter Kaf; successfully detected at 1280 resolution with conf=0.74.",
    },
    {
        "sample_id": "sample_032",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "yes",
        "likely_failure_category": "Headlight glare / blooming",
        "human_locatable": "yes",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Nighttime high-beam truck headlight glare creating an 8.35% saturation bloom directly over the front bumper.",
    },
    {
        "sample_id": "sample_034",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "partial",
        "likely_failure_category": "Headlight glare / blooming",
        "human_locatable": "partial",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Intense headlight blooming obscuring right half of plate; requires local tone mapping or adaptive exposure control.",
    },
    {
        "sample_id": "sample_036",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "partial",
        "likely_failure_category": "Low contrast / IR reflection",
        "human_locatable": "partial",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Monochrome IR night capture with severe retroreflective plate blowout and dark chassis shadow.",
    },
    {
        "sample_id": "sample_038",
        "image_dimensions": "3840x2160",
        "approximate_plate_visibility": "yes",
        "likely_failure_category": "Low contrast / IR reflection",
        "human_locatable": "yes",
        "plausibly_fixable_by_detector_improvement": "yes",
        "detailed_notes": "Night IR capture of entering tricycle; detected at 1280 resolution with conf=0.68. Affected by IR grain and sensor noise.",
    },
]

out_csv = root_dir / "artifacts" / "missions" / "mission_008" / "detection_miss_analysis.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)

with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
    fieldnames = [
        "sample_id",
        "image_dimensions",
        "approximate_plate_visibility",
        "likely_failure_category",
        "human_locatable",
        "plausibly_fixable_by_detector_improvement",
        "detailed_notes",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(miss_diagnoses)

print(f"Generated detection miss analysis: {out_csv} ({len(miss_diagnoses)} rows)")
