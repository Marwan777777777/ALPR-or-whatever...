#!/usr/bin/env python3
"""Prepares the EALPR 27-Class Character Recognition YOLO Dataset.

Extracts cropped plate images and character annotations from:
- Images: 'EALPR- Plates dataset'
- Labels: 'EALPR- LP characters dataset/Characters Labeling'

Strictly synchronizes split membership with the vehicle detection splits
from 'artifacts/splits/cleaned_split_manifest.json' to ensure zero leakage.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.core.logging import setup_logger
from vehicle_intelligence.data.audit import EALPR_CHARACTER_CLASSES


def prepare_character_dataset(
    ealpr_root: Path | str,
    manifest_path: Path | str = "artifacts/splits/cleaned_split_manifest.json",
    output_dir: Path | str = "artifacts/ealpr_characters_yolo_dataset",
) -> Path:
    logger = setup_logger("prepare_character_dataset")
    ealpr_path = Path(ealpr_root)
    manifest_file = Path(manifest_path)
    out_path = Path(output_dir)

    plates_dir = ealpr_path / "EALPR- Plates dataset"
    chars_lbl_dir = ealpr_path / "EALPR- LP characters dataset" / "Characters Labeling"

    if not plates_dir.is_dir() or not chars_lbl_dir.is_dir():
        raise FileNotFoundError(f"Missing Plates or Characters Labeling under {ealpr_path}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    splits = {
        "train": set(manifest["train_vehicles"]),
        "val": set(manifest["val_vehicles"]),
        "test": set(manifest["test_vehicles"]),
    }

    out_path.mkdir(parents=True, exist_ok=True)
    images_base = out_path / "images"
    labels_base = out_path / "labels"

    counts = {"train": 0, "val": 0, "test": 0}

    # Find all character label files
    label_files = sorted(list(chars_lbl_dir.glob("*.txt")))
    logger.info(f"Found {len(label_files)} character label files in {chars_lbl_dir}.")

    for split_name, vehicle_set in splits.items():
        img_split_dir = images_base / split_name
        lbl_split_dir = labels_base / split_name
        img_split_dir.mkdir(parents=True, exist_ok=True)
        lbl_split_dir.mkdir(parents=True, exist_ok=True)

        for lbl_path in label_files:
            # Stem format: e.g. "0001_license_plate_1"
            plate_stem = lbl_path.stem
            vehicle_stem = plate_stem.split("_license_plate_")[0]

            if vehicle_stem in vehicle_set:
                # Corresponding plate image
                img_path = plates_dir / f"{plate_stem}.png"
                if not img_path.is_file():
                    continue

                dest_img = img_split_dir / f"{plate_stem}.png"
                dest_lbl = lbl_split_dir / f"{plate_stem}.txt"

                if not dest_img.exists():
                    try:
                        os.link(img_path, dest_img)
                    except (OSError, NotImplementedError):
                        shutil.copy2(img_path, dest_img)

                if not dest_lbl.exists():
                    try:
                        os.link(lbl_path, dest_lbl)
                    except (OSError, NotImplementedError):
                        shutil.copy2(lbl_path, dest_lbl)

                counts[split_name] += 1

        logger.info(f"Split '{split_name}': {counts[split_name]} cropped plates with character annotations.")

    # Generate dataset.yaml
    yaml_lines = [
        "# EALPR 27-Class Character Detection Dataset Configuration",
        f"path: {out_path.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for cls_id in range(27):
        char = EALPR_CHARACTER_CLASSES.get(cls_id, f"char_{cls_id}")
        yaml_lines.append(f'  {cls_id}: "{char}"')

    dataset_yaml_path = out_path / "dataset.yaml"
    dataset_yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    logger.info(f"Generated dataset YAML at {dataset_yaml_path}")
    return dataset_yaml_path


def main():
    parser = argparse.ArgumentParser(description="Prepare character recognition YOLO dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"C:\Users\MARWAN\plate_project\images_raw\EALPR-master"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/splits/cleaned_split_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/ealpr_characters_yolo_dataset"),
    )
    args = parser.parse_args()

    prepare_character_dataset(
        ealpr_root=args.dataset_root,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
