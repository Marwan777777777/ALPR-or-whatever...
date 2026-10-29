#!/usr/bin/env python3
"""Prepares the EALPR YOLO-formatted dataset for training, validation, and held-out testing.

Uses NTFS hardlinks (or copies) to avoid data duplication while strictly respecting
the leakage-resistant splits established in artifacts/splits.
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

from vehicle_intelligence.core.domain import BoundingBox
from vehicle_intelligence.core.logging import setup_logger
from vehicle_intelligence.evaluation.detection_eval import GroundTruthPlate


def prepare_dataset(
    ealpr_vehicles_dir: Path | str,
    splits_dir: Path | str = "artifacts/splits",
    output_dir: Path | str = "artifacts/ealpr_yolo_dataset",
) -> Path:
    logger = setup_logger("prepare_dataset")
    ealpr_root = Path(ealpr_vehicles_dir)
    splits_path = Path(splits_dir)
    out_path = Path(output_dir)

    v_dir = ealpr_root / "Vehicles"
    l_dir = ealpr_root / "Vehicles Labeling"

    if not v_dir.is_dir() or not l_dir.is_dir():
        raise FileNotFoundError(f"Missing Vehicles or Vehicles Labeling under {ealpr_root}")

    out_path.mkdir(parents=True, exist_ok=True)
    images_base = out_path / "images"
    labels_base = out_path / "labels"

    manifest_file = splits_path / "split_manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Split manifest not found at {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    splits = {
        "train": manifest["train_vehicles"],
        "val": manifest["val_vehicles"],
        "test": manifest["test_vehicles"],
    }

    logger.info(f"Preparing dataset with splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    gt_json_dir = splits_path
    gt_json_dir.mkdir(parents=True, exist_ok=True)

    for split_name, stems in splits.items():
        img_dest_dir = images_base / split_name
        lbl_dest_dir = labels_base / split_name
        img_dest_dir.mkdir(parents=True, exist_ok=True)
        lbl_dest_dir.mkdir(parents=True, exist_ok=True)

        gt_plates_for_eval: list[dict] = []
        linked_images = 0
        linked_labels = 0
        total_boxes = 0

        for stem in stems:
            # Locate image file (.jpg or .jpeg)
            src_img = v_dir / f"{stem}.jpg"
            if not src_img.is_file():
                src_img = v_dir / f"{stem}.jpeg"
            if not src_img.is_file():
                raise FileNotFoundError(f"Could not locate image for stem '{stem}' in {v_dir}")

            dst_img = img_dest_dir / src_img.name
            if not dst_img.exists():
                try:
                    os.link(src_img, dst_img)
                except Exception:
                    shutil.copy2(src_img, dst_img)
            linked_images += 1

            # Locate label file (.txt)
            src_lbl = l_dir / f"{stem}.txt"
            if not src_lbl.is_file():
                raise FileNotFoundError(f"Could not locate label for stem '{stem}' in {l_dir}")

            dst_lbl = lbl_dest_dir / f"{stem}.txt"
            # Read label to parse ground truths and copy
            label_text = src_lbl.read_text(encoding="utf-8")
            dst_lbl.write_text(label_text, encoding="utf-8")
            linked_labels += 1

            for line in label_text.strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        cls_id = int(parts[0])
                        xc, yc, w, h = [float(p) for p in parts[1:]]
                        x1 = xc - w / 2.0
                        y1 = yc - h / 2.0
                        x2 = xc + w / 2.0
                        y2 = yc + h / 2.0
                        gt_plate = GroundTruthPlate(
                            image_id=stem,
                            box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                            class_name="license_plate"
                        )
                        gt_plates_for_eval.append(gt_plate.model_dump())
                        total_boxes += 1
                    except ValueError:
                        pass

        # Save ground truth JSON for evaluation harness
        gt_json_file = gt_json_dir / f"{split_name}_ground_truth_plates.json"
        with open(gt_json_file, "w", encoding="utf-8") as f:
            json.dump(gt_plates_for_eval, f, indent=2)

        logger.info(
            f"Split '{split_name}': {linked_images} images, {linked_labels} labels, {total_boxes} plate boxes. Saved GT to {gt_json_file.name}"
        )

    # Create dataset.yaml
    dataset_yaml_path = out_path / "dataset.yaml"
    # Format with forward slashes for cross-platform YOLO parser compatibility
    norm_out_path = str(out_path.resolve()).replace("\\", "/")
    yaml_content = f"""# EALPR License Plate Detection Dataset Configuration
# Generated automatically from leakage-resistant split manifest (Seed: {manifest.get('random_seed', 42)})
path: {norm_out_path}
train: images/train
val: images/val
test: images/test

names:
  0: license_plate
"""
    dataset_yaml_path.write_text(yaml_content, encoding="utf-8")
    logger.info(f"Generated dataset YAML configuration at: {dataset_yaml_path}")
    return dataset_yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from EALPR splits.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=r"C:\Users\MARWAN\plate_project\images_raw\EALPR-master\EALPR Vechicles dataset",
        help="Path to EALPR Vehicles dataset folder containing 'Vehicles' and 'Vehicles Labeling'",
    )
    parser.add_argument("--splits-dir", type=str, default="artifacts/splits", help="Path to splits directory")
    parser.add_argument("--output-dir", type=str, default="artifacts/ealpr_yolo_dataset", help="Output directory")
    args = parser.parse_args()

    prepare_dataset(args.dataset_root, args.splits_dir, args.output_dir)
