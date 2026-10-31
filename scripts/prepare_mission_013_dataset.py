"""Mission 013: Dataset Preparation for Domain Fine-Tuning.

Extracts ground-truth vehicle annotations into standard YOLO/RT-DETR format.
Strictly isolates Train (196 frames), Val (72 frames), and development_test_v1 (72 frames).
Ensures zero data leakage between splits.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
GATE_DIR = Path("D:/TruckEntry")
SPLITS_FILE = ROOT_DIR / "artifacts" / "missions" / "mission_012" / "benchmark_splits.json"
GT_FILE = ROOT_DIR / "artifacts" / "missions" / "mission_012" / "vehicle_ground_truth.json"
OUT_DATASET_DIR = ROOT_DIR / "artifacts" / "missions" / "mission_013" / "dataset"


def prepare_dataset():
    print("=" * 80)
    print("MISSION 013: DOMAIN DATASET PREPARATION")
    print("=" * 80)

    with open(SPLITS_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    with open(GT_FILE, "r", encoding="utf-8") as f:
        gt = json.load(f)

    # File to split map
    file_to_split = {}
    for s_name in ["train", "val", "test"]:
        for fn in splits[s_name]["filenames"]:
            file_to_split[fn] = s_name

    split_dirs = {
        "train": (OUT_DATASET_DIR / "images" / "train", OUT_DATASET_DIR / "labels" / "train"),
        "val": (OUT_DATASET_DIR / "images" / "val", OUT_DATASET_DIR / "labels" / "val"),
        "test": (OUT_DATASET_DIR / "images" / "test", OUT_DATASET_DIR / "labels" / "test"),
    }

    for img_d, lbl_d in split_dirs.values():
        img_d.mkdir(parents=True, exist_ok=True)
        lbl_d.mkdir(parents=True, exist_ok=True)

    copied_counts = {"train": 0, "val": 0, "test": 0}
    label_counts = {"train": 0, "val": 0, "test": 0}

    for sid, entry in gt.items():
        fname = entry["filename"]
        split = file_to_split[fname]
        src_img = GATE_DIR / fname
        if not src_img.is_file():
            raise FileNotFoundError(f"Source image not found: {src_img}")

        dst_img_dir, dst_lbl_dir = split_dirs[split]
        dst_img = dst_img_dir / fname
        if not dst_img.is_file():
            shutil.copy2(src_img, dst_img)
        copied_counts[split] += 1

        # Write label file
        label_file = dst_lbl_dir / (src_img.stem + ".txt")
        img_w = float(entry["image_width"])
        img_h = float(entry["image_height"])

        lines = []
        for box in entry["boxes"]:
            # Single vehicle class: 0
            x1, y1, x2, y2 = box["bbox_xyxy"]
            # Convert to center x, center y, w, h normalized
            bw = x2 - x1
            bh = y2 - y1
            cx = x1 + bw / 2.0
            cy = y1 + bh / 2.0

            norm_cx = cx / img_w
            norm_cy = cy / img_h
            norm_bw = bw / img_w
            norm_bh = bh / img_h

            lines.append(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_bw:.6f} {norm_bh:.6f}")
            label_counts[split] += 1

        with open(label_file, "w", encoding="utf-8") as lf:
            lf.write("\n".join(lines) + ("\n" if lines else ""))

    print(f"Copied images: {copied_counts}")
    print(f"Generated labels: {label_counts} boxes")

    # Generate dataset.yaml
    dataset_yaml_content = f"""# Mission 013: Domain Adaptation Dataset
path: {OUT_DATASET_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

# Classes: Single Vehicle Class
names:
  0: vehicle
"""
    yaml_path = OUT_DATASET_DIR / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(dataset_yaml_content)

    print(f"Dataset configuration saved to: {yaml_path}")


if __name__ == "__main__":
    prepare_dataset()
