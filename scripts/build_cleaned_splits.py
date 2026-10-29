#!/usr/bin/env python3
"""Thin CLI wrapper around canonical LeakageResistantSplitter to build cleaned EALPR splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.data.splitter import LeakageResistantSplitter


def build_cleaned_splits(
    ealpr_root: Path,
    output_manifest_path: Path,
    hamming_threshold: int = 2,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    test_ratio = round(1.0 - train_ratio - val_ratio, 4)
    splitter = LeakageResistantSplitter(
        dataset_root=ealpr_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
        hamming_threshold=hamming_threshold,
        group_by_near_duplicates=True,
        group_by_plate_text=True,
    )

    manifest = splitter.generate_split()

    manifest_dict = {
        "dataset_name": "EALPR-master",
        "description": "Leakage-safe splits clustered on exact pixel duplicates, dHash/pHash near-duplicates, and plate text.",
        "random_seed": seed,
        "hamming_threshold": hamming_threshold,
        "total_vehicles": len(manifest.train_vehicles) + len(manifest.val_vehicles) + len(manifest.test_vehicles),
        "total_clusters": manifest.total_clusters,
        "train_vehicles_count": len(manifest.train_vehicles),
        "val_vehicles_count": len(manifest.val_vehicles),
        "test_vehicles_count": len(manifest.test_vehicles),
        "train_vehicles": manifest.train_vehicles,
        "val_vehicles": manifest.val_vehicles,
        "test_vehicles": manifest.test_vehicles,
    }

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)

    print(f"Cleaned splits saved to: {output_manifest_path}")
    total_v = manifest_dict["total_vehicles"]
    print(f"  Train: {len(manifest.train_vehicles)} ({len(manifest.train_vehicles)/total_v*100:.1f}%)")
    print(f"  Val:   {len(manifest.val_vehicles)} ({len(manifest.val_vehicles)/total_v*100:.1f}%)")
    print(f"  Test:  {len(manifest.test_vehicles)} ({len(manifest.test_vehicles)/total_v*100:.1f}%)")

    return manifest_dict


def main():
    parser = argparse.ArgumentParser(description="Build leakage-safe EALPR splits.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(r"C:\Users\MARWAN\plate_project\images_raw\EALPR-master"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("artifacts/splits/cleaned_split_manifest.json"),
    )
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_cleaned_splits(
        ealpr_root=args.dataset_path,
        output_manifest_path=args.output_manifest,
        hamming_threshold=args.threshold,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
