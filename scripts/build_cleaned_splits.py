#!/usr/bin/env python3
"""Builds leakage-safe EALPR dataset splits by clustering connected components of:
1. Exact pixel duplicates (identical decoded RGB arrays).
2. Perceptual near-duplicates (dHash / pHash Hamming distance <= 2).
3. Identical plate text (from character annotations).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from PIL import Image

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.audit_leakage_and_duplicates import (
    compute_dhash,
    compute_phash,
    hamming_distance,
    reconstruct_plate_text_from_file,
)


def build_cleaned_splits(
    ealpr_root: Path,
    output_manifest_path: Path,
    hamming_threshold: int = 2,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    v_dir = ealpr_root / "EALPR Vechicles dataset" / "Vehicles"
    c_lbl_dir = ealpr_root / "EALPR- LP characters dataset" / "Characters Labeling"

    vehicle_files = sorted(list(v_dir.glob("*.*")))
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    vehicle_files = [f for f in vehicle_files if f.suffix.lower() in valid_exts]
    all_stems = [f.stem for f in vehicle_files]

    print(f"Loading and hashing {len(all_stems)} vehicles...")
    pixel_hashes = {}
    dhashes = {}
    phashes = {}

    for f in vehicle_files:
        stem = f.stem
        try:
            with Image.open(f) as im:
                rgb_im = im.convert("RGB")
                pixel_hashes[stem] = hashlib.md5(rgb_im.tobytes()).hexdigest()
                dhashes[stem] = compute_dhash(rgb_im)
                phashes[stem] = compute_phash(rgb_im)
        except Exception:
            pixel_hashes[stem] = f"corrupt_{stem}"
            dhashes[stem] = 0
            phashes[stem] = 0

    # Plate text mapping
    plate_texts = defaultdict(list)
    if c_lbl_dir.is_dir():
        for lbl in c_lbl_dir.glob("*.txt"):
            stem = lbl.stem.split("_license_plate_")[0]
            txt = reconstruct_plate_text_from_file(lbl)
            if txt:
                plate_texts[stem].append(txt)

    # Disjoint Set Union (DSU)
    parent = {s: s for s in all_stems}

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # 1. Exact pixel duplicate edges
    hash_to_stems = defaultdict(list)
    for stem, h in pixel_hashes.items():
        if not h.startswith("corrupt_"):
            hash_to_stems[h].append(stem)
    for stems in hash_to_stems.values():
        for i in range(1, len(stems)):
            union(stems[0], stems[i])

    # 2. Plate text edges
    text_to_stems = defaultdict(list)
    for stem, texts in plate_texts.items():
        for t in texts:
            text_to_stems[t].append(stem)
    for stems in text_to_stems.values():
        unique_stems = sorted(list(set(stems)))
        for i in range(1, len(unique_stems)):
            union(unique_stems[0], unique_stems[i])

    # 3. Perceptual hash near-duplicate edges
    n = len(all_stems)
    for i in range(n):
        s1 = all_stems[i]
        dh1, ph1 = dhashes[s1], phashes[s1]
        if dh1 == 0 and ph1 == 0:
            continue
        for j in range(i + 1, n):
            s2 = all_stems[j]
            dh2, ph2 = dhashes[s2], phashes[s2]
            if dh2 == 0 and ph2 == 0:
                continue
            if hamming_distance(dh1, dh2) <= hamming_threshold or hamming_distance(ph1, ph2) <= hamming_threshold:
                union(s1, s2)

    # Collect clusters
    clusters_map = defaultdict(list)
    for s in all_stems:
        clusters_map[find(s)].append(s)

    clusters = list(clusters_map.values())
    print(f"Formed {len(clusters)} isolated clusters from {len(all_stems)} vehicles.")

    # Sort deterministically
    clusters.sort(key=lambda c: (len(c), c[0]), reverse=True)
    rng = random.Random(seed)
    rng.shuffle(clusters)

    total_vehicles = len(all_stems)
    target_train = int(train_ratio * total_vehicles)
    target_val = int(val_ratio * total_vehicles)

    train_stems = []
    val_stems = []
    test_stems = []

    for c in clusters:
        if len(train_stems) + len(c) <= target_train:
            train_stems.extend(c)
        elif len(val_stems) + len(c) <= target_val:
            val_stems.extend(c)
        else:
            test_stems.extend(c)

    # Sort stems for clean serialization
    train_stems.sort()
    val_stems.sort()
    test_stems.sort()

    manifest = {
        "dataset_name": "EALPR-master",
        "description": "Leakage-safe splits clustered on exact pixel duplicates, dHash/pHash near-duplicates (dist<=2), and plate text.",
        "random_seed": seed,
        "hamming_threshold": hamming_threshold,
        "total_vehicles": total_vehicles,
        "total_clusters": len(clusters),
        "train_vehicles_count": len(train_stems),
        "val_vehicles_count": len(val_stems),
        "test_vehicles_count": len(test_stems),
        "train_vehicles": train_stems,
        "val_vehicles": val_stems,
        "test_vehicles": test_stems,
    }

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Cleaned splits saved to: {output_manifest_path}")
    print(f"  Train: {len(train_stems)} ({len(train_stems)/total_vehicles*100:.1f}%)")
    print(f"  Val:   {len(val_stems)} ({len(val_stems)/total_vehicles*100:.1f}%)")
    print(f"  Test:  {len(test_stems)} ({len(test_stems)/total_vehicles*100:.1f}%)")

    return manifest


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
