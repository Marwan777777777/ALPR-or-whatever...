#!/usr/bin/env python3
"""Perceptual hash (dHash/pHash) and plate text near-duplicate audit across EALPR dataset splits.

Analyzes potential dataset leakage between train, validation, and test splits:
1. Exact pixel duplicate groups (identical RGB arrays).
2. Perceptual hash near-duplicates (dHash / pHash with configurable Hamming distance).
3. Plate text grouping (reconstructed Arabic plate strings from character annotations).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
from PIL import Image

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.data.audit import EALPR_CHARACTER_CLASSES


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Computes difference hash (dHash) as a 64-bit integer.
    
    Resizes image to (hash_size + 1, hash_size), converts to grayscale,
    and compares adjacent pixels horizontally.
    """
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Flatten to integer bitstring
    hash_int = 0
    for bit in diff.flatten():
        hash_int = (hash_int << 1) | int(bit)
    return hash_int


def compute_phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Computes perceptual hash (pHash) using discrete cosine transform (DCT).
    
    Resizes to (hash_size * highfreq_factor), computes 2D DCT, extracts top-left
    low-frequency 8x8 block (excluding DC component at 0,0), and thresholds at median.
    """
    img_size = hash_size * highfreq_factor
    resized = image.convert("L").resize((img_size, img_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.float32)
    dct = cv2.dct(pixels)
    dct_lowfreq = dct[:hash_size, :hash_size]
    # Exclude DC term (0,0) when computing median
    med = np.median(dct_lowfreq.flatten()[1:])
    diff = dct_lowfreq > med
    hash_int = 0
    for bit in diff.flatten():
        hash_int = (hash_int << 1) | int(bit)
    return hash_int


def hamming_distance(h1: int, h2: int) -> int:
    """Computes the Hamming distance (number of differing bits) between two 64-bit integers."""
    return bin(h1 ^ h2).count("1")


def reconstruct_plate_text_from_file(label_path: Path) -> str:
    """Reconstructs the license plate string from character annotation file.
    
    Separates digits (classes 17-26) and letters (classes 0-16),
    sorts digits by ascending x (left-to-right),
    and letters by descending x (Arabic right-to-left).
    Returns canonical string: '{digits} {letters}' (or '{letters}' / '{digits}' if partial).
    """
    if not label_path.is_file():
        return ""
    
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    digits = []
    letters = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            x_center = float(parts[1])
            char = EALPR_CHARACTER_CLASSES.get(cls_id, "?")
            if 17 <= cls_id <= 26:
                digits.append((x_center, char))
            elif 0 <= cls_id <= 16:
                letters.append((x_center, char))
                
    # Digits: LTR (increasing x)
    digits.sort(key=lambda item: item[0])
    digits_str = "".join([d[1] for d in digits])
    
    # Letters: RTL (decreasing x, standard Arabic reading direction)
    letters.sort(key=lambda item: item[0], reverse=True)
    letters_str = "".join([l[1] for l in letters])
    
    if digits_str and letters_str:
        return f"{digits_str} {letters_str}"
    return digits_str or letters_str


def find_cross_split_leakage(
    ealpr_root: Path,
    splits_manifest_path: Path,
    hamming_threshold: int = 4,
) -> dict:
    """Audits splits for pixel duplicates, perceptual near-duplicates, and plate text matches."""
    with open(splits_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    split_membership = {}
    for stem in manifest["train_vehicles"]:
        split_membership[stem] = "train"
    for stem in manifest["val_vehicles"]:
        split_membership[stem] = "val"
    for stem in manifest["test_vehicles"]:
        split_membership[stem] = "test"

    v_dir = ealpr_root / "EALPR Vechicles dataset" / "Vehicles"
    c_lbl_dir = ealpr_root / "EALPR- LP characters dataset" / "Characters Labeling"

    vehicle_files = sorted(list(v_dir.glob("*.*")))
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    vehicle_files = [f for f in vehicle_files if f.suffix.lower() in valid_extensions]

    # 1. Gather plate texts per vehicle stem
    plate_texts_by_vehicle: Dict[str, List[str]] = defaultdict(list)
    if c_lbl_dir.is_dir():
        for lbl_path in c_lbl_dir.glob("*.txt"):
            stem = lbl_path.stem.split("_license_plate_")[0]
            txt = reconstruct_plate_text_from_file(lbl_path)
            if txt:
                plate_texts_by_vehicle[stem].append(txt)

    # 2. Compute hashes and signatures
    pixel_hashes: Dict[str, str] = {}
    dhashes: Dict[str, int] = {}
    phashes: Dict[str, int] = {}
    all_stems = []

    print(f"Computing perceptual hashes and pixel signatures for {len(vehicle_files)} images...")
    for idx, img_path in enumerate(vehicle_files):
        stem = img_path.stem
        all_stems.append(stem)
        try:
            with Image.open(img_path) as im:
                rgb_im = im.convert("RGB")
                # Decoded RGB MD5
                raw_bytes = rgb_im.tobytes()
                pixel_hashes[stem] = hashlib.md5(raw_bytes).hexdigest()
                # dHash & pHash
                dhashes[stem] = compute_dhash(rgb_im)
                phashes[stem] = compute_phash(rgb_im)
        except Exception as e:
            # Corrupted image e.g. 2018.jpg
            pixel_hashes[stem] = f"corrupt_{stem}"
            dhashes[stem] = 0
            phashes[stem] = 0

    # 3. Detect duplicate pairs & cross-split violations
    # (a) Exact pixel duplicates
    pixel_clusters = defaultdict(list)
    for stem, h in pixel_hashes.items():
        if not h.startswith("corrupt_"):
            pixel_clusters[h].append(stem)

    pixel_cross_split_pairs = []
    for h, stems in pixel_clusters.items():
        if len(stems) > 1:
            for i in range(len(stems)):
                for j in range(i + 1, len(stems)):
                    s1, s2 = stems[i], stems[j]
                    sp1 = split_membership.get(s1)
                    sp2 = split_membership.get(s2)
                    if sp1 and sp2 and sp1 != sp2:
                        pixel_cross_split_pairs.append({
                            "type": "exact_pixel_duplicate",
                            "stem_a": s1, "split_a": sp1,
                            "stem_b": s2, "split_b": sp2,
                            "pixel_hash": h
                        })

    # (b) Plate text matches
    plate_text_to_stems = defaultdict(list)
    for stem, texts in plate_texts_by_vehicle.items():
        for t in texts:
            plate_text_to_stems[t].append(stem)

    plate_text_cross_split_pairs = []
    seen_text_pairs = set()
    for text, stems in plate_text_to_stems.items():
        unique_stems = sorted(list(set(stems)))
        if len(unique_stems) > 1:
            for i in range(len(unique_stems)):
                for j in range(i + 1, len(unique_stems)):
                    s1, s2 = unique_stems[i], unique_stems[j]
                    pair_key = tuple(sorted([s1, s2]))
                    if pair_key in seen_text_pairs:
                        continue
                    seen_text_pairs.add(pair_key)
                    sp1 = split_membership.get(s1)
                    sp2 = split_membership.get(s2)
                    if sp1 and sp2 and sp1 != sp2:
                        plate_text_cross_split_pairs.append({
                            "type": "identical_plate_text",
                            "stem_a": s1, "split_a": sp1,
                            "stem_b": s2, "split_b": sp2,
                            "plate_text": text
                        })

    # (c) Perceptual hash near-duplicates
    # Compare all pairs (N=2087 is ~2.17M pairs, runs in <1 second)
    phash_cross_split_pairs = []
    dhash_cross_split_pairs = []
    n = len(all_stems)

    stems_in_splits = [s for s in all_stems if s in split_membership]
    n_splits = len(stems_in_splits)

    for i in range(n_splits):
        s1 = stems_in_splits[i]
        sp1 = split_membership[s1]
        dh1 = dhashes[s1]
        ph1 = phashes[s1]
        if dh1 == 0 and ph1 == 0:
            continue

        for j in range(i + 1, n_splits):
            s2 = stems_in_splits[j]
            sp2 = split_membership[s2]
            if sp1 == sp2:
                continue  # only interested in cross-split pairs

            dh2 = dhashes[s2]
            ph2 = phashes[s2]
            if dh2 == 0 and ph2 == 0:
                continue

            # Check dHash
            dist_d = hamming_distance(dh1, dh2)
            if dist_d <= hamming_threshold:
                dhash_cross_split_pairs.append({
                    "stem_a": s1, "split_a": sp1,
                    "stem_b": s2, "split_b": sp2,
                    "hamming_dist": dist_d,
                    "is_exact_pixel": (pixel_hashes.get(s1) == pixel_hashes.get(s2))
                })

            # Check pHash
            dist_p = hamming_distance(ph1, ph2)
            if dist_p <= hamming_threshold:
                phash_cross_split_pairs.append({
                    "stem_a": s1, "split_a": sp1,
                    "stem_b": s2, "split_b": sp2,
                    "hamming_dist": dist_p,
                    "is_exact_pixel": (pixel_hashes.get(s1) == pixel_hashes.get(s2))
                })

    return {
        "total_vehicles": len(all_stems),
        "total_vehicles_in_splits": n_splits,
        "hamming_threshold": hamming_threshold,
        "pixel_duplicate_cross_split_count": len(pixel_cross_split_pairs),
        "pixel_duplicate_cross_split_pairs": pixel_cross_split_pairs,
        "plate_text_cross_split_count": len(plate_text_cross_split_pairs),
        "plate_text_cross_split_pairs": plate_text_cross_split_pairs,
        "dhash_cross_split_count": len(dhash_cross_split_pairs),
        "dhash_cross_split_pairs": dhash_cross_split_pairs,
        "phash_cross_split_count": len(phash_cross_split_pairs),
        "phash_cross_split_pairs": phash_cross_split_pairs,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit cross-split leakage with perceptual hashes and plate texts.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(r"C:\Users\MARWAN\plate_project\images_raw\EALPR-master"),
        help="Path to EALPR root directory.",
    )
    parser.add_argument(
        "--splits-manifest",
        type=Path,
        default=Path("artifacts/splits/split_manifest.json"),
        help="Path to split manifest JSON.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help="Maximum Hamming distance for perceptual hash near-duplicate matching (default: 2).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/splits/leakage_audit_report.json"),
        help="Output path for audit report JSON.",
    )
    args = parser.parse_args()

    results = find_cross_split_leakage(
        ealpr_root=args.dataset_path,
        splits_manifest_path=args.splits_manifest,
        hamming_threshold=args.threshold,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("LEAKAGE AUDIT REPORT SUMMARY")
    print("=" * 60)
    print(f"Total Vehicles Evaluated: {results['total_vehicles_in_splits']}")
    print(f"Hamming Distance Threshold: {results['hamming_threshold']}")
    print(f"Cross-split Exact Pixel Duplicates: {results['pixel_duplicate_cross_split_count']}")
    print(f"Cross-split Plate Text Duplicates: {results['plate_text_cross_split_count']}")
    print(f"Cross-split dHash Near-Duplicates: {results['dhash_cross_split_count']}")
    print(f"Cross-split pHash Near-Duplicates: {results['phash_cross_split_count']}")
    print(f"Report saved to: {args.output_json}\n")


if __name__ == "__main__":
    main()
