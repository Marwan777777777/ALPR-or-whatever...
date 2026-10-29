"""Leakage-resistant, vehicle-grouped dataset splitting utility.

Clusters connected components across:
1. Exact pixel duplicates (identical decoded RGB arrays).
2. Perceptual near-duplicates (dHash / pHash Hamming distance <= threshold).
3. Identical license plate text (from character annotations).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from vehicle_intelligence.data.audit import EALPR_CHARACTER_CLASSES


class SplitManifest(BaseModel):
    train_vehicles: List[str]
    val_vehicles: List[str]
    test_vehicles: List[str]
    train_plates: List[str] = Field(default_factory=list)
    val_plates: List[str] = Field(default_factory=list)
    test_plates: List[str] = Field(default_factory=list)
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int
    duplicate_clusters_isolated: int
    leakage_check_passed: bool
    hamming_threshold: Optional[int] = None
    total_clusters: Optional[int] = None
    cluster_size_distribution: Dict[str, int] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Computes difference hash (dHash) as a 64-bit integer."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    hash_int = 0
    for bit in diff.flatten():
        hash_int = (hash_int << 1) | int(bit)
    return hash_int


def compute_phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Computes perceptual hash (pHash) using 2D DCT."""
    img_size = hash_size * highfreq_factor
    resized = image.convert("L").resize((img_size, img_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.float32)
    dct = cv2.dct(pixels)
    dct_lowfreq = dct[:hash_size, :hash_size]
    med = float(np.median(dct_lowfreq.flatten()[1:]))
    diff = dct_lowfreq > med
    hash_int = 0
    for bit in diff.flatten():
        hash_int = (hash_int << 1) | int(bit)
    return hash_int


def hamming_distance(h1: int, h2: int) -> int:
    """Computes Hamming distance between two 64-bit integer hashes."""
    return bin(h1 ^ h2).count("1")


def reconstruct_plate_text_from_file(label_path: Path) -> str:
    """Reconstructs the license plate string from character annotation file."""
    if not label_path.is_file():
        return ""
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    digits = []
    letters = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            xc = float(parts[1])
            char = EALPR_CHARACTER_CLASSES.get(cls_id, "?")
            if 17 <= cls_id <= 26:
                digits.append((xc, char))
            elif 0 <= cls_id <= 16:
                letters.append((xc, char))

    digits.sort(key=lambda x: x[0])  # Left to right
    letters.sort(key=lambda x: x[0], reverse=True)  # Arabic Right to left

    d_str = "".join(c for _, c in digits)
    l_str = "".join(c for _, c in letters)
    if d_str and l_str:
        return f"{d_str} {l_str}"
    return d_str or l_str


class LeakageResistantSplitter:
    """Canonical generator for reproducible train/val/test splits grouped by vehicle ID and multimodal clusters."""

    def __init__(
        self,
        dataset_root: Path | str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        hamming_threshold: int = 2,
        group_by_near_duplicates: bool = True,
        group_by_plate_text: bool = True,
        min_plate_text_length: int = 3,
    ):
        self.root = Path(dataset_root)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = random_seed
        self.hamming_threshold = hamming_threshold
        self.group_by_near_duplicates = group_by_near_duplicates
        self.group_by_plate_text = group_by_plate_text
        self.min_plate_text_length = min_plate_text_length

        total = round(train_ratio + val_ratio + test_ratio, 4)
        if total != 1.0:
            raise ValueError(f"Ratios must sum to 1.0 (Received: {total})")

    def _get_vehicle_and_plate_stems(self) -> Tuple[List[str], Dict[str, List[str]]]:
        v_dir = self.root / "EALPR Vechicles dataset" / "Vehicles"
        p_dir = self.root / "EALPR- Plates dataset"

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        v_stems = sorted([f.stem for f in v_dir.glob("*") if f.is_file() and f.suffix.lower() in valid_exts]) if v_dir.exists() else []
        p_files = sorted([f.name for f in p_dir.glob("*.png") if f.is_file()]) if p_dir.exists() else []

        v_to_plates: Dict[str, List[str]] = defaultdict(list)
        for pf in p_files:
            v_prefix = pf.split("_")[0]
            v_to_plates[v_prefix].append(pf)

        return v_stems, v_to_plates

    def generate_split(self) -> SplitManifest:
        v_stems, v_to_plates = self._get_vehicle_and_plate_stems()
        v_dir = self.root / "EALPR Vechicles dataset" / "Vehicles"
        c_lbl_dir = self.root / "EALPR- LP characters dataset" / "Characters Labeling"

        # Disjoint Set Union (DSU)
        parent = {s: s for s in v_stems}

        def find(i: str) -> str:
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i: str, j: str) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        # Compute hashes
        pixel_hashes: Dict[str, str] = {}
        dhashes: Dict[str, int] = {}
        phashes: Dict[str, int] = {}

        if v_dir.exists():
            for s in v_stems:
                f = v_dir / f"{s}.jpg"
                if not f.is_file():
                    f = v_dir / f"{s}.jpeg"
                if not f.is_file():
                    f = v_dir / f"{s}.png"
                if f.is_file():
                    try:
                        with Image.open(f) as im:
                            rgb_im = im.convert("RGB")
                            pixel_hashes[s] = hashlib.md5(rgb_im.tobytes()).hexdigest()
                            if self.group_by_near_duplicates:
                                dhashes[s] = compute_dhash(rgb_im)
                                phashes[s] = compute_phash(rgb_im)
                    except Exception:
                        pixel_hashes[s] = f"corrupt_{s}"

        # 1. Exact pixel duplicate edges
        hash_to_stems = defaultdict(list)
        for s, h in pixel_hashes.items():
            if not h.startswith("corrupt_"):
                hash_to_stems[h].append(s)
        for stems in hash_to_stems.values():
            for i in range(1, len(stems)):
                union(stems[0], stems[i])

        # 2. Plate text edges
        if self.group_by_plate_text and c_lbl_dir.is_dir():
            plate_texts = defaultdict(list)
            for lbl in c_lbl_dir.glob("*.txt"):
                s = lbl.stem.split("_license_plate_")[0]
                if s in parent:
                    txt = reconstruct_plate_text_from_file(lbl)
                    # Filter out partial single-letter annotations that might merge unrelated cars
                    if len(txt.replace(" ", "")) >= self.min_plate_text_length:
                        plate_texts[txt].append(s)

            for stems in plate_texts.values():
                unique_stems = sorted(list(set(stems)))
                for i in range(1, len(unique_stems)):
                    union(unique_stems[0], unique_stems[i])

        # 3. Perceptual hash near-duplicate edges
        if self.group_by_near_duplicates and dhashes and phashes:
            n = len(v_stems)
            for i in range(n):
                s1 = v_stems[i]
                dh1, ph1 = dhashes.get(s1, 0), phashes.get(s1, 0)
                if dh1 == 0 and ph1 == 0:
                    continue
                for j in range(i + 1, n):
                    s2 = v_stems[j]
                    dh2, ph2 = dhashes.get(s2, 0), phashes.get(s2, 0)
                    if dh2 == 0 and ph2 == 0:
                        continue
                    if (
                        hamming_distance(dh1, dh2) <= self.hamming_threshold
                        or hamming_distance(ph1, ph2) <= self.hamming_threshold
                    ):
                        union(s1, s2)

        # Collect clusters
        clusters_map = defaultdict(list)
        for s in v_stems:
            clusters_map[find(s)].append(s)

        clusters = list(clusters_map.values())
        duplicate_clusters_isolated = sum(1 for c in clusters if len(c) > 1)

        # Compute cluster size distribution
        size_dist = defaultdict(int)
        for c in clusters:
            size_dist[f"size_{len(c)}"] += 1

        # Sort clusters deterministically by (length descending, first stem ascending)
        clusters.sort(key=lambda c: (len(c), c[0]), reverse=True)
        rng = random.Random(self.seed)
        rng.shuffle(clusters)

        total_vehicles = len(v_stems)
        target_train = int(round(self.train_ratio * total_vehicles))
        target_val = int(round(self.val_ratio * total_vehicles))

        train_stems: List[str] = []
        val_stems: List[str] = []
        test_stems: List[str] = []

        for c in clusters:
            if len(train_stems) + len(c) <= target_train:
                train_stems.extend(c)
            elif len(val_stems) + len(c) <= target_val:
                val_stems.extend(c)
            else:
                test_stems.extend(c)

        train_stems.sort()
        val_stems.sort()
        test_stems.sort()

        # Map to plate crops
        train_plates: List[str] = []
        val_plates: List[str] = []
        test_plates: List[str] = []
        for v in train_stems:
            train_plates.extend(v_to_plates.get(v, []))
        for v in val_stems:
            val_plates.extend(v_to_plates.get(v, []))
        for v in test_stems:
            test_plates.extend(v_to_plates.get(v, []))

        # Verification check
        v_train_set = set(train_stems)
        v_val_set = set(val_stems)
        v_test_set = set(test_stems)

        vehicles_leakage_free = (
            len(v_train_set.intersection(v_val_set)) == 0
            and len(v_train_set.intersection(v_test_set)) == 0
            and len(v_val_set.intersection(v_test_set)) == 0
        )

        p_train_set = set(train_plates)
        p_val_set = set(val_plates)
        p_test_set = set(test_plates)

        plates_leakage_free = (
            len(p_train_set.intersection(p_val_set)) == 0
            and len(p_train_set.intersection(p_test_set)) == 0
            and len(p_val_set.intersection(p_test_set)) == 0
        )

        overall_passed = vehicles_leakage_free and plates_leakage_free

        notes = [
            f"Vehicles partitioned: Train={len(train_stems)}, Val={len(val_stems)}, Test={len(test_stems)}.",
            f"Plates partitioned: Train={len(train_plates)}, Val={len(val_plates)}, Test={len(test_plates)}.",
            f"Isolated {duplicate_clusters_isolated} multimodal cluster(s) with size > 1 across {len(clusters)} total components.",
            f"Near-duplicate Hamming threshold: {self.hamming_threshold}.",
            "Multimodal clustering prevents exact pixel, perceptual hash, and plate text cross-split contamination.",
        ]

        return SplitManifest(
            train_vehicles=train_stems,
            val_vehicles=val_stems,
            test_vehicles=test_stems,
            train_plates=train_plates,
            val_plates=val_plates,
            test_plates=test_plates,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            random_seed=self.seed,
            duplicate_clusters_isolated=duplicate_clusters_isolated,
            leakage_check_passed=overall_passed,
            hamming_threshold=self.hamming_threshold,
            total_clusters=len(clusters),
            cluster_size_distribution=dict(size_dist),
            notes=notes,
        )

    def save_split_files(
        self,
        manifest: SplitManifest,
        output_dir: Path | str = "artifacts/splits",
    ) -> Dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths = {
            "manifest_json": out / "split_manifest.json",
            "train_vehicles": out / "train_vehicles.txt",
            "val_vehicles": out / "val_vehicles.txt",
            "test_vehicles": out / "test_vehicles.txt",
            "train_plates": out / "train_plates.txt",
            "val_plates": out / "val_plates.txt",
            "test_plates": out / "test_plates.txt",
        }

        paths["manifest_json"].write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        paths["train_vehicles"].write_text("\n".join(manifest.train_vehicles), encoding="utf-8")
        paths["val_vehicles"].write_text("\n".join(manifest.val_vehicles), encoding="utf-8")
        paths["test_vehicles"].write_text("\n".join(manifest.test_vehicles), encoding="utf-8")
        paths["train_plates"].write_text("\n".join(manifest.train_plates), encoding="utf-8")
        paths["val_plates"].write_text("\n".join(manifest.val_plates), encoding="utf-8")
        paths["test_plates"].write_text("\n".join(manifest.test_plates), encoding="utf-8")

        return paths
