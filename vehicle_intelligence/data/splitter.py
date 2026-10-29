"""Leakage-resistant, vehicle-grouped dataset splitting utility."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field


class SplitManifest(BaseModel):
    train_vehicles: List[str]
    val_vehicles: List[str]
    test_vehicles: List[str]
    train_plates: List[str]
    val_plates: List[str]
    test_plates: List[str]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int
    duplicate_clusters_isolated: int
    leakage_check_passed: bool
    notes: List[str] = Field(default_factory=list)


class LeakageResistantSplitter:
    """Generates reproducible train/val/test splits grouped by vehicle ID and image hash clusters."""

    def __init__(
        self,
        dataset_root: Path | str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        self.root = Path(dataset_root)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = random_seed

        total = round(train_ratio + val_ratio + test_ratio, 4)
        if total != 1.0:
            raise ValueError(f"Ratios must sum to 1.0 (Received: {total})")

    def _get_vehicle_and_plate_stems(self) -> Tuple[List[str], Dict[str, List[str]]]:
        v_dir = self.root / "EALPR Vechicles dataset" / "Vehicles"
        p_dir = self.root / "EALPR- Plates dataset"

        v_stems = sorted([f.stem for f in v_dir.glob("*") if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        p_files = sorted([f.name for f in p_dir.glob("*.png") if f.is_file()])

        v_to_plates: Dict[str, List[str]] = defaultdict(list)
        for pf in p_files:
            v_prefix = pf.split("_")[0]
            v_to_plates[v_prefix].append(pf)

        return v_stems, v_to_plates

    def _find_duplicate_clusters(self) -> List[Set[str]]:
        """Finds vehicle stems that share identical file or decoded pixel hashes."""
        v_dir = self.root / "EALPR Vechicles dataset" / "Vehicles"
        hash_to_stems = defaultdict(list)

        if v_dir.exists():
            import cv2
            for f in sorted(v_dir.glob("*")):
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    try:
                        img = cv2.imread(str(f))
                        if img is not None:
                            h = hashlib.md5(img.tobytes()).hexdigest()
                        else:
                            h = hashlib.md5(f.read_bytes()).hexdigest()
                        hash_to_stems[h].append(f.stem)
                    except Exception:
                        pass

        # Return clusters of duplicate stems
        return [set(stems) for stems in hash_to_stems.values() if len(stems) > 1]

    def generate_split(self) -> SplitManifest:
        v_stems, v_to_plates = self._get_vehicle_and_plate_stems()
        dup_clusters = self._find_duplicate_clusters()

        # Build connected components of vehicle stems so duplicates always stay together
        stem_to_cluster_id: Dict[str, int] = {}
        cluster_id_to_stems: Dict[int, Set[str]] = defaultdict(set)
        current_id = 0

        # Add duplicate clusters first
        for cluster in dup_clusters:
            for s in cluster:
                stem_to_cluster_id[s] = current_id
                cluster_id_to_stems[current_id].add(s)
            current_id += 1

        # Add remaining individual vehicle stems
        for s in v_stems:
            if s not in stem_to_cluster_id:
                stem_to_cluster_id[s] = current_id
                cluster_id_to_stems[current_id].add(s)
                current_id += 1

        # Shuffle cluster IDs deterministically
        cluster_ids = sorted(list(cluster_id_to_stems.keys()))
        rng = random.Random(self.seed)
        rng.shuffle(cluster_ids)

        # Allocate clusters into splits proportionally
        n_clusters = len(cluster_ids)
        n_train = int(round(n_clusters * self.train_ratio))
        n_val = int(round(n_clusters * self.val_ratio))

        train_cids = set(cluster_ids[:n_train])
        val_cids = set(cluster_ids[n_train:n_train + n_val])
        test_cids = set(cluster_ids[n_train + n_val:])

        train_vehicles: List[str] = []
        val_vehicles: List[str] = []
        test_vehicles: List[str] = []

        for cid in cluster_ids:
            stems = sorted(list(cluster_id_to_stems[cid]))
            if cid in train_cids:
                train_vehicles.extend(stems)
            elif cid in val_cids:
                val_vehicles.extend(stems)
            else:
                test_vehicles.extend(stems)

        # Map to plates
        train_plates: List[str] = []
        val_plates: List[str] = []
        test_plates: List[str] = []

        for v in train_vehicles:
            train_plates.extend(v_to_plates.get(v, []))
        for v in val_vehicles:
            val_plates.extend(v_to_plates.get(v, []))
        for v in test_vehicles:
            test_plates.extend(v_to_plates.get(v, []))

        # Verification check: zero intersection between splits
        v_train_set = set(train_vehicles)
        v_val_set = set(val_vehicles)
        v_test_set = set(test_vehicles)

        leakage_free = (
            len(v_train_set.intersection(v_val_set)) == 0 and
            len(v_train_set.intersection(v_test_set)) == 0 and
            len(v_val_set.intersection(v_test_set)) == 0
        )

        p_train_set = set(train_plates)
        p_val_set = set(val_plates)
        p_test_set = set(test_plates)

        plates_leakage_free = (
            len(p_train_set.intersection(p_val_set)) == 0 and
            len(p_train_set.intersection(p_test_set)) == 0 and
            len(p_val_set.intersection(p_test_set)) == 0
        )

        overall_leakage_passed = leakage_free and plates_leakage_free

        notes = [
            f"Vehicles partitioned: Train={len(train_vehicles)}, Val={len(val_vehicles)}, Test={len(test_vehicles)}.",
            f"Plates partitioned: Train={len(train_plates)}, Val={len(val_plates)}, Test={len(test_plates)}.",
            f"Isolated {len(dup_clusters)} duplicate vehicle image cluster(s) into single splits to prevent data contamination.",
            "Vehicle-level grouping strictly enforces that all plates from a given vehicle are kept in the same split.",
        ]

        return SplitManifest(
            train_vehicles=train_vehicles,
            val_vehicles=val_vehicles,
            test_vehicles=test_vehicles,
            train_plates=train_plates,
            val_plates=val_plates,
            test_plates=test_plates,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            random_seed=self.seed,
            duplicate_clusters_isolated=len(dup_clusters),
            leakage_check_passed=overall_leakage_passed,
            notes=notes,
        )

    def save_split_files(
        self,
        manifest: SplitManifest,
        output_dir: Path | str = "artifacts/splits"
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
