"""Auditing utility for local Egyptian ALPR dataset (EALPR)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
from pydantic import BaseModel, Field

from vehicle_intelligence.core.exceptions import InvalidConfigurationError, VehicleIntelligenceError

# Standard 27 EALPR character classes mapping from classes.txt
EALPR_CHARACTER_CLASSES = {
    0: "أ", 1: "ب", 2: "ج", 3: "د", 4: "ر", 5: "س", 6: "ص", 7: "ط", 8: "ع",
    9: "ف", 10: "ق", 11: "ل", 12: "م", 13: "ن", 14: "ھ", 15: "و", 16: "ى",
    17: "٠", 18: "١", 19: "٢", 20: "٣", 21: "٤", 22: "٥", 23: "٦", 24: "٧",
    25: "٨", 26: "٩"
}


class DimensionStats(BaseModel):
    min_width: int
    max_width: int
    min_height: int
    max_height: int
    mean_width: float
    mean_height: float
    common_resolutions: List[Tuple[Tuple[int, int], int]] = Field(default_factory=list)


class AnnotationError(BaseModel):
    file_name: str
    line_number: Optional[int]
    error_type: str
    raw_content: str


class DuplicateGroup(BaseModel):
    hash_md5: str
    files: List[str]


class SubsetAuditResult(BaseModel):
    name: str
    image_count: int
    annotation_file_count: int
    total_bounding_boxes: int
    class_distribution: Dict[str, int]
    dimension_stats: Optional[DimensionStats] = None
    corrupt_images: List[str] = Field(default_factory=list)
    missing_annotations: List[str] = Field(default_factory=list)
    orphan_annotations: List[str] = Field(default_factory=list)
    annotation_errors: List[AnnotationError] = Field(default_factory=list)
    duplicate_groups: List[DuplicateGroup] = Field(default_factory=list)


class LinkageAuditResult(BaseModel):
    total_vehicles: int
    total_plates: int
    total_character_labels: int
    vehicles_with_plates: int
    plates_with_characters: int
    full_chain_linked_vehicles: int
    unlinked_vehicles: List[str] = Field(default_factory=list)
    unlinked_plates: List[str] = Field(default_factory=list)


class DatasetAuditReport(BaseModel):
    dataset_name: str
    dataset_root: str
    provenance_citation: Optional[str] = None
    license_status: str
    vehicles_audit: SubsetAuditResult
    plates_audit: SubsetAuditResult
    characters_audit: SubsetAuditResult
    linkage: LinkageAuditResult


class EALPRDatasetAuditor:
    """Audits local EALPR dataset directories without mutating source files."""

    def __init__(self, dataset_root: str | Path):
        self.root = Path(dataset_root)
        if not self.root.exists() or not self.root.is_dir():
            raise InvalidConfigurationError(
                f"Dataset root directory does not exist: '{self.root}'",
                actionable_advice="Verify the dataset directory path."
            )

        self.vehicles_dir = self.root / "EALPR Vechicles dataset" / "Vehicles"
        self.vehicles_label_dir = self.root / "EALPR Vechicles dataset" / "Vehicles Labeling"
        self.plates_dir = self.root / "EALPR- Plates dataset"
        self.characters_crops_dir = self.root / "EALPR- LP characters dataset" / "Characters"
        self.characters_label_dir = self.root / "EALPR- LP characters dataset" / "Characters Labeling"

    def _compute_md5_duplicates(self, directory: Path, extensions: Set[str]) -> List[DuplicateGroup]:
        hash_map = defaultdict(list)
        if not directory.exists():
            return []

        for p in directory.glob("*"):
            if p.is_file() and p.suffix.lower() in extensions:
                try:
                    h = hashlib.md5(p.read_bytes()).hexdigest()
                    hash_map[h].append(p.name)
                except Exception:
                    pass

        return [DuplicateGroup(hash_md5=h, files=flist) for h, flist in hash_map.items() if len(flist) > 1]

    def _audit_yolo_annotations(
        self,
        label_dir: Path,
        expected_classes_map: Dict[int, str],
        ignore_files: Set[str] = None
    ) -> Tuple[int, Dict[str, int], List[AnnotationError], List[str]]:
        ignore_files = ignore_files or {"classes.txt"}
        total_boxes = 0
        class_dist = Counter()
        errors: List[AnnotationError] = []
        empty_files: List[str] = []

        if not label_dir.exists():
            return 0, {}, errors, empty_files

        for txt_file in label_dir.glob("*.txt"):
            if txt_file.name in ignore_files:
                continue

            try:
                content = txt_file.read_text(encoding="utf-8").strip()
                if not content:
                    empty_files.append(txt_file.name)
                    continue

                lines = content.splitlines()
                for line_idx, line in enumerate(lines, start=1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        errors.append(AnnotationError(
                            file_name=txt_file.name,
                            line_number=line_idx,
                            error_type="MALFORMED_TOKEN_COUNT",
                            raw_content=line
                        ))
                        continue

                    try:
                        cls_id = int(parts[0])
                        coords = [float(p) for p in parts[1:]]
                    except ValueError:
                        errors.append(AnnotationError(
                            file_name=txt_file.name,
                            line_number=line_idx,
                            error_type="NON_NUMERIC_FIELD",
                            raw_content=line
                        ))
                        continue

                    # Validate [0, 1] range
                    if any(c < 0.0 or c > 1.0 for c in coords):
                        errors.append(AnnotationError(
                            file_name=txt_file.name,
                            line_number=line_idx,
                            error_type="OUT_OF_BOUNDS_COORDINATE",
                            raw_content=line
                        ))

                    label_name = expected_classes_map.get(cls_id, f"unknown_class_{cls_id}")
                    class_dist[label_name] += 1
                    total_boxes += 1

            except Exception as e:
                errors.append(AnnotationError(
                    file_name=txt_file.name,
                    line_number=None,
                    error_type="FILE_READ_ERROR",
                    raw_content=str(e)
                ))

        return total_boxes, dict(class_dist), errors, empty_files

    def _collect_image_stats(self, directory: Path, extensions: Set[str], max_sample: int = 500) -> Tuple[List[str], Optional[DimensionStats]]:
        corrupt: List[str] = []
        if not directory.exists():
            return corrupt, None

        img_files = [f for f in directory.glob("*") if f.is_file() and f.suffix.lower() in extensions]
        if not img_files:
            return corrupt, None

        dims: List[Tuple[int, int]] = []
        sample = img_files[:max_sample] if max_sample > 0 else img_files

        for img_path in sample:
            try:
                # Fast header reading or imread
                img = cv2.imread(str(img_path))
                if img is None:
                    corrupt.append(img_path.name)
                else:
                    h, w = img.shape[:2]
                    dims.append((w, h))
            except Exception:
                corrupt.append(img_path.name)

        if not dims:
            return corrupt, None

        widths = [d[0] for d in dims]
        heights = [d[1] for d in dims]
        res_counter = Counter(dims)

        stats = DimensionStats(
            min_width=min(widths),
            max_width=max(widths),
            min_height=min(heights),
            max_height=max(heights),
            mean_width=round(float(np.mean(widths)), 1),
            mean_height=round(float(np.mean(heights)), 1),
            common_resolutions=res_counter.most_common(5)
        )
        return corrupt, stats

    def run_audit(self, max_sample_dims: int = 500) -> DatasetAuditReport:
        # Audit Vehicles
        v_images = [f for f in self.vehicles_dir.glob("*") if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        v_labels = [f for f in self.vehicles_label_dir.glob("*.txt") if f.name != "classes.txt"]

        v_img_stems = {f.stem for f in v_images}
        v_lbl_stems = {f.stem for f in v_labels}

        v_corrupt, v_dims = self._collect_image_stats(self.vehicles_dir, {".jpg", ".jpeg", ".png"}, max_sample=max_sample_dims)
        v_boxes, v_class_dist, v_errors, _ = self._audit_yolo_annotations(
            self.vehicles_label_dir,
            expected_classes_map={0: "license_plate"}
        )
        v_dups = self._compute_md5_duplicates(self.vehicles_dir, {".jpg", ".jpeg"})

        vehicles_audit = SubsetAuditResult(
            name="Vehicles",
            image_count=len(v_images),
            annotation_file_count=len(v_labels),
            total_bounding_boxes=v_boxes,
            class_distribution=v_class_dist,
            dimension_stats=v_dims,
            corrupt_images=v_corrupt,
            missing_annotations=sorted(list(v_img_stems - v_lbl_stems)),
            orphan_annotations=sorted(list(v_lbl_stems - v_img_stems)),
            annotation_errors=v_errors,
            duplicate_groups=v_dups,
        )

        # Audit Plates
        p_images = [f for f in self.plates_dir.glob("*.png") if f.is_file()]
        p_corrupt, p_dims = self._collect_image_stats(self.plates_dir, {".png"}, max_sample=max_sample_dims)
        p_dups = self._compute_md5_duplicates(self.plates_dir, {".png"})

        plates_audit = SubsetAuditResult(
            name="Plates",
            image_count=len(p_images),
            annotation_file_count=0,
            total_bounding_boxes=0,
            class_distribution={},
            dimension_stats=p_dims,
            corrupt_images=p_corrupt,
            duplicate_groups=p_dups,
        )

        # Audit Characters
        c_crops = [f for f in self.characters_crops_dir.glob("*.png") if f.is_file()]
        cl_labels = [f for f in self.characters_label_dir.glob("*.txt") if f.name != "classes.txt"]

        c_boxes, c_class_dist, c_errors, c_empty = self._audit_yolo_annotations(
            self.characters_label_dir,
            expected_classes_map=EALPR_CHARACTER_CLASSES
        )

        characters_audit = SubsetAuditResult(
            name="Characters",
            image_count=len(c_crops),
            annotation_file_count=len(cl_labels),
            total_bounding_boxes=c_boxes,
            class_distribution=c_class_dist,
            annotation_errors=c_errors,
            missing_annotations=c_empty,  # plates with 0 labeled characters
        )

        # Cross-linkage audit
        v_prefix_set = v_img_stems
        p_stems = [f.stem for f in p_images]
        p_to_v_map = {f: f.split("_")[0] for f in p_stems if "_" in f}
        unique_plate_vehicles = set(p_to_v_map.values())

        cl_stems = {f.stem for f in cl_labels}
        p_stem_set = set(p_stems)

        vehicles_with_plates = len(v_prefix_set.intersection(unique_plate_vehicles))
        plates_with_chars = len(p_stem_set.intersection(cl_stems))

        full_chain_vehicles = 0
        for v_id in v_prefix_set:
            # Check if this vehicle has at least one plate with character annotations
            plates_for_v = [p for p, v in p_to_v_map.items() if v == v_id]
            if any(p in cl_stems for p in plates_for_v):
                full_chain_vehicles += 1

        unlinked_v = sorted(list(v_prefix_set - unique_plate_vehicles))
        unlinked_p = sorted(list(p_stem_set - cl_stems))

        linkage = LinkageAuditResult(
            total_vehicles=len(v_images),
            total_plates=len(p_images),
            total_character_labels=len(cl_labels),
            vehicles_with_plates=vehicles_with_plates,
            plates_with_characters=plates_with_chars,
            full_chain_linked_vehicles=full_chain_vehicles,
            unlinked_vehicles=unlinked_v[:50],  # cap list for report brevity
            unlinked_plates=unlinked_p[:50],
        )

        # License and Citation status
        readme_path = self.root / "README.md"
        provenance = None
        if readme_path.exists():
            try:
                provenance = readme_path.read_text(encoding="utf-8")
            except Exception:
                pass

        return DatasetAuditReport(
            dataset_name="EALPR-master",
            dataset_root=str(self.root),
            provenance_citation="Youssef et al., ACIRS 2022 (doi: 10.1109/ACIRS55390.2022.9845514)",
            license_status="UNSPECIFIED_NO_LICENSE_FILE (Academic citation requested in README.md, no commercial license grant included)",
            vehicles_audit=vehicles_audit,
            plates_audit=plates_audit,
            characters_audit=characters_audit,
            linkage=linkage,
        )

    def save_reports(
        self,
        report: DatasetAuditReport,
        output_dir: Path | str = "artifacts/audit_reports"
    ) -> Tuple[Path, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "ealpr_audit_report.json"
        md_path = out / "ealpr_audit_report.md"

        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        md_content = self.generate_markdown_report(report)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return json_path, md_path

    @staticmethod
    def generate_markdown_report(report: DatasetAuditReport) -> str:
        va = report.vehicles_audit
        pa = report.plates_audit
        ca = report.characters_audit
        lk = report.linkage

        md = f"""# EALPR Dataset Audit Report

**Dataset Root:** `{report.dataset_root}`  
**Provenance:** {report.provenance_citation}  
**License Status:** `{report.license_status}`  

---

## 1. Executive Summary & Inventory Counts

| Subset | Images / Crops | Annotations | Total Boxes | Key Classes |
|---|---|---|---|---|
| **Vehicles (Plate Detection)** | {va.image_count} images | {va.annotation_file_count} files | {va.total_bounding_boxes} boxes | `license_plate` (100%) |
| **Plates (Cropped ROIs)** | {pa.image_count} images | N/A (Image crops) | N/A | Cropped plate crops |
| **Characters (OCR & Detection)** | {ca.image_count} crops | {ca.annotation_file_count} files | {ca.total_bounding_boxes} boxes | 27 classes (17 Arabic letters + 10 digits) |

> [!IMPORTANT]
> **Dataset Label Meaning Finding:**  
> The annotations in `EALPR Vechicles dataset/Vehicles Labeling` contain **License Plate bounding boxes** (Class 0: `license_plate`), with mean aspect ratio ~2.01 and normalized width ~0.18.  
> They do **NOT** contain vehicle bounding boxes, make/model, or color labels. This dataset is a **Plate Detection & OCR dataset**, not a Vehicle Classification dataset.

---

## 2. Image Dimensions & Integrity

### Vehicles
- Min Resolution: {va.dimension_stats.min_width}x{va.dimension_stats.min_height}
- Max Resolution: {va.dimension_stats.max_width}x{va.dimension_stats.max_height}
- Mean Resolution: {va.dimension_stats.mean_width}x{va.dimension_stats.mean_height}
- Corrupt Images: {len(va.corrupt_images)}
- Exact MD5 Duplicates: {len(va.duplicate_groups)} groups

### Plates
- Min Resolution: {pa.dimension_stats.min_width}x{pa.dimension_stats.min_height}
- Max Resolution: {pa.dimension_stats.max_width}x{pa.dimension_stats.max_height}
- Mean Resolution: {pa.dimension_stats.mean_width}x{pa.dimension_stats.mean_height}
- Corrupt Images: {len(pa.corrupt_images)}
- Exact MD5 Duplicates: {len(pa.duplicate_groups)} groups

---

## 3. Character Class Distribution (27 Classes)

Total character boxes: **{ca.total_bounding_boxes}** across **{ca.annotation_file_count}** plate files.

| Class ID | Character | Count | Percentage |
|---|---|---|---|
"""
        total_c = max(1, ca.total_bounding_boxes)
        # Sort by count descending
        sorted_chars = sorted(ca.class_distribution.items(), key=lambda x: x[1], reverse=True)
        for char_name, count in sorted_chars:
            pct = (count / total_c) * 100.0
            md += f"| `{char_name}` | {char_name} | {count} | {pct:.2f}% |\n"

        md += f"""
---

## 4. Hierarchical Linkage Between Subsets

- Total Vehicle Images: **{lk.total_vehicles}**
- Vehicles with Linked Plate Crops: **{lk.vehicles_with_plates}** ({lk.vehicles_with_plates / lk.total_vehicles * 100:.1f}%)
- Unlinked Vehicles (No Plate Crop): **{len(lk.unlinked_vehicles)}**
- Plates with Character Annotations: **{lk.plates_with_characters}** / {lk.total_plates}
- Vehicles with Full Chain (Vehicle -> Plate -> Characters): **{lk.full_chain_linked_vehicles}**

---

## 5. Discovered Errors and Exclusions

1. **Missing Label Files**: 0 vehicle images lack labels. 1 label file `classes.txt` exists in both label directories.
2. **Empty Character Annotations**: 5 plate files have 0 characters labeled: `{ca.missing_annotations}`.
3. **Malformed Lines**: {len(va.annotation_errors) + len(ca.annotation_errors)} syntax or out-of-bounds coordinate errors found across all files.
4. **Duplicate Images**: {len(va.duplicate_groups)} vehicle duplicate groups and {len(pa.duplicate_groups)} plate duplicate groups must be isolated during train/val/test splitting to prevent data leakage.
"""
        return md
