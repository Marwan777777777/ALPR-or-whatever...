"""Visualizer utility to render and verify bounding box annotations on images."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from vehicle_intelligence.data.audit import EALPR_CHARACTER_CLASSES


class AnnotationVisualizer:
    """Renders YOLO-format annotations onto images for human inspection and QA."""

    @staticmethod
    def draw_yolo_boxes(
        image: np.ndarray,
        annotation_file: Path | str,
        class_names_map: Optional[dict[int, str]] = None,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        line_thickness: int = 2
    ) -> np.ndarray:
        """Draws normalized YOLO bounding boxes from a .txt file onto a copy of the image."""
        annot_path = Path(annotation_file)
        if not annot_path.is_file():
            return image.copy()

        h, w = image.shape[:2]
        canvas = image.copy()

        lines = annot_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            try:
                cls_id = int(parts[0])
                xc, yc, bw, bh = [float(p) for p in parts[1:]]
            except ValueError:
                continue

            # Convert normalized center-width-height to absolute pixel coordinates
            x1 = int(round((xc - bw / 2.0) * w))
            y1 = int(round((yc - bh / 2.0) * h))
            x2 = int(round((xc + bw / 2.0) * w))
            y2 = int(round((yc + bh / 2.0) * h))

            # Clamp coordinates to canvas
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            label = class_names_map.get(cls_id, str(cls_id)) if class_names_map else str(cls_id)

            # Draw box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, line_thickness)

            # Draw label banner
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size, _ = cv2.getTextSize(str(label), font, font_scale, thickness)
            tw, th = text_size
            by2 = max(0, y1 - 4)
            by1 = max(0, by2 - th - 4)
            bx2 = min(w, x1 + tw + 6)
            cv2.rectangle(canvas, (x1, by1), (bx2, by2), box_color, -1)
            cv2.putText(
                canvas,
                str(label),
                (x1 + 3, by2 - 3),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        return canvas

    @classmethod
    def export_sample_visualizations(
        cls,
        dataset_root: Path | str,
        output_dir: Path | str = "artifacts/visualizations",
        num_samples: int = 4
    ) -> List[Path]:
        """Exports visual QA images for vehicle-to-plate detection and plate-to-character detection."""
        root = Path(dataset_root)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        saved_paths: List[Path] = []

        # 1. Vehicle Plate Detection Samples
        v_img_dir = root / "EALPR Vechicles dataset" / "Vehicles"
        v_lbl_dir = root / "EALPR Vechicles dataset" / "Vehicles Labeling"

        if v_img_dir.exists() and v_lbl_dir.exists():
            v_images = sorted([f for f in v_img_dir.glob("*.jpg")])[:num_samples]
            for img_file in v_images:
                lbl_file = v_lbl_dir / f"{img_file.stem}.txt"
                if lbl_file.is_file():
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        annotated = cls.draw_yolo_boxes(
                            image=img,
                            annotation_file=lbl_file,
                            class_names_map={0: "Plate"},
                            box_color=(0, 220, 0),
                            line_thickness=3
                        )
                        target = out / f"sample_vehicle_plate_{img_file.stem}.jpg"
                        cv2.imwrite(str(target), annotated)
                        saved_paths.append(target)

        # 2. Plate Character Detection Samples
        p_img_dir = root / "EALPR- Plates dataset"
        c_lbl_dir = root / "EALPR- LP characters dataset" / "Characters Labeling"

        if p_img_dir.exists() and c_lbl_dir.exists():
            p_images = sorted([f for f in p_img_dir.glob("*.png")])[:num_samples]
            for img_file in p_images:
                lbl_file = c_lbl_dir / f"{img_file.stem}.txt"
                if lbl_file.is_file():
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        annotated = cls.draw_yolo_boxes(
                            image=img,
                            annotation_file=lbl_file,
                            class_names_map={k: f"{k}" for k in EALPR_CHARACTER_CLASSES},
                            box_color=(255, 100, 0),
                            line_thickness=2
                        )
                        target = out / f"sample_plate_chars_{img_file.stem}.png"
                        cv2.imwrite(str(target), annotated)
                        saved_paths.append(target)

        return saved_paths
