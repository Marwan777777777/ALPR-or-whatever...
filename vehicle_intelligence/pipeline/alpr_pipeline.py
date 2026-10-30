"""End-to-End Egyptian License Plate Recognition Pipeline.
Encapsulates Plate Detection -> Crop Extraction -> Character Detection -> Reading Order -> Confidence Scoring.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from vehicle_intelligence.core.domain import ALPRPlateResult, BoundingBox
from vehicle_intelligence.ocr.reading_order import (
    CharacterBox,
    EgyptianPlateReadingOrder,
)
from vehicle_intelligence.tracking.tracker import CentroidIoUTracker


class EndToEndALPRPipeline:
    """Production ALPR Pipeline executing plate detection and character OCR."""

    def __init__(
        self,
        detector_weights: str | Path = "artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt",
        char_weights: str | Path = "artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt",
        device: str = "cuda:0",
        det_conf: float = 0.25,
        char_conf: float = 0.25,
        confidence_threshold: float = 0.82,  # Calibrated on validation split
        agnostic_nms: bool = False,
    ):
        """Initializes the ALPR pipeline.

        Args:
            detector_weights: Path to trained plate detector checkpoint.
            char_weights: Path to trained 27-class character detector checkpoint.
            device: Torch execution device.
            det_conf: Detection confidence cutoff (default: 0.25).
            char_conf: Character confidence cutoff (default: 0.25).
            confidence_threshold: Calibrated multi-tier confidence threshold (default: 0.82).
            agnostic_nms: If True, executes class-agnostic NMS across all character classes.
                Defaults to False for reproducibility. Measured validation fact: validation split
                exact match is 276/300 with agnostic_nms=True vs 275/300 with agnostic_nms=False
                (no test-split comparison prior to freezing).
        """
        self.device = device
        self.det_conf = det_conf
        self.char_conf = char_conf
        self.confidence_threshold = confidence_threshold
        self.agnostic_nms = agnostic_nms

        self.detector = YOLO(str(detector_weights))
        self.char_detector = YOLO(str(char_weights))

        # Initialize tracker for video transit replay
        self.tracker = CentroidIoUTracker()

    @staticmethod
    def compute_plate_confidence(det_conf: float, char_confs: List[float]) -> float:
        if not char_confs:
            return 0.0
        log_sum = sum(math.log(max(1e-5, c)) for c in char_confs)
        geom_mean = math.exp(log_sum / len(char_confs))
        return round(float(det_conf * geom_mean), 4)

    def read_plate_image(
        self,
        image: np.ndarray,
        source_id: str = "frame",
        timestamp: Optional[float] = None,
        save_crops_dir: Optional[Path] = None,
        save_annotated_dir: Optional[Path] = None,
        track_id: Optional[int] = None,
    ) -> List[ALPRPlateResult]:
        """Runs full ALPR pipeline on a single image array."""
        ts = timestamp if timestamp is not None else time.time()
        h, w = image.shape[:2]

        # 1. Detect plates
        res = self.detector.predict(
            source=image,
            conf=self.det_conf,
            imgsz=640,
            device=self.device,
            verbose=False,
        )[0]

        results: List[ALPRPlateResult] = []
        annotated_img = image.copy() if save_annotated_dir else None

        if res.boxes is None or len(res.boxes) == 0:
            return results

        # Process each detected plate candidate
        for p_idx, b in enumerate(res.boxes):
            bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
            det_score = float(b.conf[0])

            # Clamp coordinates safely
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)
            if bx2 <= bx1 or by2 <= by1:
                continue

            plate_box = BoundingBox(x1=bx1, y1=by1, x2=bx2, y2=by2)
            plate_crop = image[by1:by2, bx1:bx2]

            # 2. Run character detection with configurable agnostic_nms
            c_res = self.char_detector.predict(
                source=plate_crop,
                conf=self.char_conf,
                iou=0.70,
                agnostic_nms=self.agnostic_nms,
                imgsz=320,
                device=self.device,
                verbose=False,
            )[0]

            char_boxes: List[CharacterBox] = []
            char_confs: List[float] = []

            if c_res.boxes is not None and len(c_res.boxes) > 0:
                for cb in c_res.boxes:
                    c_conf = float(cb.conf[0])
                    if c_conf >= self.char_conf:
                        cls_id = int(cb.cls[0])
                        xywhn = cb.xywhn[0].tolist()
                        char_boxes.append(
                            CharacterBox(
                                class_id=cls_id,
                                x_center=xywhn[0],
                                y_center=xywhn[1],
                                width=xywhn[2],
                                height=xywhn[3],
                                confidence=c_conf,
                            )
                        )
                        char_confs.append(round(c_conf, 4))

            # 3. Reading Order Reconstruction
            plate_str, letters_str, digits_str = EgyptianPlateReadingOrder.reconstruct_plate(char_boxes)

            # 4. Composite Confidence and Needs Review Evaluation
            plate_conf = self.compute_plate_confidence(det_score, char_confs)
            review_reasons = []

            if plate_conf < self.confidence_threshold:
                review_reasons.append(
                    f"LOW_PLATE_CONFIDENCE (score={plate_conf:.3f} < threshold={self.confidence_threshold})"
                )
            if det_score < 0.50:
                review_reasons.append(f"LOW_DETECTOR_CONFIDENCE ({det_score:.3f})")
            if len(char_boxes) < 2:
                review_reasons.append(f"UNUSUAL_CHARACTER_COUNT ({len(char_boxes)} characters)")
            if any(c < 0.40 for c in char_confs):
                review_reasons.append("LOW_CHARACTER_CONFIDENCE (< 0.40)")

            needs_review = len(review_reasons) > 0

            # 5. Save crop if requested
            crop_path_str = None
            if save_crops_dir:
                save_crops_dir.mkdir(parents=True, exist_ok=True)
                clean_src = Path(source_id).stem
                c_file = save_crops_dir / f"{clean_src}_plate_{p_idx+1}.jpg"
                cv2.imwrite(str(c_file), plate_crop)
                crop_path_str = str(c_file)

            # 6. Render visual annotations
            annotated_path_str = None
            if annotated_img is not None:
                color = (0, 0, 220) if needs_review else (0, 200, 0)
                cv2.rectangle(annotated_img, (bx1, by1), (bx2, by2), color, 2)
                tag = f"{digits_str} {letters_str} [{plate_conf:.2f}]"
                if needs_review:
                    tag += " (REVIEW)"
                cv2.putText(
                    annotated_img, tag, (bx1, max(20, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

            res_item = ALPRPlateResult(
                source_id=source_id,
                timestamp=ts,
                plate_text=plate_str,
                letters=letters_str,
                digits=digits_str,
                per_character_confidences=char_confs,
                plate_confidence=plate_conf,
                detector_confidence=round(det_score, 4),
                needs_review=needs_review,
                review_reasons=review_reasons,
                bounding_box=plate_box,
                track_id=track_id,
                crop_path=crop_path_str,
                annotated_image_path=None,  # Populated after saving full image
            )
            results.append(res_item)

        if annotated_img is not None and save_annotated_dir:
            save_annotated_dir.mkdir(parents=True, exist_ok=True)
            clean_src = Path(source_id).stem
            ann_file = save_annotated_dir / f"{clean_src}_annotated.jpg"
            cv2.imwrite(str(ann_file), annotated_img)
            for r in results:
                r.annotated_image_path = str(ann_file)

        return results
