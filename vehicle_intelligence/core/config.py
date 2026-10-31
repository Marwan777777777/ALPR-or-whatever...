"""Configuration management with pydantic-settings and YAML loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DetectorConfig(BaseModel):
    """Vehicle and Plate detector configuration."""
    model_family: str = Field(default="mock", description="Detector type: 'mock', 'yolov10', 'rtdetr', 'yolov8'")
    weights_path: Optional[str] = Field(default=None, description="Path to model weights file (.pt, .onnx, .engine)")
    confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    input_size: int = Field(default=640, description="Inference input image resolution (square)")
    imgsz: int = Field(default=640, description="Plate detector input resolution (640 baseline, 1280 experimental)")
    device: str = Field(default="cpu", description="Compute device: 'cpu', 'cuda', 'cuda:0'")
    half_precision: bool = Field(default=True, description="Use FP16 half precision on CUDA")
    agnostic_nms: bool = Field(default=False, description="Class-agnostic NMS for plate detector (unchanged)")


class OCRConfig(BaseModel):
    """Plate OCR engine configuration."""
    engine_family: str = Field(default="mock", description="OCR engine: 'mock', 'paddleocr', 'easyocr', 'custom_crnn'")
    weights_path: Optional[str] = Field(default=None)
    min_confidence_for_acceptance: float = Field(default=0.75, ge=0.0, le=1.0)
    device: str = Field(default="cpu")
    enable_arabic_transcription: bool = Field(default=True)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="IoU threshold for character NMS (Mission 008/009 standard: 0.45)")
    imgsz: int = Field(default=320, description="Character detector input resolution (square)")
    agnostic_nms: bool = Field(default=True, description="Class-agnostic NMS across character classes (Mission 008/009 standard)")


class TrackingConfig(BaseModel):
    """Multi-object tracking configuration."""
    tracker_type: str = Field(default="bytetrack", description="'bytetrack', 'sort', 'mock'")
    track_thresh: float = Field(default=0.5, ge=0.0, le=1.0)
    track_buffer: int = Field(default=30, description="Frames to keep lost tracks before retiring")
    match_thresh: float = Field(default=0.8, ge=0.0, le=1.0)


class StreamConfig(BaseModel):
    """Per-camera CCTV stream configuration."""
    camera_id: str
    source_url: str
    target_fps: float = Field(default=10.0, description="Ingestion FPS limit to prevent buffer blowup")
    roi_coordinates: Optional[List[int]] = Field(default=None, description="Optional [x1, y1, x2, y2] gate trigger zone")


class PrivacyRetentionConfig(BaseModel):
    """Privacy and data governance retention policy."""
    enable_face_blurring: bool = Field(default=True, description="Blur human faces in stored evidence")
    enable_background_masking: bool = Field(default=True, description="Mask non-vehicle surroundings")
    retention_days: int = Field(default=30, description="Days to retain raw CCTV video/evidence")
    anonymize_after_days: int = Field(default=7, description="Days after which raw plate images are purged")
    audit_logging_enabled: bool = Field(default=True, description="Record operator access to plate records")


class PathsConfig(BaseModel):
    """Local and environment-specific dataset, footage, and checkpoint directories."""
    data_dir: Path = Field(default=Path("data"), description="Root directory for local data")
    gate_dir: Optional[Path] = Field(default=None, description="Path to gate CCTV captures (e.g. D:/TruckEntry)")
    weights_dir: Path = Field(default=Path("weights"), description="Directory where model checkpoints are stored")
    artifacts_dir: Path = Field(default=Path("artifacts"), description="Directory where run artifacts are written")
    golden_dir: Path = Field(default=Path("tests/golden"), description="Directory containing regression golden samples")


class SystemConfig(BaseSettings):
    """Root system configuration."""
    model_config = SettingsConfigDict(
        env_prefix="VEHICLE_INTEL_",
        env_nested_delimiter="__",
        extra="ignore"
    )

    app_name: str = "VehicleIntelligenceSystem"
    environment: str = Field(default="development", description="'development', 'staging', 'production'")
    log_level: str = Field(default="INFO")
    log_as_json: bool = Field(default=False)

    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    privacy: PrivacyRetentionConfig = Field(default_factory=PrivacyRetentionConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    streams: List[StreamConfig] = Field(default_factory=list)

    def resolve_gate_dir(self) -> Optional[Path]:
        """Resolves gate CCTV footage directory from config or environment variables."""
        if self.paths.gate_dir and self.paths.gate_dir.is_dir():
            return self.paths.gate_dir
        env_val = os.environ.get("GATE_DIR") or os.environ.get("TRUCKENTRY_DIR")
        if env_val:
            p = Path(env_val)
            if p.is_dir():
                return p
        d_truck = Path("D:/TruckEntry")
        if d_truck.is_dir():
            return d_truck
        return None

    def ensure_directories(self) -> None:
        """Safely creates runtime directories if they do not exist."""
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.paths.weights_dir.mkdir(parents=True, exist_ok=True)
        self.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> SystemConfig:
        """Load configuration from a YAML file with environment variable fallback."""
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

