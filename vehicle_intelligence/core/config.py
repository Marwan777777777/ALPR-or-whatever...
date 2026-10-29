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
    device: str = Field(default="cpu", description="Compute device: 'cpu', 'cuda', 'cuda:0'")
    half_precision: bool = Field(default=True, description="Use FP16 half precision on CUDA")


class OCRConfig(BaseModel):
    """Plate OCR engine configuration."""
    engine_family: str = Field(default="mock", description="OCR engine: 'mock', 'paddleocr', 'easyocr', 'custom_crnn'")
    weights_path: Optional[str] = Field(default=None)
    min_confidence_for_acceptance: float = Field(default=0.75, ge=0.0, le=1.0)
    device: str = Field(default="cpu")
    enable_arabic_transcription: bool = Field(default=True)


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
    streams: List[StreamConfig] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> SystemConfig:
        """Load configuration from a YAML file with environment variable fallback."""
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)
