"""Tests for configuration loading from YAML and environment variables."""

import os
from pathlib import Path
import pytest
from vehicle_intelligence.core.config import SystemConfig


def test_default_config_initialization():
    config = SystemConfig()
    assert config.app_name == "VehicleIntelligenceSystem"
    assert config.detector.model_family == "mock"
    assert config.detector.confidence_threshold == 0.45
    assert config.privacy.retention_days == 30


def test_load_config_from_yaml(tmp_path: Path):
    yaml_file = tmp_path / "custom_config.yaml"
    yaml_file.write_text(
        """
app_name: "TestGateSystem"
environment: "staging"
detector:
  model_family: "yolov10"
  confidence_threshold: 0.65
  device: "cpu"
privacy:
  retention_days: 14
""",
        encoding="utf-8"
    )

    loaded = SystemConfig.from_yaml(yaml_file)
    assert loaded.app_name == "TestGateSystem"
    assert loaded.environment == "staging"
    assert loaded.detector.model_family == "yolov10"
    assert loaded.detector.confidence_threshold == 0.65
    assert loaded.privacy.retention_days == 14


def test_load_nonexistent_yaml_raises():
    with pytest.raises(FileNotFoundError):
        SystemConfig.from_yaml("non_existent_config.yaml")
