"""Tests for dataset auditing and annotation validation."""

from pathlib import Path
import pytest

from vehicle_intelligence.core.exceptions import InvalidConfigurationError
from vehicle_intelligence.data.audit import EALPRDatasetAuditor


@pytest.fixture
def mock_ealpr_dataset(tmp_path: Path):
    """Constructs a minimal synthetic EALPR directory structure for testing."""
    root = tmp_path / "mock_ealpr"
    root.mkdir()

    # Directories
    v_dir = root / "EALPR Vechicles dataset" / "Vehicles"
    vl_dir = root / "EALPR Vechicles dataset" / "Vehicles Labeling"
    p_dir = root / "EALPR- Plates dataset"
    c_dir = root / "EALPR- LP characters dataset" / "Characters"
    cl_dir = root / "EALPR- LP characters dataset" / "Characters Labeling"

    for d in [v_dir, vl_dir, p_dir, c_dir, cl_dir]:
        d.mkdir(parents=True)

    # Valid vehicle image and label
    import cv2
    import numpy as np
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    cv2.imwrite(str(v_dir / "0001.jpg"), dummy_img)
    cv2.imwrite(str(v_dir / "0002.jpg"), dummy_img)

    (vl_dir / "classes.txt").write_text("License Plate\n", encoding="utf-8")
    (vl_dir / "0001.txt").write_text("0 0.5 0.5 0.2 0.1\n", encoding="utf-8")
    # Deliberately malformed annotation in 0002.txt
    (vl_dir / "0002.txt").write_text("0 0.5 0.5\n", encoding="utf-8") # Only 3 tokens!

    # Valid plate image
    cv2.imwrite(str(p_dir / "0001_license_plate_1.png"), dummy_img)

    # Valid character labeling
    (cl_dir / "classes.txt").write_text("س\nط\nن\n", encoding="utf-8")
    (cl_dir / "0001_license_plate_1.txt").write_text("0 0.2 0.5 0.1 0.3\n1 0.4 0.5 0.1 0.3\n", encoding="utf-8")

    return root


def test_auditor_raises_on_missing_directory():
    with pytest.raises(InvalidConfigurationError):
        EALPRDatasetAuditor("non_existent_dataset_directory_123")


def test_auditor_detects_malformed_annotation(mock_ealpr_dataset: Path):
    auditor = EALPRDatasetAuditor(mock_ealpr_dataset)
    report = auditor.run_audit()

    assert report.vehicles_audit.image_count == 2
    assert report.vehicles_audit.annotation_file_count == 2 # 0001.txt and 0002.txt
    # Should catch the malformed 3-token line in 0002.txt
    errors = report.vehicles_audit.annotation_errors
    assert len(errors) == 1
    assert errors[0].file_name == "0002.txt"
    assert errors[0].error_type == "MALFORMED_TOKEN_COUNT"


def test_auditor_saves_json_and_markdown_reports(mock_ealpr_dataset: Path, tmp_path: Path):
    auditor = EALPRDatasetAuditor(mock_ealpr_dataset)
    report = auditor.run_audit()

    out_dir = tmp_path / "reports"
    json_path, md_path = auditor.save_reports(report, output_dir=out_dir)

    assert json_path.exists()
    assert md_path.exists()

    md_text = md_path.read_text(encoding="utf-8")
    assert "EALPR Dataset Audit Report" in md_text
    assert "License Plate" in md_text
