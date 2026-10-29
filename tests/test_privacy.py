"""Tests for privacy redaction and data retention lifecycle."""

import time
from pathlib import Path
import numpy as np

from vehicle_intelligence.core.domain import BoundingBox
from vehicle_intelligence.core.privacy import PrivacyRedactor, RetentionPolicyManager


def test_privacy_blur_roi():
    # Create high-contrast test image (black with sharp white center)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = 255

    box = BoundingBox(x1=30, y1=30, x2=70, y2=70)
    blurred = PrivacyRedactor.blur_roi(img, box, kernel_size=25)

    # Pixel intensity in the sharp square must have smoothed / diffused
    assert not np.array_equal(img[45:55, 45:55], blurred[45:55, 45:55])


def test_privacy_anonymize_plate_text():
    h1 = PrivacyRedactor.anonymize_plate_text("س ع د 1234")
    h2 = PrivacyRedactor.anonymize_plate_text("س ع د 1234")
    h3 = PrivacyRedactor.anonymize_plate_text("ب ج د 5678")

    assert h1 == h2, "Anonymization must be deterministic for identical plate text"
    assert h1 != h3, "Different plates must produce different hashes"
    assert len(h1) == 16


def test_retention_policy_pruning(tmp_path: Path):
    manager = RetentionPolicyManager(raw_retention_days=1, metadata_retention_days=3)

    # Test file created recently
    recent_file = tmp_path / "recent_event.jpg"
    recent_file.write_text("dummy", encoding="utf-8")

    pruned = manager.scan_and_prune_expired_files(tmp_path)
    assert len(pruned) == 0
    assert recent_file.exists()
