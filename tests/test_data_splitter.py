"""Tests for leakage-resistant data splitting."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from vehicle_intelligence.data.splitter import LeakageResistantSplitter


@pytest.fixture
def mock_dataset_with_duplicates(tmp_path: Path):
    root = tmp_path / "mock_dataset"
    v_dir = root / "EALPR Vechicles dataset" / "Vehicles"
    p_dir = root / "EALPR- Plates dataset"
    v_dir.mkdir(parents=True)
    p_dir.mkdir(parents=True)

    img_a = np.full((50, 50, 3), 10, dtype=np.uint8)
    img_b = np.full((50, 50, 3), 20, dtype=np.uint8)
    img_c = np.full((50, 50, 3), 30, dtype=np.uint8)

    # 0001 and 0002 are exact duplicates (same pixel content img_a)
    cv2.imwrite(str(v_dir / "0001.jpg"), img_a)
    cv2.imwrite(str(v_dir / "0002.jpg"), img_a)

    # 0003, 0004, 0005, 0006 are unique
    cv2.imwrite(str(v_dir / "0003.jpg"), img_b)
    cv2.imwrite(str(v_dir / "0004.jpg"), img_c)
    cv2.imwrite(str(v_dir / "0005.jpg"), img_a + 5)
    cv2.imwrite(str(v_dir / "0006.jpg"), img_b + 5)

    # Vehicle 0001 has 2 plate crops
    cv2.imwrite(str(p_dir / "0001_license_plate_1.png"), img_a)
    cv2.imwrite(str(p_dir / "0001_license_plate_2.png"), img_a)
    cv2.imwrite(str(p_dir / "0003_license_plate_1.png"), img_b)

    return root


def test_splitter_keeps_duplicates_in_same_split(mock_dataset_with_duplicates: Path):
    splitter = LeakageResistantSplitter(
        dataset_root=mock_dataset_with_duplicates,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        random_seed=42
    )

    manifest = splitter.generate_split()
    assert manifest.leakage_check_passed is True

    # 0001 and 0002 are exact duplicates and MUST be in the same split
    in_train = ("0001" in manifest.train_vehicles and "0002" in manifest.train_vehicles)
    in_val = ("0001" in manifest.val_vehicles and "0002" in manifest.val_vehicles)
    in_test = ("0001" in manifest.test_vehicles and "0002" in manifest.test_vehicles)
    assert in_train or in_val or in_test, "Duplicate vehicles 0001 and 0002 must be placed in the same split"


def test_splitter_groups_plates_by_vehicle(mock_dataset_with_duplicates: Path):
    splitter = LeakageResistantSplitter(
        dataset_root=mock_dataset_with_duplicates,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        random_seed=42
    )

    manifest = splitter.generate_split()
    # Both plates of vehicle 0001 must be in the same plate split
    plate1 = "0001_license_plate_1.png"
    plate2 = "0001_license_plate_2.png"

    all_splits = [manifest.train_plates, manifest.val_plates, manifest.test_plates]
    found = False
    for s in all_splits:
        if plate1 in s and plate2 in s:
            found = True
            break
    assert found, "All plate crops from vehicle 0001 must reside in the same split"


def test_splitter_determinism(mock_dataset_with_duplicates: Path):
    s1 = LeakageResistantSplitter(mock_dataset_with_duplicates, random_seed=123)
    s2 = LeakageResistantSplitter(mock_dataset_with_duplicates, random_seed=123)

    m1 = s1.generate_split()
    m2 = s2.generate_split()

    assert m1.train_vehicles == m2.train_vehicles
    assert m1.val_vehicles == m2.val_vehicles
    assert m1.test_vehicles == m2.test_vehicles
