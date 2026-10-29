"""Tests for leakage-resistant data splitting."""

from pathlib import Path
import sys
import cv2
import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

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


def test_splitter_groups_perceptual_near_duplicates(tmp_path: Path):
    """Validates that images with Hamming distance <= 2 are clustered together."""
    root = tmp_path / "mock_near_dup"
    v_dir = root / "EALPR Vechicles dataset" / "Vehicles"
    v_dir.mkdir(parents=True)

    base = np.zeros((64, 64, 3), dtype=np.uint8)
    base[:, :32] = 255  # Distinct vertical split pattern

    # Slightly perturbed image (1 pixel difference)
    perturbed = base.copy()
    perturbed[0, 0] = 128

    # Unrelated image (horizontal split pattern)
    unrelated = np.zeros((64, 64, 3), dtype=np.uint8)
    unrelated[:32, :] = 255

    cv2.imwrite(str(v_dir / "v1.jpg"), base)
    cv2.imwrite(str(v_dir / "v2.jpg"), perturbed)
    cv2.imwrite(str(v_dir / "v3.jpg"), unrelated)
    cv2.imwrite(str(v_dir / "v4.jpg"), np.full((64, 64, 3), 50, dtype=np.uint8))

    splitter = LeakageResistantSplitter(
        dataset_root=root,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        random_seed=42,
        hamming_threshold=2,
    )
    manifest = splitter.generate_split()
    assert manifest.leakage_check_passed is True

    # v1 and v2 are perceptual near-duplicates and must be in the same split
    in_train = ("v1" in manifest.train_vehicles and "v2" in manifest.train_vehicles)
    in_val = ("v1" in manifest.val_vehicles and "v2" in manifest.val_vehicles)
    in_test = ("v1" in manifest.test_vehicles and "v2" in manifest.test_vehicles)
    assert in_train or in_val or in_test, "Perceptual near-duplicates v1 and v2 must be in the same split"


def test_splitter_groups_identical_plate_text(tmp_path: Path):
    """Validates that distinct vehicle images with identical plate text are clustered together."""
    root = tmp_path / "mock_plate_text"
    v_dir = root / "EALPR Vechicles dataset" / "Vehicles"
    c_dir = root / "EALPR- LP characters dataset" / "Characters Labeling"
    v_dir.mkdir(parents=True)
    c_dir.mkdir(parents=True)

    # 4 distinct vehicle images
    for i in range(1, 5):
        cv2.imwrite(str(v_dir / f"veh_{i}.jpg"), np.full((64, 64, 3), i * 40, dtype=np.uint8))

    # veh_1 and veh_3 share the exact same registration text: '123 أ' (cls 18, 19, 20, 0)
    # Format: cls xc yc w h
    text_label = "18 0.1 0.5 0.1 0.3\n19 0.2 0.5 0.1 0.3\n20 0.3 0.5 0.1 0.3\n0 0.7 0.5 0.1 0.3"
    (c_dir / "veh_1_license_plate_1.txt").write_text(text_label, encoding="utf-8")
    (c_dir / "veh_3_license_plate_1.txt").write_text(text_label, encoding="utf-8")

    # veh_2 has different plate text
    diff_label = "21 0.1 0.5 0.1 0.3\n22 0.2 0.5 0.1 0.3\n23 0.3 0.5 0.1 0.3\n1 0.7 0.5 0.1 0.3"
    (c_dir / "veh_2_license_plate_1.txt").write_text(diff_label, encoding="utf-8")

    splitter = LeakageResistantSplitter(
        dataset_root=root,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        random_seed=42,
        group_by_near_duplicates=False,
        group_by_plate_text=True,
    )
    manifest = splitter.generate_split()
    assert manifest.leakage_check_passed is True

    # veh_1 and veh_3 must be clustered together
    in_train = ("veh_1" in manifest.train_vehicles and "veh_3" in manifest.train_vehicles)
    in_val = ("veh_1" in manifest.val_vehicles and "veh_3" in manifest.val_vehicles)
    in_test = ("veh_1" in manifest.test_vehicles and "veh_3" in manifest.test_vehicles)
    assert in_train or in_val or in_test, "Vehicles sharing plate text must be in the same split"


def test_build_cleaned_splits_cli_wrapper_consistency(tmp_path: Path):
    """Validates that build_cleaned_splits wrapper produces identical output to LeakageResistantSplitter."""
    from scripts.build_cleaned_splits import build_cleaned_splits

    root = tmp_path / "mock_wrapper"
    v_dir = root / "EALPR Vechicles dataset" / "Vehicles"
    v_dir.mkdir(parents=True)

    for i in range(1, 11):
        cv2.imwrite(str(v_dir / f"car_{i:02d}.jpg"), np.full((64, 64, 3), i * 20, dtype=np.uint8))

    out_json = tmp_path / "out_manifest.json"
    manifest_dict = build_cleaned_splits(
        ealpr_root=root,
        output_manifest_path=out_json,
        seed=42,
    )

    direct_splitter = LeakageResistantSplitter(
        dataset_root=root,
        random_seed=42,
    )
    direct_manifest = direct_splitter.generate_split()

    assert manifest_dict["train_vehicles"] == direct_manifest.train_vehicles
    assert manifest_dict["val_vehicles"] == direct_manifest.val_vehicles
    assert manifest_dict["test_vehicles"] == direct_manifest.test_vehicles

