"""Tests for image loading, video streaming, and engine execution."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from vehicle_intelligence.core.exceptions import MediaReadError
from vehicle_intelligence.pipeline.engine import VehicleIntelligenceEngine
from vehicle_intelligence.pipeline.media_loader import MediaLoader


def test_load_image_success(tmp_path: Path):
    img_path = tmp_path / "test_vehicle.jpg"
    test_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(img_path), test_img)

    loaded = MediaLoader.load_image(img_path)
    assert loaded is not None
    assert loaded.shape == (100, 100, 3)


def test_load_image_missing_raises_error():
    with pytest.raises(MediaReadError) as exc_info:
        MediaLoader.load_image("non_existent_car_image.png")

    assert "File does not exist" in str(exc_info.value)


def test_video_stream_and_engine_execution(tmp_path: Path, synthetic_image):
    video_path = tmp_path / "test_gate_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (640, 480))

    # Write 5 frames
    for _ in range(5):
        out.write(synthetic_image)
    out.release()

    engine = VehicleIntelligenceEngine()
    stream = MediaLoader.stream_video(video_path, max_frames=3)

    processed_count = 0
    for frame_id, ts, frame in stream:
        res = engine.process_frame(frame, frame_id=frame_id, timestamp=ts)
        processed_count += 1
        assert len(res.vehicle_detections) >= 1
        assert len(res.plate_detections) >= 1
        assert len(res.plate_readings) >= 1
        assert res.latency_ms > 0

    assert processed_count == 3
