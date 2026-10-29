"""Unified OpenCV image and video loader supporting files and CCTV streams."""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterator, Optional, Tuple, Union
import cv2
import numpy as np

from vehicle_intelligence.core.exceptions import MediaReadError


class MediaLoader:
    """Handles loading and streaming of images and video feeds via OpenCV."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".ts"}

    @classmethod
    def load_image(cls, image_path: Union[str, Path]) -> np.ndarray:
        """Loads an image file into a BGR numpy array.

        Args:
            image_path: Path to the image file.

        Returns:
            np.ndarray of shape [H, W, 3] in BGR color order.

        Raises:
            MediaReadError: If the file does not exist or OpenCV fails to decode.
        """
        path = Path(image_path)
        if not path.is_file():
            raise MediaReadError(
                source_path=str(path),
                reason=f"File does not exist on disk."
            )

        # Using cv2.imdecode to safely handle UTF-8 and non-ASCII file paths on Windows
        try:
            image_bytes = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            raise MediaReadError(source_path=str(path), reason=str(e)) from e

        if img is None:
            raise MediaReadError(
                source_path=str(path),
                reason="OpenCV returned None (unsupported image format or corrupt file)."
            )

        return img

    @classmethod
    def stream_video(
        cls,
        source: Union[str, Path, int],
        target_fps: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Streams frames from a video file, webcam, or RTSP CCTV stream.

        Args:
            source: File path, RTSP URL, or integer device index.
            target_fps: Optional target FPS to downsample high-frame-rate streams.
            max_frames: Maximum number of frames to yield before stopping.

        Yields:
            Tuple of (frame_index, timestamp_seconds, frame_bgr_array).

        Raises:
            MediaReadError: If video stream cannot be opened.
        """
        source_str = str(source) if isinstance(source, Path) else source
        cap = cv2.VideoCapture(source_str)

        if not cap.isOpened():
            raise MediaReadError(
                source_path=str(source),
                reason="Could not open video stream with OpenCV VideoCapture."
            )

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = int(round(source_fps / target_fps)) if (target_fps and target_fps < source_fps) else 1

        raw_frame_idx = 0
        yielded_frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if raw_frame_idx % frame_interval == 0:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    yield (yielded_frames, timestamp, frame)
                    yielded_frames += 1

                    if max_frames is not None and yielded_frames >= max_frames:
                        break

                raw_frame_idx += 1
        finally:
            cap.release()
