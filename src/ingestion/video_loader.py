"""
video_loader.py
---------------
Reads a patrol video file (or a directory of synthetic frames) and yields
(frame_id, timestamp_str, PIL.Image) tuples at a configurable sample rate.
"""

import cv2
import os
from datetime import datetime, timedelta
from PIL import Image
from typing import Iterator, Tuple


def extract_frames(
    video_path: str,
    sample_fps: float = 1.0,
    start_time: datetime | None = None,
) -> Iterator[Tuple[int, str, Image.Image]]:
    """
    Yield (frame_id, iso_timestamp, PIL_image) for each sampled keyframe.

    Args:
        video_path:  Path to video file (.mp4 / .avi) or image directory.
        sample_fps:  Frames to extract per second of video (default: 1 FPS).
        start_time:  Mission start datetime (default: now).
    """
    if start_time is None:
        start_time = datetime.utcnow()

    # --- Directory of images mode ---
    if os.path.isdir(video_path):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = sorted(
            f for f in os.listdir(video_path)
            if os.path.splitext(f)[1].lower() in exts
        )
        for frame_id, fname in enumerate(files):
            ts = start_time + timedelta(seconds=frame_id / sample_fps)
            img = Image.open(os.path.join(video_path, fname)).convert("RGB")
            yield frame_id, ts.strftime("%Y-%m-%dT%H:%M:%SZ"), img
        return

    # --- Video file mode ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = int(video_fps / sample_fps)  # sample every Nth frame
    frame_id = 0
    raw_frame_idx = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if raw_frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            ts = start_time + timedelta(seconds=frame_id / sample_fps)
            yield frame_id, ts.strftime("%Y-%m-%dT%H:%M:%SZ"), img
            frame_id += 1
        raw_frame_idx += 1

    cap.release()
