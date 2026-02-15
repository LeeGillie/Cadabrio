"""Video importer for Cadabrio.

Handles importing video files (MP4, AVI, MOV, MKV) for frame extraction,
photogrammetry input, and reference playback. Supports high-resolution
drone footage.
"""

from pathlib import Path
from typing import Any

from loguru import logger

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def can_import(path: Path) -> bool:
    """Check if the file is a supported video format."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def import_video(path: Path) -> dict[str, Any]:
    """Import a video file. Returns metadata dict."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {path}")
        return {"type": "video", "path": str(path), "error": "Cannot open file"}

    metadata = {
        "type": "video",
        "path": str(path),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        / max(cap.get(cv2.CAP_PROP_FPS), 1),
    }
    cap.release()
    return metadata
