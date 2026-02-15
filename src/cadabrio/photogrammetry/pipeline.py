"""Photogrammetry pipeline for Cadabrio.

Reconstructs 3D models from sets of photographs or video frames,
including support for high-resolution drone footage.
Maintains scale information throughout the reconstruction process.
"""

from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from loguru import logger

from cadabrio.config.manager import ConfigManager


class ReconstructionMethod(Enum):
    """3D reconstruction methods."""

    MVS = "mvs"  # Multi-View Stereo (traditional)
    NERF = "nerf"  # Neural Radiance Fields
    GAUSSIAN_SPLATTING = "gaussian_splatting"  # 3D Gaussian Splatting


@dataclass
class PhotogrammetryJob:
    """A photogrammetry reconstruction job."""

    input_paths: list[Path]
    method: ReconstructionMethod = ReconstructionMethod.MVS
    quality: str = "high"
    output_path: Path | None = None
    progress: float = 0.0
    status: str = "pending"


class PhotogrammetryPipeline:
    """Manages photogrammetry reconstruction from images/video."""

    def __init__(self, config: ConfigManager):
        self._config = config

    def create_job(
        self, input_paths: list[str | Path], method: str | None = None
    ) -> PhotogrammetryJob:
        """Create a new photogrammetry job."""
        if method is None:
            method = self._config.get("photogrammetry", "dense_reconstruction", "mvs")

        return PhotogrammetryJob(
            input_paths=[Path(p) for p in input_paths],
            method=ReconstructionMethod(method),
            quality=self._config.get("photogrammetry", "mesh_quality", "high"),
        )

    def extract_frames_from_video(self, video_path: Path, output_dir: Path, fps: float = 2.0):
        """Extract frames from a video file for photogrammetry."""
        # TODO: Implement frame extraction using OpenCV
        logger.info(f"Frame extraction not yet implemented for: {video_path}")

    def run_job(self, job: PhotogrammetryJob):
        """Execute a photogrammetry reconstruction job."""
        # TODO: Implement reconstruction pipeline
        job.status = "not_implemented"
        logger.info("Photogrammetry pipeline execution not yet implemented")
