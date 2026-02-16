"""Blender integration for Cadabrio.

Provides communication with Blender via its Python API,
supporting project export, scene synchronization, and addon management.
"""

from pathlib import Path

from loguru import logger

from cadabrio.config.manager import ConfigManager


class BlenderIntegration:
    """Interface to Blender 3D."""

    def __init__(self, config: ConfigManager):
        self._config = config
        self._blender_path: Path | None = None

    def detect(self) -> bool:
        """Detect Blender installation."""
        configured = self._config.get("integrations", "blender_path", "")
        if configured and Path(configured).exists():
            self._blender_path = Path(configured)
            logger.info(f"Blender found at: {self._blender_path}")
            return True

        # Common Windows install locations
        common_paths = [
            Path("C:/Program Files/Blender Foundation"),
            Path.home() / "AppData" / "Local" / "Blender Foundation",
        ]
        for base in common_paths:
            if base.exists():
                blenders = sorted(base.glob("Blender*/blender.exe"), reverse=True)
                if blenders:
                    self._blender_path = blenders[0]
                    logger.info(f"Blender auto-detected: {self._blender_path}")
                    return True

        logger.info("Blender not found")
        return False

    def export_to_blender(self, filepath: str | Path):
        """Export current project as a Blender-compatible file."""
        # TODO: Implement .blend or glTF export targeting Blender
        pass

    @property
    def available(self) -> bool:
        return self._blender_path is not None

    @property
    def executable_path(self) -> Path | None:
        return self._blender_path
