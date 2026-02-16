"""Bambu Studio integration for Cadabrio.

Provides export targeting Bambu Studio for 3D printing workflows,
including 3MF export with print settings and plate layout.
"""

from pathlib import Path

from loguru import logger

from cadabrio.config.manager import ConfigManager


class BambuStudioIntegration:
    """Interface to Bambu Studio."""

    def __init__(self, config: ConfigManager):
        self._config = config
        self._bambu_path: Path | None = None

    def detect(self) -> bool:
        """Detect Bambu Studio installation."""
        configured = self._config.get("integrations", "bambu_studio_path", "")
        if configured and Path(configured).exists():
            self._bambu_path = Path(configured)
            return True

        common_paths = [
            Path("C:/Program Files/Bambu Studio"),
            Path.home() / "AppData" / "Local" / "BambuStudio",
        ]
        for base in common_paths:
            if base.exists():
                exes = list(base.rglob("bambu-studio.exe"))
                if exes:
                    self._bambu_path = exes[0]
                    logger.info(f"Bambu Studio auto-detected: {self._bambu_path}")
                    return True

        logger.info("Bambu Studio not found")
        return False

    @property
    def available(self) -> bool:
        return self._bambu_path is not None

    @property
    def executable_path(self) -> Path | None:
        return self._bambu_path
