"""FreeCAD integration for Cadabrio.

Provides communication with FreeCAD for parametric CAD workflows,
STEP/IGES export, and engineering-focused modeling.
"""

from pathlib import Path

from loguru import logger

from cadabrio.config.manager import ConfigManager


class FreecadIntegration:
    """Interface to FreeCAD."""

    def __init__(self, config: ConfigManager):
        self._config = config
        self._freecad_path: Path | None = None

    def detect(self) -> bool:
        """Detect FreeCAD installation."""
        configured = self._config.get("integrations", "freecad_path", "")
        if configured and Path(configured).exists():
            self._freecad_path = Path(configured)
            return True

        common_paths = [
            Path("C:/Program Files/FreeCAD"),
            Path.home() / "AppData" / "Local" / "FreeCAD",
        ]
        for base in common_paths:
            exes = list(base.rglob("FreeCAD.exe")) if base.exists() else []
            if exes:
                self._freecad_path = exes[0]
                logger.info(f"FreeCAD auto-detected: {self._freecad_path}")
                return True

        logger.info("FreeCAD not found")
        return False

    @property
    def available(self) -> bool:
        return self._freecad_path is not None
