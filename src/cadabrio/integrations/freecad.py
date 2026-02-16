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

        # Search Program Files for versioned folders (e.g. "FreeCAD 1.0")
        search_roots = [
            Path("C:/Program Files"),
            Path("C:/Program Files (x86)"),
            Path.home() / "AppData" / "Local",
        ]
        for root in search_roots:
            if not root.exists():
                continue
            for candidate in sorted(root.glob("FreeCAD*"), reverse=True):
                if candidate.is_dir():
                    # Check bin/ subdirectory first (FreeCAD 1.0+ layout)
                    bin_exe = candidate / "bin" / "freecad.exe"
                    if bin_exe.exists():
                        self._freecad_path = bin_exe
                        logger.info(f"FreeCAD auto-detected: {self._freecad_path}")
                        return True
                    # Fallback: search recursively (case-insensitive on Windows)
                    exes = list(candidate.rglob("freecad.exe"))
                    if exes:
                        self._freecad_path = exes[0]
                        logger.info(f"FreeCAD auto-detected: {self._freecad_path}")
                        return True

        logger.info("FreeCAD not found")
        return False

    @property
    def available(self) -> bool:
        return self._freecad_path is not None

    @property
    def executable_path(self) -> Path | None:
        return self._freecad_path
