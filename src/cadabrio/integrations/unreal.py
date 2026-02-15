"""Unreal Engine integration for Cadabrio.

Provides asset export targeting Unreal Engine projects,
including FBX/USD export with material setup.
"""

from pathlib import Path

from loguru import logger

from cadabrio.config.manager import ConfigManager


class UnrealIntegration:
    """Interface to Unreal Engine."""

    def __init__(self, config: ConfigManager):
        self._config = config
        self._ue_path: Path | None = None

    def detect(self) -> bool:
        """Detect Unreal Engine installation."""
        configured = self._config.get("integrations", "unreal_engine_path", "")
        if configured and Path(configured).exists():
            self._ue_path = Path(configured)
            return True

        common_paths = [
            Path("C:/Program Files/Epic Games"),
            Path("C:/Program Files (x86)/Epic Games"),
        ]
        for base in common_paths:
            if base.exists():
                engines = sorted(base.glob("UE_*/Engine"), reverse=True)
                if engines:
                    self._ue_path = engines[0].parent
                    logger.info(f"Unreal Engine auto-detected: {self._ue_path}")
                    return True

        logger.info("Unreal Engine not found")
        return False

    @property
    def available(self) -> bool:
        return self._ue_path is not None
