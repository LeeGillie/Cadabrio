"""Centralized logging facility for Cadabrio.

Provides file-based logging with rotation and retention,
structured for debugging diagnostics. Log files are stored
in the user's config directory under a logs/ subfolder.

Usage:
    from cadabrio.core.logging import setup_logging, get_log_dir
    setup_logging(config)

All modules already use loguru via `from loguru import logger`.
This module configures loguru's sinks (file output, rotation, format).
"""

import sys
from pathlib import Path

from loguru import logger
from platformdirs import user_log_dir

from cadabrio.config.manager import ConfigManager


_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

_LOG_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)

_log_dir: Path | None = None


def get_log_dir() -> Path:
    """Return the directory where log files are stored."""
    global _log_dir
    if _log_dir is None:
        _log_dir = Path(user_log_dir("Cadabrio", "Cadabrio"))
    return _log_dir


def get_current_log_path() -> Path:
    """Return the path to the current (active) log file."""
    return get_log_dir() / "cadabrio.log"


def setup_logging(config: ConfigManager):
    """Configure the logging system based on user settings.

    Call once during application startup, after config is loaded.
    Sets up:
    - Console output (with color, respects log_console_output setting)
    - File output with rotation and retention
    """
    level = config.get("logging", "log_level", "INFO")
    log_to_file = config.get("logging", "log_to_file", True)
    console_output = config.get("logging", "log_console_output", True)
    retention_days = config.get("logging", "log_retention_days", 30)
    max_size_mb = config.get("logging", "log_max_size_mb", 50)

    # Remove default loguru handler
    logger.remove()

    # Console sink
    if console_output:
        logger.add(
            sys.stderr,
            format=_LOG_FORMAT,
            level=level,
            colorize=True,
        )

    # File sink with rotation and retention
    if log_to_file:
        log_dir = get_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "cadabrio.log"

        logger.add(
            str(log_path),
            format=_LOG_FILE_FORMAT,
            level="DEBUG",  # Always capture DEBUG to file for diagnostics
            rotation=f"{max_size_mb} MB",
            retention=f"{retention_days} days",
            compression="zip",
            encoding="utf-8",
            enqueue=True,  # Thread-safe async writes
        )

        logger.info(f"Log file: {log_path}")

    logger.info(f"Logging initialized (console={level}, file=DEBUG)")
