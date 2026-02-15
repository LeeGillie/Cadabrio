"""Configuration manager for Cadabrio.

Handles loading, saving, and accessing configuration values.
Configuration is stored as JSON and organized into groups,
each of which appears as a tab in the configuration editor.
"""

import json
import copy
from pathlib import Path
from typing import Any

from loguru import logger
from platformdirs import user_config_dir

from cadabrio.config.defaults import DEFAULT_CONFIG


class ConfigManager:
    """Manages application configuration with grouped settings."""

    CONFIG_FILENAME = "cadabrio_config.json"

    def __init__(self, config_dir: str | Path | None = None):
        if config_dir is None:
            self._config_dir = Path(user_config_dir("Cadabrio", "Cadabrio"))
        else:
            self._config_dir = Path(config_dir)

        self._config_path = self._config_dir / self.CONFIG_FILENAME
        self._data: dict[str, dict[str, Any]] = {}
        self._listeners: list = []

    @property
    def config_dir(self) -> Path:
        return self._config_dir

    def load(self):
        """Load configuration from disk, merging with defaults."""
        self._data = copy.deepcopy(DEFAULT_CONFIG)

        if self._config_path.exists():
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    user_data = json.load(f)

                # Merge user values over defaults (preserving defaults for missing keys)
                for group, values in user_data.items():
                    if group in self._data and isinstance(values, dict):
                        self._data[group].update(values)
                    else:
                        self._data[group] = values

                logger.info(f"Configuration loaded from {self._config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config, using defaults: {e}")
        else:
            logger.info("No config file found, using defaults.")

    def save(self):
        """Save current configuration to disk (excluding internal keys)."""
        self._config_dir.mkdir(parents=True, exist_ok=True)

        save_data = {}
        for group, values in self._data.items():
            save_data[group] = {k: v for k, v in values.items() if not k.startswith("_")}

        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)

        logger.info(f"Configuration saved to {self._config_path}")

    def get(self, group: str, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._data.get(group, {}).get(key, default)

    def set(self, group: str, key: str, value: Any):
        """Set a configuration value and notify listeners."""
        if group not in self._data:
            self._data[group] = {}
        old_value = self._data[group].get(key)
        self._data[group][key] = value
        if old_value != value:
            self._notify_listeners(group, key, value, old_value)

    def get_group(self, group: str) -> dict[str, Any]:
        """Get all values in a configuration group."""
        return {k: v for k, v in self._data.get(group, {}).items() if not k.startswith("_")}

    def get_group_label(self, group: str) -> str:
        """Get the display label for a configuration group."""
        return self._data.get(group, {}).get("_label", group.title())

    def groups(self) -> list[str]:
        """Get list of configuration group names."""
        return list(self._data.keys())

    def add_listener(self, callback):
        """Register a callback for config changes: callback(group, key, new_value, old_value)."""
        self._listeners.append(callback)

    def _notify_listeners(self, group: str, key: str, new_value: Any, old_value: Any):
        for listener in self._listeners:
            try:
                listener(group, key, new_value, old_value)
            except Exception as e:
                logger.error(f"Config listener error: {e}")
