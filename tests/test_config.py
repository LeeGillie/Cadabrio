"""Tests for the configuration system."""

from cadabrio.config.manager import ConfigManager
from cadabrio.config.defaults import DEFAULT_CONFIG


class TestConfigManager:
    def test_load_defaults(self, config_manager):
        """Config loads with default values."""
        assert config_manager.get("appearance", "theme") == "cadabrio_dark"
        assert config_manager.get("general", "auto_save") is True

    def test_set_and_get(self, config_manager):
        """Can set and retrieve values."""
        config_manager.set("appearance", "theme", "solarized")
        assert config_manager.get("appearance", "theme") == "solarized"

    def test_save_and_reload(self, tmp_config_dir):
        """Config persists across save/load cycles."""
        mgr = ConfigManager(config_dir=tmp_config_dir)
        mgr.load()
        mgr.set("general", "language", "fr")
        mgr.save()

        mgr2 = ConfigManager(config_dir=tmp_config_dir)
        mgr2.load()
        assert mgr2.get("general", "language") == "fr"

    def test_groups(self, config_manager):
        """All expected groups are present."""
        groups = config_manager.groups()
        assert "general" in groups
        assert "appearance" in groups
        assert "gpu" in groups
        assert "ai" in groups
        assert "integrations" in groups

    def test_group_labels(self, config_manager):
        """Groups have display labels."""
        assert config_manager.get_group_label("gpu") == "GPU & Compute"
        assert config_manager.get_group_label("ai") == "AI Models"

    def test_listener_called(self, config_manager):
        """Config change listeners are notified."""
        changes = []
        config_manager.add_listener(
            lambda group, key, new, old: changes.append((group, key, new, old))
        )
        config_manager.set("appearance", "font_size", 14)
        assert len(changes) == 1
        assert changes[0] == ("appearance", "font_size", 14, 10)

    def test_default_config_has_labels(self):
        """Every default config group has a _label."""
        for group, values in DEFAULT_CONFIG.items():
            assert "_label" in values, f"Group '{group}' missing _label"
