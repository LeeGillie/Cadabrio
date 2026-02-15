"""Shared test fixtures for Cadabrio."""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Provide a temporary configuration directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def config_manager(tmp_config_dir):
    """Provide a ConfigManager with a temp directory."""
    from cadabrio.config.manager import ConfigManager

    mgr = ConfigManager(config_dir=tmp_config_dir)
    mgr.load()
    return mgr


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent
