"""Tests for version info."""

from cadabrio.version import __version__, __version_display__


def test_version_format():
    """Version string follows semver format."""
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_version_display():
    """Display version includes program name."""
    assert __version_display__.startswith("Cadabrio V")
    assert __version__ in __version_display__
