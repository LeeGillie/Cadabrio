"""Scale management for Cadabrio.

Maintains consistent real-world scale throughout all transformational
steps in the pipeline, from ingestion through export.
"""

from enum import Enum


class Unit(Enum):
    """Supported measurement units."""

    MILLIMETERS = "millimeters"
    CENTIMETERS = "centimeters"
    METERS = "meters"
    INCHES = "inches"
    FEET = "feet"


# Conversion factors to meters (base unit)
_TO_METERS = {
    Unit.MILLIMETERS: 0.001,
    Unit.CENTIMETERS: 0.01,
    Unit.METERS: 1.0,
    Unit.INCHES: 0.0254,
    Unit.FEET: 0.3048,
}


class ScaleManager:
    """Tracks and converts scale across pipeline stages."""

    def __init__(self, default_unit: Unit = Unit.MILLIMETERS):
        self._unit = default_unit
        self._reference_scale: float | None = None

    @property
    def unit(self) -> Unit:
        return self._unit

    @unit.setter
    def unit(self, value: Unit):
        self._unit = value

    def convert(self, value: float, from_unit: Unit, to_unit: Unit) -> float:
        """Convert a measurement between units."""
        meters = value * _TO_METERS[from_unit]
        return meters / _TO_METERS[to_unit]

    def set_reference_scale(self, known_dimension: float, unit: Unit):
        """Set a reference scale from a known real-world measurement.

        Used to calibrate photogrammetry outputs against a known dimension.
        """
        self._reference_scale = known_dimension * _TO_METERS[unit]

    def apply_reference_scale(self, raw_value: float, to_unit: Unit) -> float:
        """Apply reference scale correction to a raw measurement."""
        if self._reference_scale is None:
            return raw_value
        return raw_value * self._reference_scale / _TO_METERS[to_unit]
