"""Project management for Cadabrio.

A Cadabrio project encapsulates all assets, settings, and pipeline state
for a 3D creation session. Projects can target specific outputs:
printable model, Blender, FreeCAD, Unreal Engine, or Bambu Studio.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from enum import Enum


class ProjectTarget(Enum):
    """Target output type that guides workflow suggestions."""

    GENERAL = "general"
    PRINT = "print"  # 3D printing via Bambu Studio or slicer
    BLENDER = "blender"
    FREECAD = "freecad"
    UNREAL = "unreal"
    BAMBU = "bambu"


@dataclass
class Project:
    """Represents a Cadabrio project."""

    name: str = "Untitled"
    path: Path | None = None
    target: ProjectTarget = ProjectTarget.GENERAL
    units: str = "millimeters"
    scale_factor: float = 1.0
    assets: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self):
        """Save the project to disk."""
        # TODO: Implement project serialization
        pass

    @classmethod
    def load(cls, path: Path) -> "Project":
        """Load a project from disk."""
        # TODO: Implement project deserialization
        return cls(path=path)
