"""Project management for Cadabrio.

A Cadabrio project encapsulates all assets, settings, and pipeline state
for a 3D creation session. Projects can target specific outputs:
printable model, Blender, FreeCAD, Unreal Engine, or Bambu Studio.

Project files use the .cadabrio extension and are stored as JSON.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from enum import Enum

from loguru import logger

from cadabrio.version import __version__


PROJECT_EXTENSION = ".cadabrio"
PROJECT_FILE_VERSION = 1


class ProjectTarget(Enum):
    """Target output type that guides workflow suggestions."""

    GENERAL = "general"
    PRINT = "print"
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
    dirty: bool = False

    def add_asset(self, asset_meta: dict[str, Any]):
        """Add an imported asset to the project."""
        self.assets.append(asset_meta)
        self.dirty = True

    def save(self, path: Path | None = None):
        """Save the project to disk as JSON."""
        save_path = path or self.path
        if save_path is None:
            raise ValueError("No save path specified")

        save_path = Path(save_path)
        if save_path.suffix != PROJECT_EXTENSION:
            save_path = save_path.with_suffix(PROJECT_EXTENSION)

        data = {
            "_cadabrio_project": PROJECT_FILE_VERSION,
            "_app_version": __version__,
            "_saved_at": datetime.now(timezone.utc).isoformat(),
            "name": self.name,
            "target": self.target.value,
            "units": self.units,
            "scale_factor": self.scale_factor,
            "assets": self.assets,
            "metadata": self.metadata,
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        self.path = save_path
        self.dirty = False
        logger.info(f"Project saved: {save_path}")

    @classmethod
    def load(cls, path: Path) -> "Project":
        """Load a project from a .cadabrio JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_ver = data.get("_cadabrio_project", 0)
        if file_ver < 1:
            raise ValueError(f"Unknown project file version: {file_ver}")

        project = cls(
            name=data.get("name", path.stem),
            path=path,
            target=ProjectTarget(data.get("target", "general")),
            units=data.get("units", "millimeters"),
            scale_factor=data.get("scale_factor", 1.0),
            assets=data.get("assets", []),
            metadata=data.get("metadata", {}),
        )
        project.dirty = False
        logger.info(f"Project loaded: {path}")
        return project

    @classmethod
    def new(cls, name: str = "Untitled", target: str = "general") -> "Project":
        """Create a fresh project."""
        return cls(
            name=name,
            target=ProjectTarget(target),
            dirty=False,
        )
