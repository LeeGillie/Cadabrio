"""Export format handlers for Cadabrio.

Supports exporting to: GLB, glTF, OBJ, STL, FBX, 3MF, USD.
Each format can be tuned for a specific target application.
"""

from enum import Enum
from pathlib import Path

from loguru import logger


class ExportFormat(Enum):
    """Supported export formats."""

    GLB = "glb"
    GLTF = "gltf"
    OBJ = "obj"
    STL = "stl"
    FBX = "fbx"
    THREE_MF = "3mf"
    USD = "usd"


# Recommended format per target
TARGET_FORMAT_MAP = {
    "general": ExportFormat.GLB,
    "print": ExportFormat.THREE_MF,
    "blender": ExportFormat.GLB,
    "freecad": ExportFormat.OBJ,  # or STEP via FreeCAD integration
    "unreal": ExportFormat.FBX,
    "bambu": ExportFormat.THREE_MF,
}


def export_mesh(mesh, output_path: Path, fmt: ExportFormat, **kwargs):
    """Export a trimesh mesh to the specified format."""
    import trimesh

    suffix_map = {
        ExportFormat.GLB: ".glb",
        ExportFormat.GLTF: ".gltf",
        ExportFormat.OBJ: ".obj",
        ExportFormat.STL: ".stl",
        ExportFormat.FBX: ".fbx",
        ExportFormat.THREE_MF: ".3mf",
        ExportFormat.USD: ".usd",
    }

    output_path = output_path.with_suffix(suffix_map.get(fmt, ".glb"))

    try:
        mesh.export(str(output_path), file_type=fmt.value)
        logger.info(f"Exported to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return None
