"""3D model importer for Cadabrio.

Handles importing 3D model files (OBJ, STL, FBX, glTF/GLB, 3MF, STEP, PLY, USD)
for use as starting points, reference geometry, or parts to incorporate.
"""

from pathlib import Path
from typing import Any

from loguru import logger

SUPPORTED_EXTENSIONS = {
    ".obj", ".stl", ".fbx", ".glb", ".gltf", ".3mf",
    ".ply", ".usd", ".usda", ".usdz",
    ".step", ".stp", ".iges", ".igs",
    ".blend",
}


def can_import(path: Path) -> bool:
    """Check if the file is a supported 3D model format."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def import_model(path: Path) -> dict[str, Any]:
    """Import a 3D model file. Returns metadata dict."""
    import trimesh

    try:
        mesh = trimesh.load(str(path))
        if isinstance(mesh, trimesh.Scene):
            # Multi-object file
            return {
                "type": "model_3d",
                "path": str(path),
                "format": path.suffix.lower(),
                "objects": len(mesh.geometry),
                "bounds": mesh.bounds.tolist() if hasattr(mesh, "bounds") else None,
            }
        else:
            return {
                "type": "model_3d",
                "path": str(path),
                "format": path.suffix.lower(),
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "bounds": mesh.bounds.tolist(),
                "is_watertight": mesh.is_watertight,
            }
    except Exception as e:
        logger.error(f"Failed to import model {path}: {e}")
        return {"type": "model_3d", "path": str(path), "error": str(e)}
