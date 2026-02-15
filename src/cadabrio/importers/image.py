"""Image importer for Cadabrio.

Handles importing images (PNG, JPEG, TIFF, EXR, HDR) for use as
textures, reference images, or input to AI image-to-3D pipelines.
"""

from pathlib import Path
from typing import Any

from loguru import logger

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".exr", ".hdr"}


def can_import(path: Path) -> bool:
    """Check if the file is a supported image format."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def import_image(path: Path) -> dict[str, Any]:
    """Import an image file. Returns metadata dict."""
    import numpy as np
    from PIL import Image

    img = Image.open(path)
    return {
        "type": "image",
        "path": str(path),
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "format": img.format,
    }
