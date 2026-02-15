"""AI Model Manager for Cadabrio.

Handles discovering, downloading, and managing AI model files.
Supports Hugging Face Hub, local directories, and manual imports.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from enum import Enum

from loguru import logger

from cadabrio.config.manager import ConfigManager


class ModelFormat(Enum):
    """Supported AI model file formats."""

    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    ONNX = "onnx"
    PYTORCH = "pytorch"  # .pt or .pth
    UNKNOWN = "unknown"


class ModelCategory(Enum):
    """Categories of AI models used by Cadabrio."""

    TEXT_TO_3D = "text_to_3d"
    IMAGE_TO_3D = "image_to_3d"
    TEXT_TO_IMAGE = "text_to_image"
    DEPTH_ESTIMATION = "depth_estimation"
    SEGMENTATION = "segmentation"
    CHAT = "chat"
    UPSCALE = "upscale"
    OTHER = "other"


@dataclass
class ModelInfo:
    """Metadata about an AI model."""

    model_id: str
    name: str
    category: ModelCategory
    format: ModelFormat
    path: Path | None = None
    size_bytes: int = 0
    source_url: str = ""
    description: str = ""
    downloaded: bool = False


class ModelManager:
    """Manages AI model discovery, download, and lifecycle."""

    def __init__(self, config: ConfigManager):
        self._config = config
        self._models: dict[str, ModelInfo] = {}
        self._models_dir = self._resolve_models_dir()

    def _resolve_models_dir(self) -> Path:
        """Get the models storage directory."""
        configured = self._config.get("ai", "models_directory", "")
        if configured:
            return Path(configured)
        return Path.home() / "cadabrio" / "models"

    def scan_local_models(self):
        """Scan the models directory for available model files."""
        if not self._models_dir.exists():
            self._models_dir.mkdir(parents=True, exist_ok=True)
            return

        extensions = {
            ".safetensors": ModelFormat.SAFETENSORS,
            ".gguf": ModelFormat.GGUF,
            ".onnx": ModelFormat.ONNX,
            ".pt": ModelFormat.PYTORCH,
            ".pth": ModelFormat.PYTORCH,
        }

        for ext, fmt in extensions.items():
            for path in self._models_dir.rglob(f"*{ext}"):
                model_id = path.stem
                if model_id not in self._models:
                    self._models[model_id] = ModelInfo(
                        model_id=model_id,
                        name=model_id.replace("-", " ").replace("_", " ").title(),
                        category=ModelCategory.OTHER,
                        format=fmt,
                        path=path,
                        size_bytes=path.stat().st_size,
                        downloaded=True,
                    )
        logger.info(f"Found {len(self._models)} local models in {self._models_dir}")

    def list_models(self, category: ModelCategory | None = None) -> list[ModelInfo]:
        """List available models, optionally filtered by category."""
        models = list(self._models.values())
        if category:
            models = [m for m in models if m.category == category]
        return models

    def download_model(self, repo_id: str, filename: str | None = None) -> ModelInfo | None:
        """Download a model from Hugging Face Hub."""
        if self._config.get("network", "offline_mode", False):
            logger.warning("Cannot download model in offline mode")
            return None

        # TODO: Implement HuggingFace Hub download
        logger.info(f"Model download not yet implemented: {repo_id}")
        return None

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Get info about a specific model."""
        return self._models.get(model_id)
