"""AI Model Manager for Cadabrio.

Handles discovering, downloading, and managing AI model files.
All models use the HuggingFace Hub cache — no separate directory needed.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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


# Maps ModelCategory to HuggingFace pipeline tags for searching
CATEGORY_TO_HF_TAG: dict[ModelCategory, list[str]] = {
    ModelCategory.TEXT_TO_IMAGE: ["text-to-image"],
    ModelCategory.TEXT_TO_3D: ["text-to-3d"],
    ModelCategory.IMAGE_TO_3D: ["image-to-3d"],
    ModelCategory.DEPTH_ESTIMATION: ["depth-estimation"],
    ModelCategory.SEGMENTATION: ["image-segmentation", "mask-generation"],
    ModelCategory.CHAT: ["text-generation", "conversational"],
    ModelCategory.UPSCALE: ["image-to-image"],
}

# Reverse: HF pipeline tag -> our category
HF_TAG_TO_CATEGORY: dict[str, ModelCategory] = {}
for _cat, _tags in CATEGORY_TO_HF_TAG.items():
    for _tag in _tags:
        HF_TAG_TO_CATEGORY[_tag] = _cat


# -------------------------------------------------------------------
# VRAM utilities
# -------------------------------------------------------------------

def get_gpu_vram_gb() -> float:
    """Return total GPU VRAM in GB, or 0 if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except ImportError:
        pass
    return 0.0


def get_gpu_vram_free_gb() -> float:
    """Return free GPU VRAM in GB, or 0 if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info(0)
            return free / (1024**3)
    except (ImportError, RuntimeError):
        pass
    return 0.0


def vram_fit_level(model_size_gb: float) -> str:
    """Return 'good', 'tight', 'bad', or 'unknown' based on VRAM fit.

    Compares estimated model VRAM to total GPU VRAM:
      good  : model < 70% of VRAM (plenty of room)
      tight : model 70-100% of VRAM (will work but leaves little room)
      bad   : model > 100% of VRAM (likely won't fit)
      unknown: can't determine (no GPU or no size info)
    """
    if model_size_gb <= 0:
        return "unknown"
    vram = get_gpu_vram_gb()
    if vram <= 0:
        return "unknown"
    ratio = model_size_gb / vram
    if ratio < 0.70:
        return "good"
    if ratio <= 1.0:
        return "tight"
    return "bad"


def vram_fit_text(model_size_gb: float) -> str:
    """Human-readable VRAM fit description."""
    level = vram_fit_level(model_size_gb)
    vram = get_gpu_vram_gb()
    if level == "unknown":
        return ""
    if level == "good":
        return f"Fits well ({model_size_gb:.1f} / {vram:.0f} GB VRAM)"
    if level == "tight":
        return f"Tight fit ({model_size_gb:.1f} / {vram:.0f} GB VRAM)"
    return f"Too large ({model_size_gb:.1f} / {vram:.0f} GB VRAM)"


# -------------------------------------------------------------------
# Formatting helpers
# -------------------------------------------------------------------

def _format_size(size_bytes: int) -> str:
    """Format byte size into human-readable string."""
    if size_bytes <= 0:
        return "—"
    gb = size_bytes / (1024**3)
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    mb = size_bytes / (1024**2)
    if mb >= 1.0:
        return f"{mb:.0f} MB"
    return f"{size_bytes / 1024:.0f} KB"


def _format_count(n: int) -> str:
    """Format a large count (downloads, likes) compactly."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# -------------------------------------------------------------------
# ModelInfo dataclass
# -------------------------------------------------------------------

@dataclass
class ModelInfo:
    """Metadata about an AI model."""

    model_id: str
    name: str
    category: ModelCategory
    format: ModelFormat
    path: Path | None = None
    size_bytes: int = 0       # Total disk size (may include duplicate formats)
    weight_bytes: int = 0     # Estimated model weight size (for VRAM estimation)
    source_url: str = ""
    description: str = ""
    downloaded: bool = False
    downloads: int = 0
    likes: int = 0
    pipeline_tag: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def size_display(self) -> str:
        return _format_size(self.size_bytes)

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024**3) if self.size_bytes > 0 else 0.0

    @property
    def weight_gb(self) -> float:
        """Model weight size in GB (used for VRAM estimation)."""
        wb = self.weight_bytes if self.weight_bytes > 0 else self.size_bytes
        return wb / (1024**3) if wb > 0 else 0.0

    @property
    def estimated_vram_gb(self) -> float:
        """Estimate VRAM needed. For diffusion models, VRAM ≈ weight size
        (minimal activation overhead unlike LLMs with KV caches)."""
        if self.weight_gb <= 0:
            return 0.0
        return self.weight_gb

    @property
    def vram_fit(self) -> str:
        """Return 'good', 'tight', 'bad', or 'unknown'."""
        return vram_fit_level(self.estimated_vram_gb)

    @property
    def downloads_display(self) -> str:
        return _format_count(self.downloads) if self.downloads else "—"

    @property
    def likes_display(self) -> str:
        return _format_count(self.likes) if self.likes else "—"


# -------------------------------------------------------------------
# HF cache location helper
# -------------------------------------------------------------------

def get_hf_cache_dir() -> Path:
    """Return the HuggingFace Hub cache directory."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except ImportError:
        return Path.home() / ".cache" / "huggingface" / "hub"


# -------------------------------------------------------------------
# ModelManager
# -------------------------------------------------------------------

class ModelManager:
    """Manages AI model discovery, download, and lifecycle."""

    def __init__(self, config: ConfigManager):
        self._config = config
        self._models: dict[str, ModelInfo] = {}

    def scan_local_models(self):
        """Scan the HF cache for downloaded models."""
        self._models.clear()
        self._scan_hf_cache()
        logger.info(f"Found {len(self._models)} models total")

    def _scan_hf_cache(self):
        """Scan the HuggingFace Hub cache for downloaded models."""
        try:
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if repo.repo_type != "model":
                    continue
                repo_id = repo.repo_id
                if repo_id in self._models:
                    continue
                category = self._categorize_repo(repo_id)
                # Use model weight size (safetensors) for VRAM estimation,
                # not full repo size which may include duplicate formats
                weight_size = self._estimate_weight_size(repo.repo_path)
                disk_size = repo.size_on_disk
                self._models[repo_id] = ModelInfo(
                    model_id=repo_id,
                    name=repo_id,
                    category=category,
                    format=ModelFormat.SAFETENSORS,
                    path=repo.repo_path,
                    size_bytes=disk_size,
                    weight_bytes=weight_size if weight_size > 0 else disk_size,
                    source_url=f"https://huggingface.co/{repo_id}",
                    downloaded=True,
                )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not scan HF cache: {e}")

    @staticmethod
    def _estimate_weight_size(repo_path: Path) -> int:
        """Estimate actual model weight size from the latest snapshot.

        When a repo has weights in multiple formats (e.g., FLUX has both a
        single-file export and split diffusers files), only count the
        diffusers-format files in subdirectories to avoid double-counting.
        """
        try:
            snapshots_dir = repo_path / "snapshots"
            if not snapshots_dir.exists():
                return 0
            snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if not snapshots:
                return 0
            latest = snapshots[0]

            # Separate top-level safetensors from those in subdirectories
            top_level = []
            in_subdirs = []
            for st_file in latest.rglob("*.safetensors"):
                if st_file.parent == latest:
                    top_level.append(st_file)
                else:
                    in_subdirs.append(st_file)

            # If we have both top-level and subdirectory weights, prefer subdirectories
            # (diffusers format) — the top-level file is likely a duplicate single-file export
            if in_subdirs and top_level:
                return sum(f.stat().st_size for f in in_subdirs)

            # Otherwise use all safetensors files
            all_files = top_level + in_subdirs
            return sum(f.stat().st_size for f in all_files)
        except Exception:
            return 0

    @staticmethod
    def _categorize_repo(repo_id: str) -> ModelCategory:
        """Guess a model category from its repo ID."""
        rid = repo_id.lower()
        if any(k in rid for k in ("stable-diffusion", "sdxl", "text-to-image", "flux", "dall")):
            return ModelCategory.TEXT_TO_IMAGE
        if any(k in rid for k in ("triposr", "text-to-3d", "shap-e")):
            return ModelCategory.TEXT_TO_3D
        if any(k in rid for k in ("image-to-3d", "zero123")):
            return ModelCategory.IMAGE_TO_3D
        if any(k in rid for k in ("dpt", "midas", "depth")):
            return ModelCategory.DEPTH_ESTIMATION
        if any(k in rid for k in ("sam", "segment")):
            return ModelCategory.SEGMENTATION
        if any(k in rid for k in ("whisper", "chat", "llama", "mistral", "gpt", "chatter", "qwen")):
            return ModelCategory.CHAT
        if any(k in rid for k in ("upscal", "esrgan", "real-esrgan")):
            return ModelCategory.UPSCALE
        return ModelCategory.OTHER

    def list_models(self, category: ModelCategory | None = None) -> list[ModelInfo]:
        """List available models, optionally filtered by category."""
        models = list(self._models.values())
        if category:
            models = [m for m in models if m.category == category]
        return sorted(models, key=lambda m: m.size_bytes, reverse=True)

    def search_hub(
        self,
        query: str = "",
        category: ModelCategory | None = None,
        limit: int = 25,
    ) -> list[ModelInfo]:
        """Search HuggingFace Hub for models, optionally filtered by purpose."""
        if self._config.get("network", "offline_mode", False):
            return []

        try:
            from huggingface_hub import HfApi
            api = HfApi()

            pipeline_tag = None
            if category and category in CATEGORY_TO_HF_TAG:
                pipeline_tag = CATEGORY_TO_HF_TAG[category][0]

            kwargs: dict[str, Any] = {
                "limit": limit,
                "sort": "downloads",
            }
            if query:
                kwargs["search"] = query
            if pipeline_tag:
                kwargs["pipeline_tag"] = pipeline_tag

            results = list(api.list_models(**kwargs))
            models = []
            for r in results:
                ptag = getattr(r, "pipeline_tag", "") or ""
                cat = HF_TAG_TO_CATEGORY.get(ptag, self._categorize_repo(r.modelId))
                is_downloaded = r.modelId in self._models

                # Use local size if downloaded, otherwise 0 (fetched async later)
                size_bytes = 0
                if is_downloaded and r.modelId in self._models:
                    size_bytes = self._models[r.modelId].size_bytes

                models.append(ModelInfo(
                    model_id=r.modelId,
                    name=r.modelId,
                    category=cat,
                    format=ModelFormat.SAFETENSORS,
                    size_bytes=size_bytes,
                    source_url=f"https://huggingface.co/{r.modelId}",
                    downloaded=is_downloaded,
                    downloads=getattr(r, "downloads", 0) or 0,
                    likes=getattr(r, "likes", 0) or 0,
                    pipeline_tag=ptag,
                    tags=(list(r.tags[:5]) if r.tags else []),
                ))
            return models
        except ImportError:
            logger.warning("huggingface-hub not installed")
            return []
        except Exception as e:
            logger.error(f"HF Hub search failed: {e}")
            return []

    @staticmethod
    def fetch_model_size(model_id: str) -> int:
        """Fetch the total file size for a model from HF Hub (network call).

        Returns size in bytes, or 0 if unavailable.
        """
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.model_info(model_id, files_metadata=True)
            if info.siblings:
                return sum(getattr(s, "size", 0) or 0 for s in info.siblings)
        except Exception as e:
            logger.debug(f"Could not fetch size for {model_id}: {e}")
        return 0

    @staticmethod
    def fetch_model_sizes_batch(model_ids: list[str]) -> dict[str, int]:
        """Fetch sizes for multiple models in parallel. Returns {model_id: bytes}."""
        sizes: dict[str, int] = {}

        def _fetch_one(mid: str) -> tuple[str, int]:
            return mid, ModelManager.fetch_model_size(mid)

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(_fetch_one, mid): mid for mid in model_ids}
            for future in as_completed(futures):
                try:
                    mid, size = future.result()
                    sizes[mid] = size
                except Exception:
                    pass
        return sizes

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Get info about a specific model."""
        return self._models.get(model_id)
