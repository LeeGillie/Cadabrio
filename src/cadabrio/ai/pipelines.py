"""AI pipeline runners for Cadabrio.

Each pipeline lazily loads its model on first use, runs inference on the GPU,
and caches the loaded model for subsequent calls. All pipelines use bf16
on Blackwell (RTX 5090) for optimal speed/quality.

Models are configurable — pass a model_id to override the default.

Default text-to-image: FLUX.1-schnell (black-forest-labs/FLUX.1-schnell)
  - 12B params, Apache 2.0 licensed
  - T5 text encoder — no 77-token limit, excellent prompt following
  - Only 4 inference steps needed (distilled model)
  - ~24GB VRAM in bf16 (fits on RTX 5090 32GB)
"""

import gc
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from loguru import logger
from PIL import Image

from cadabrio.ai.inference import get_optimal_dtype


class GenerationCancelled(Exception):
    """Raised when the user cancels an ongoing generation."""
    pass


# Type for progress callbacks: (step_index, total_steps) -> None
ProgressCallback = Callable[[int, int], None]

# Default model IDs — users can override these in the AI Tools dialog
DEFAULT_TXT2IMG = "black-forest-labs/FLUX.1-schnell"
DEFAULT_DEPTH = "Intel/dpt-large"
DEFAULT_TRIPOSR = "stabilityai/TripoSR"
DEFAULT_SAM = "facebook/sam-vit-huge"

# Models known to be FLUX-architecture (different pipeline, different defaults)
_FLUX_MODELS = {"black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev"}


def _is_flux(model_id: str) -> bool:
    """Check if a model ID is a FLUX model."""
    return model_id in _FLUX_MODELS or "flux" in model_id.lower()


def _is_sdxl(model_id: str) -> bool:
    """Check if a model is SDXL architecture."""
    mid = model_id.lower()
    return "sdxl" in mid or "stable-diffusion-xl" in mid


def _is_sd1x(model_id: str) -> bool:
    """Check if a model is SD 1.x architecture (512x512 native)."""
    mid = model_id.lower()
    return any(p in mid for p in (
        "stable-diffusion-v1", "sd-v1", "compvis/stable-diffusion",
    ))


def model_native_res(model_id: str) -> int:
    """Return the native resolution for a model architecture.

    FLUX / SDXL: 1024, SD 2.x: 768, SD 1.x: 512.
    """
    if _is_flux(model_id) or _is_sdxl(model_id):
        return 1024
    if _is_sd1x(model_id):
        return 512
    return 768  # SD 2.x and unknown


def model_default_params(model_id: str) -> dict[str, Any]:
    """Return sensible default generation params for a model architecture.

    Keys: steps, guidance_scale, width, height.
    """
    if _is_flux(model_id):
        return {"steps": 4, "guidance_scale": 0.0, "width": 1024, "height": 1024}
    if _is_sdxl(model_id):
        return {"steps": 25, "guidance_scale": 7.5, "width": 1024, "height": 1024}
    if _is_sd1x(model_id):
        return {"steps": 25, "guidance_scale": 7.5, "width": 512, "height": 512}
    # SD 2.x / unknown
    return {"steps": 25, "guidance_scale": 7.5, "width": 768, "height": 768}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype() -> torch.dtype:
    return get_optimal_dtype() or torch.float32


# ---------------------------------------------------------------------------
# Model cache — keeps one instance of each pipeline in VRAM
# Keyed by (pipeline_type, model_id) so switching models works correctly
# ---------------------------------------------------------------------------
_CACHE: dict[str, tuple[str, Any]] = {}  # name -> (model_id, pipeline)


def _get_cached(name: str, model_id: str):
    """Return cached pipeline if it matches model_id, else None."""
    if name in _CACHE:
        cached_id, pipe = _CACHE[name]
        if cached_id == model_id:
            return pipe
        # Model changed — unload the old one
        unload_pipeline(name)
    return None


def unload_pipeline(name: str):
    """Unload a cached pipeline to free VRAM."""
    entry = _CACHE.pop(name, None)
    del entry
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Unloaded pipeline: {name}")


def unload_all():
    """Unload every cached pipeline."""
    for name in list(_CACHE):
        unload_pipeline(name)


def loaded_pipelines() -> list[dict[str, str]]:
    """Return info about currently loaded pipelines (for UI display)."""
    result = []
    for name, (model_id, _) in _CACHE.items():
        result.append({"pipeline": name, "model_id": model_id})
    return result


# ===================================================================
# 1. Text-to-Image
# ===================================================================

def _load_txt2img(model_id: str):
    cached = _get_cached("txt2img", model_id)
    if cached:
        return cached

    logger.info(f"Loading text-to-image model: {model_id}")

    if _is_flux(model_id):
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=_dtype(),
        )
    else:
        from diffusers import DiffusionPipeline
        try:
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=_dtype(),
                variant="fp16",
                use_safetensors=True,
            )
        except (OSError, ValueError):
            # Model may not have fp16 variant (e.g. SD 1.x) — load without it
            logger.info(f"No fp16 variant for {model_id}, loading default weights")
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=_dtype(),
            )

    pipe.to(_device())
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    _CACHE["txt2img"] = (model_id, pipe)
    logger.info(f"Text-to-image pipeline ready: {model_id}")
    return pipe


def text_to_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    guidance_scale: float = 0.0,
    seed: int | None = None,
    model_id: str = DEFAULT_TXT2IMG,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> Image.Image:
    """Generate an image from a text prompt."""
    pipe = _load_txt2img(model_id)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device()).manual_seed(seed)

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }

    # FLUX doesn't support negative prompts; SDXL/SD do
    if not _is_flux(model_id) and negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    # Step progress + cancellation callback
    if progress_callback or cancel_event:
        kwargs["callback_on_step_end"] = _make_step_callback(
            steps, progress_callback, cancel_event,
        )

    result = pipe(**kwargs)
    return result.images[0]


def _load_img2img(model_id: str):
    cached = _get_cached("img2img", model_id)
    if cached:
        return cached

    logger.info(f"Loading img2img pipeline: {model_id}")

    if _is_flux(model_id):
        from diffusers import FluxImg2ImgPipeline
        pipe = FluxImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=_dtype(),
        )
    elif _is_sdxl(model_id):
        from diffusers import StableDiffusionXLImg2ImgPipeline
        try:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=_dtype(),
                variant="fp16",
                use_safetensors=True,
            )
        except (OSError, ValueError):
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=_dtype(),
            )
    else:
        # SD 1.x, SD 2.x, and other non-XL models
        from diffusers import StableDiffusionImg2ImgPipeline
        try:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=_dtype(),
                variant="fp16",
                use_safetensors=True,
            )
        except (OSError, ValueError):
            logger.info(f"No fp16 variant for {model_id}, loading default weights")
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=_dtype(),
            )

    pipe.to(_device())
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    _CACHE["img2img"] = (model_id, pipe)
    logger.info(f"Img2img pipeline ready: {model_id}")
    return pipe


def image_guided_generate(
    prompt: str,
    reference_image: Image.Image,
    negative_prompt: str = "",
    strength: float = 0.5,
    steps: int = 4,
    guidance_scale: float = 0.0,
    seed: int | None = None,
    model_id: str = DEFAULT_TXT2IMG,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> Image.Image:
    """Generate an image using a reference photo as a starting point.

    The 'strength' parameter controls how much to deviate from the reference:
    - 0.2-0.3: Stay very close to the reference (color/composition tweaks)
    - 0.4-0.6: Moderate changes (recommended for style transfer)
    - 0.7-0.9: Major changes, reference is just a rough guide
    """
    pipe = _load_img2img(model_id)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device()).manual_seed(seed)

    res = model_native_res(model_id)
    ref = reference_image.convert("RGB").resize((res, res), Image.Resampling.LANCZOS)

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "image": ref,
        "strength": strength,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }

    if not _is_flux(model_id) and negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    if progress_callback or cancel_event:
        kwargs["callback_on_step_end"] = _make_step_callback(
            steps, progress_callback, cancel_event,
        )

    result = pipe(**kwargs)
    return result.images[0]


# -------------------------------------------------------------------
# Step callback for progress + cancellation
# -------------------------------------------------------------------

def multi_reference_generate(
    prompt: str,
    references: list[Image.Image],
    negative_prompt: str = "",
    strength: float = 0.5,
    steps: int = 4,
    guidance_scale: float = 0.0,
    seed: int | None = None,
    model_id: str = DEFAULT_TXT2IMG,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> Image.Image:
    """Generate an image guided by multiple reference images.

    Phase 1: Uses the first reference image with image_guided_generate().
    Phase 2 (future): Will use IP-Adapter for true multi-image conditioning.
    """
    if not references:
        return text_to_image(
            prompt, negative_prompt, 1024, 1024, steps, guidance_scale,
            seed, model_id, progress_callback, cancel_event,
        )
    return image_guided_generate(
        prompt, references[0], negative_prompt, strength, steps,
        guidance_scale, seed, model_id, progress_callback, cancel_event,
    )


def _make_step_callback(total_steps, progress_cb, cancel_event):
    """Create a diffusers callback_on_step_end function."""
    def callback(pipe, step_index, timestep, callback_kwargs):
        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled("Cancelled by user")
        if progress_cb:
            progress_cb(step_index + 1, total_steps)
        return callback_kwargs
    return callback


# ===================================================================
# 2. Depth Estimation
# ===================================================================

def _load_depth(model_id: str):
    cached = _get_cached("depth", model_id)
    if cached:
        return cached

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    logger.info(f"Loading depth estimation model: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(
        model_id, torch_dtype=_dtype()
    )
    model.to(_device())
    model.eval()
    pair = (processor, model)
    _CACHE["depth"] = (model_id, pair)
    logger.info(f"Depth estimation model ready: {model_id}")
    return pair


def estimate_depth(
    image: Image.Image,
    model_id: str = DEFAULT_DEPTH,
) -> Image.Image:
    """Estimate a depth map from an RGB image. Returns a grayscale depth image."""
    processor, model = _load_depth(model_id)

    inputs = processor(images=image, return_tensors="pt").to(_device())
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_np = prediction.cpu().numpy()
    depth_min, depth_max = depth_np.min(), depth_np.max()
    depth_norm = ((depth_np - depth_min) / (depth_max - depth_min + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(depth_norm)


# ===================================================================
# 3. Image-to-3D  (TripoSR)
# ===================================================================

def _load_triposr(model_id: str):
    cached = _get_cached("img2mesh", model_id)
    if cached:
        return cached

    logger.info(f"Loading image-to-3D model: {model_id}")
    try:
        from tsr.system import TSR
        model = TSR.from_pretrained(
            model_id,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.to(_device())
        _CACHE["img2mesh"] = (model_id, model)
        logger.info(f"Image-to-3D model ready: {model_id}")
        return model
    except ImportError:
        logger.warning(
            "tsr package not installed. Install with: pip install tsr"
        )
        return None


def image_to_3d(
    image: Image.Image,
    output_path: Path,
    model_id: str = DEFAULT_TRIPOSR,
) -> Path | None:
    """Generate a 3D mesh from a single image. Returns path to GLB or None."""
    model = _load_triposr(model_id)
    if model is None:
        return None

    try:
        image = image.convert("RGB").resize((512, 512))
        with torch.no_grad():
            scene_codes = model([image], device=_device())

        output_path = output_path.with_suffix(".glb")
        meshes = model.extract_mesh(scene_codes)
        meshes[0].export(str(output_path))
        logger.info(f"3D mesh exported to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Image-to-3D failed: {e}")
        return None


# ===================================================================
# 4. Image Segmentation  (SAM)
# ===================================================================

def _load_sam(model_id: str):
    cached = _get_cached("sam", model_id)
    if cached:
        return cached

    from transformers import SamModel, SamProcessor

    logger.info(f"Loading segmentation model: {model_id}")
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id, torch_dtype=_dtype())
    model.to(_device())
    model.eval()
    pair = (processor, model)
    _CACHE["sam"] = (model_id, pair)
    logger.info(f"Segmentation model ready: {model_id}")
    return pair


def segment_image(
    image: Image.Image,
    points: list[list[int]] | None = None,
    model_id: str = DEFAULT_SAM,
) -> list[np.ndarray]:
    """Segment an image using SAM. Returns list of binary mask arrays."""
    processor, model = _load_sam(model_id)

    if points is not None:
        inputs = processor(
            image, input_points=[points], return_tensors="pt",
        ).to(_device())
    else:
        w, h = image.size
        grid_points = [
            [int(x * w), int(y * h)]
            for y in [0.25, 0.5, 0.75]
            for x in [0.25, 0.5, 0.75]
        ]
        inputs = processor(
            image, input_points=[grid_points], return_tensors="pt",
        ).to(_device())

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    mask_list = []
    if masks:
        batch_masks = masks[0]
        for i in range(batch_masks.shape[0]):
            for j in range(batch_masks.shape[1]):
                mask_list.append(batch_masks[i, j].numpy().astype(bool))

    return mask_list


# ===================================================================
# 5. Chat / LLM
# ===================================================================

DEFAULT_CHAT = ""  # No default — user picks from downloaded models


def _load_chat(model_id: str):
    cached = _get_cached("chat", model_id)
    if cached:
        return cached

    from transformers import AutoTokenizer, AutoModelForCausalLM

    logger.info(f"Loading chat model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=_dtype(),
    )
    model.to(_device())
    model.eval()
    pair = (tokenizer, model)
    _CACHE["chat"] = (model_id, pair)
    logger.info(f"Chat model ready: {model_id}")
    return pair


def chat_generate(
    messages: list[dict[str, str]],
    model_id: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate a chat response from a list of message dicts.

    Args:
        messages: List of {"role": "user"|"assistant"|"system", "content": "..."}.
        model_id: HuggingFace model ID for the chat model.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).

    Returns:
        The assistant's reply text.
    """
    tokenizer, model = _load_chat(model_id)

    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # apply_chat_template may return a BatchEncoding dict or a raw tensor
    if isinstance(tokenized, dict) or hasattr(tokenized, "input_ids"):
        input_ids = tokenized["input_ids"].to(_device())
    else:
        input_ids = tokenized.to(_device())

    prompt_len = input_ids.shape[1]

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)

    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def segment_to_image(image: Image.Image, masks: list[np.ndarray]) -> Image.Image:
    """Overlay segmentation masks on an image with colored regions."""
    img_array = np.array(image.convert("RGB")).copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
        (0, 128, 255),
    ]

    for idx, mask in enumerate(masks):
        color = colors[idx % len(colors)]
        for c in range(3):
            img_array[:, :, c] = np.where(
                mask,
                img_array[:, :, c] * 0.5 + color[c] * 0.5,
                img_array[:, :, c],
            )

    return Image.fromarray(img_array.astype(np.uint8))
