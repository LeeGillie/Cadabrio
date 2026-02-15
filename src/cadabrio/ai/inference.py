"""Inference utilities for Cadabrio.

Provides common inference patterns and GPU memory management
optimized for RTX 5090 Blackwell architecture.
"""

from contextlib import contextmanager
from typing import Any, Generator

from loguru import logger


@contextmanager
def gpu_inference_context(device_idx: int = 0) -> Generator[Any, None, None]:
    """Context manager for GPU inference with proper memory management.

    Handles CUDA stream synchronization and memory cleanup,
    important for RTX 5090 with its large 32GB VRAM.
    """
    try:
        import torch

        device = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(device)

        yield device

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise
    finally:
        try:
            import torch
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass


def get_optimal_dtype():
    """Get the optimal tensor dtype for the current GPU.

    RTX 5090 Blackwell supports FP8, BF16, FP16, and FP32.
    BF16 is generally preferred for inference quality/speed balance.
    """
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if props.major >= 10:
                # Blackwell: prefer bfloat16
                return torch.bfloat16
            elif props.major >= 8:
                # Ampere/Ada: bfloat16 supported
                return torch.bfloat16
            else:
                return torch.float16
    except ImportError:
        pass

    return None
