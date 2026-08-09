"""Shared definitions for generic API data types."""
from typing import Optional

from pydantic import BaseModel

from intrapaint_api.api.shared_data.controlnet.controlnet_unit import ControlNetUnit
from intrapaint_api.util.api_defaults import UPSCALING_DENOISING_STRENGTH_DEFAULT, UPSCALING_STEP_COUNT_DEFAULT, \
    GENERATION_SIZE_DEFAULT


class DiffusionUpscalingParams(BaseModel):
    """Dataclass defining parameters for Stable Diffusion upscaling"""
    upscaling_mode: str = ""
    """Optional basic upscaler to use for basic upscaling or preprocessing before enhancing details via diffusion."""

    use_stable_diffusion_upscaling: bool = False
    """Whether to use tiled Stable Diffusion upscaling to refine details."""

    denoising_strength: float = UPSCALING_DENOISING_STRENGTH_DEFAULT
    """
    Denoising strength (range: 0.0, 1.0) to use when enhancing details via diffusion when
    use_stable_diffusion_upscaling=True
    """

    step_count: int = UPSCALING_STEP_COUNT_DEFAULT
    """Number of detail-enhancing diffusion steps to apply per image tile when use_stable_diffusion_upscaling=True"""

    tile_width: int = GENERATION_SIZE_DEFAULT
    """Width (pixels) for each tiled diffusion step when use_stable_diffusion_upscaling=True"""

    tile_height: int = GENERATION_SIZE_DEFAULT
    """Height (pixels) for each tiled diffusion step when use_stable_diffusion_upscaling=True"""

    tile_controlnet: Optional[ControlNetUnit] = None
    """Optional ControlNet to apply during tiled diffusion when use_stable_diffusion_upscaling=True"""
