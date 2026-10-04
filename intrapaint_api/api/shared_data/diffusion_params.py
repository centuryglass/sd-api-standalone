"""Shared image diffusion operation parameter set."""
import logging
from typing import Any, Optional

from PIL import Image
from pydantic import BaseModel, ConfigDict

from intrapaint_api.api.shared_data.controlnet.controlnet_unit import ControlNetUnit

logger = logging.getLogger(__name__)


class DiffusionParams(BaseModel):
    """Request body format for image generation (all types)"""
    # use_enum_values: store/serialize enum fields (e.g. WebUI ResizeMode, InpaintFillOption) as their underlying
    # values, so model_dump() produces the JSON-serializable ints the APIs expect. validate_default ensures enum
    # *defaults* (e.g. inpainting_fill=InpaintFillOption.ORIGINAL) are converted too, not just explicitly-set values.
    model_config = ConfigDict(arbitrary_types_allowed=True, use_enum_values=True, validate_default=True)

    ### Basic image generation:
    sd_model_name: str = ''
    """Stable Diffusion model name."""

    sampler_name: str = 'Euler a'
    """The algorithm used to iteratively denoise the image during generation. Valid options vary by API."""

    batch_size: int = 1
    """Number of images to generate per batch."""

    steps: int = 30
    """Number of denoising steps per image generation."""

    cfg_scale: float = 7.0  # guidance scale
    """
    How strongly generation follows the text prompt. Typical useful range is model-specific, usually 6-10 for typical
    models, 1-2 for LCM and Turbo models.
    """

    width: int = 512
    """Generated image width in pixels."""

    height: int = 512
    """Generated image height in pixels."""


    ### Prompt:
    prompt: str = ''
    """Text description guiding what the generated image should contain."""

    negative_prompt: str = ''
    """Text description guiding what the generated image should avoid containing."""

    seed: int = -1
    """Random seed controlling generation. Any value less than zero selects a new random seed."""


    ### Img2img and inpainting only:
    init_images: Optional[list[Image.Image]] = None
    """
    List of Pil images for inpainting or image to image.
    """

    denoising_strength: Optional[float] = None
    """
    Amount that the source image should change in image to image and inpainting generations, ranging from 0.0 (no
    change) to 1.0 (complete replacement).
    """


    ### Inpainting only:
    mask: Optional[Image.Image] = None
    """Inpainting mask image on both backends: opaque or white pixels mark the region to change.

    Either an alpha mask or an opaque grayscale mask is accepted (see `mask_to_grayscale`). Each backend converts to
    its own wire format, so callers never invert the mask for ComfyUI.
    """

    controlnet_units: list[ControlNetUnit] = []

    def to_dict(self) -> dict[str, Any]:
        """Convert the request body to a dict, removing unused optional parameters."""
        return self.model_dump(exclude_none=True)
