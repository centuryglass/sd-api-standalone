"""ComfyUI extended diffusion parameters"""
import logging
import os.path
from enum import Enum
from typing import Any, Optional

from PIL import Image

from intrapaint_api.api.shared_data.diffusion_params import DiffusionParams
from intrapaint_api.api.webui.controlnet_webui_constants import CONTROLNET_SCRIPT_KEY, ControlNetUnitDict
from intrapaint_api.api.webui.script_info_types import ScriptRequestData
from intrapaint_api.util.visual.image_utils import image_to_base64

logger = logging.getLogger(__name__)

class ResizeMode(Enum):
    """Controls how source images are resized when source resolution doesn't match generated image size."""
    JUST_RESIZE = 0
    CROP_AND_RESIZE = 1
    RESIZE_AND_FILL = 2
    RESIZE_LATENT_UPSCALE = 3

class InpaintFillOption(Enum):
    """Controls how source image data is processed before inpainting. (ORIGINAL is almost always the right choice.)"""
    FILL = 0
    ORIGINAL = 1
    LATENT_NOISE = 2
    LATENT_NOTHING = 3


class ComfyUIDiffusionParams(DiffusionParams):
    """ComfyUI extended diffusion parameters"""

    sampler: str = 'euler'
    scheduler: str = 'karras'
    load_as_inpainting_model: bool = False
    vae_tiling_enabled: bool = False
    vae_tile_size: int = 512
    clip_skip: int = 1
    sd_model_config: Optional[str] = None

