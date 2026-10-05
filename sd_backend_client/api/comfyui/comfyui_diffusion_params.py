"""ComfyUI extended diffusion parameters"""
import logging
from typing import Optional

from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams

logger = logging.getLogger(__name__)


class ComfyUIDiffusionParams(DiffusionParams):
    """ComfyUI extended diffusion parameters"""

    sampler: str = 'euler'
    scheduler: str = 'karras'
    load_as_inpainting_model: bool = False
    vae_tiling_enabled: bool = False
    vae_tile_size: int = 512
    clip_skip: int = 1
    sd_model_config: Optional[str] = None

