"""A ComfyUI node used to decode latent image data in tiled blocks."""
from typing import ClassVar

from pydantic import Field

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output
from sd_backend_client.api.comfyui.nodes.vae.vae_encode_tiled_node import (OVERLAP_DEFAULT, TEMPORAL_OVERLAP_DEFAULT,
                                                                           TEMPORAL_SIZE_DEFAULT, TILE_MAX, TILE_MIN,
                                                                           TILE_STEP)


class VAEDecodeTiledNode(ComfyNode):
    """A ComfyUI node used to decode latent image data in tiled blocks."""
    CLASS_TYPE: ClassVar[str] = 'VAEDecodeTiled'

    tile_size: int = Field(ge=TILE_MIN, le=TILE_MAX, multiple_of=TILE_STEP)
    overlap: int = OVERLAP_DEFAULT
    temporal_size: int = TEMPORAL_SIZE_DEFAULT
    """Video-VAE-only: frames decoded at a time. Ignored by image VAEs."""
    temporal_overlap: int = TEMPORAL_OVERLAP_DEFAULT
    """Video-VAE-only: frames of overlap between temporal tiles. Ignored by image VAEs."""
    samples: Connection = None  # Latent image data, e.g. from KSampler.
    vae: Connection = None  # VAE model used for decoding. May be baked-in to a regular SD model.

    image_out = Output(0)
