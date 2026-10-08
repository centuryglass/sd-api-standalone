"""A ComfyUI node used to decode latent image data in tiled blocks."""
from typing import ClassVar

from pydantic import Field

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output
from sd_backend_client.api.comfyui.nodes.vae.vae_encode_tiled_node import TILE_MAX, TILE_MIN, TILE_STEP

OVERLAP_DEFAULT = 64


class VAEDecodeTiledNode(ComfyNode):
    """A ComfyUI node used to decode latent image data in tiled blocks."""
    CLASS_TYPE: ClassVar[str] = 'VAEDecodeTiled'

    tile_size: int = Field(ge=TILE_MIN, le=TILE_MAX, multiple_of=TILE_STEP)
    overlap: int = OVERLAP_DEFAULT
    samples: Connection = None  # Latent image data, e.g. from KSampler.
    vae: Connection = None  # VAE model used for decoding. May be baked-in to a regular SD model.

    image_out = Output(0)
