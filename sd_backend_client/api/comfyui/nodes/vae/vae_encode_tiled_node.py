"""A ComfyUI node used to encode latent image data in tiled blocks."""
from typing import ClassVar

from pydantic import Field

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output

TILE_MIN = 320
TILE_MAX = 4096
TILE_STEP = 64

OVERLAP_DEFAULT = 64
TEMPORAL_SIZE_DEFAULT = 64
TEMPORAL_OVERLAP_DEFAULT = 8


class VAEEncodeTiledNode(ComfyNode):
    """A ComfyUI node used to encode latent image data in tiled blocks."""
    CLASS_TYPE: ClassVar[str] = 'VAEEncodeTiled'

    tile_size: int = Field(ge=TILE_MIN, le=TILE_MAX, multiple_of=TILE_STEP)
    overlap: int = OVERLAP_DEFAULT
    temporal_size: int = TEMPORAL_SIZE_DEFAULT
    """Video-VAE-only: frames encoded at a time. Ignored by image VAEs."""
    temporal_overlap: int = TEMPORAL_OVERLAP_DEFAULT
    """Video-VAE-only: frames of overlap between temporal tiles. Ignored by image VAEs."""
    pixels: Connection = None  # raw image data, e.g. from LoadImage.
    vae: Connection = None  # VAE model used for encoding. May be baked-in to a regular SD model.

    latent_out = Output(0)
