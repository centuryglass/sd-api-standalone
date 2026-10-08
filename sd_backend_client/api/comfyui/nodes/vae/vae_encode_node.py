"""A ComfyUI node used to encode latent image data."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output


class VAEEncodeNode(ComfyNode):
    """A ComfyUI node used to encode images into latent image space."""
    CLASS_TYPE: ClassVar[str] = 'VAEEncode'

    pixels: Connection = None  # raw image data, e.g. from LoadImage.
    vae: Connection = None  # VAE model used for encoding. May be baked-in to a regular SD model.

    latent_out = Output(0)
