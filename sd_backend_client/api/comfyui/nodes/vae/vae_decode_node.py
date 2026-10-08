"""A ComfyUI node used to decode latent image data."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output


class VAEDecodeNode(ComfyNode):
    """A ComfyUI node used to decode latent image data."""
    CLASS_TYPE: ClassVar[str] = 'VAEDecode'

    samples: Connection = None  # Latent image data, e.g. from KSampler.
    vae: Connection = None  # VAE model used for decoding. May be baked-in to a regular SD model.

    image_out = Output(0)
