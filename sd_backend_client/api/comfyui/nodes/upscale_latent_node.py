"""A ComfyUI node used to scale a latent image to an arbitrary resolution."""
from typing import ClassVar, Literal

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output


class UpscaleLatentNode(ComfyNode):
    """A ComfyUI node used to scale a latent image using a basic pixel scaling algorithm."""
    CLASS_TYPE: ClassVar[str] = 'LatentUpscale'

    width: int
    height: int
    upscale_method: Literal['nearest-exact', 'bilinear', 'area', 'bicubic', 'bislerp'] = 'bilinear'
    crop: Literal['disabled', 'center'] = 'disabled'
    samples: Connection = None

    latent_out = Output(0)
