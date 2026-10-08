"""A ComfyUI node used to adjust conditioning for a dedicated inpainting model."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output


class InpaintModelConditioningNode(ComfyNode):
    """A ComfyUI node used to adjust conditioning for a dedicated inpainting model."""
    CLASS_TYPE: ClassVar[str] = 'InpaintModelConditioning'

    positive: Connection = None  # Usually CLIPTextEncode
    negative: Connection = None  # Usually CLIPTextEncode
    vae: Connection = None  # VAE model used for encoding. May be baked-in to a regular SD model.
    pixels: Connection = None  # raw image data, e.g. from LoadImage.
    mask: Connection = None  # Inpainting mask.

    positive_out = Output(0)
    negative_out = Output(1)
    latent_out = Output(2)
