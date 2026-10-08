"""A ComfyUI node used to apply a mask to latent image data."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output


class LatentMaskNode(ComfyNode):
    """A ComfyUI node used to apply a mask to latent image data."""
    CLASS_TYPE: ClassVar[str] = 'SetLatentNoiseMask'

    samples: Connection = None  # latent image data, e.g. from VAEEncodeNode.
    mask: Connection = None  # mask data, e.g. from LoadImageMask.

    latent_out = Output(0)
