"""A ComfyUI node used to apply an upscaling model."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output


class ApplyUpscalerNode(ComfyNode):
    """A ComfyUI node used to apply an upscaling model."""
    CLASS_TYPE: ClassVar[str] = 'ImageUpscaleWithModel'

    upscale_model: Connection = None
    image: Connection = None

    image_out = Output(0)
