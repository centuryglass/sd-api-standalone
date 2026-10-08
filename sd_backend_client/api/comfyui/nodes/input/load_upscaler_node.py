"""A ComfyUI node used to load an image upscaling model."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Output


class LoadUpscalerNode(ComfyNode):
    """A ComfyUI node used to load an image upscaling model."""
    CLASS_TYPE: ClassVar[str] = 'UpscaleModelLoader'

    model_name: str

    upscale_model_out = Output(0)
