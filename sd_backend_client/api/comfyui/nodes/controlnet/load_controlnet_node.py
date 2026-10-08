"""A ComfyUI node used to load a ControlNet model."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Output


class LoadControlNetNode(ComfyNode):
    """A ComfyUI node used to load a ControlNet Model"""
    CLASS_TYPE: ClassVar[str] = 'ControlNetLoader'

    control_net_name: str  # model name

    controlnet_out = Output(0)
