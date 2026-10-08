"""A ComfyUI node used to apply a ControlNet model to diffusion conditioning data."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output


class ApplyControlNetNode(ComfyNode):
    """A ComfyUI node used to apply a ControlNet model to diffusion conditioning data."""
    CLASS_TYPE: ClassVar[str] = 'ControlNetApplyAdvanced'

    strength: float
    start_percent: float
    end_percent: float
    positive: Connection = None  # Usually CLIPTextEncode
    negative: Connection = None  # Usually CLIPTextEncode
    control_net: Connection = None
    image: Connection = None
    vae: Connection = None

    positive_out = Output(0)
    negative_out = Output(1)
