"""A ComfyUI node used to load a LoRA extension model."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output


class LoraLoaderNode(ComfyNode):
    """A ComfyUI node used to load a LoRA extension model."""
    CLASS_TYPE: ClassVar[str] = 'LoraLoader'

    lora_name: str
    strength_model: float
    strength_clip: float
    model: Connection = None
    clip: Connection = None

    model_out = Output(0)
    clip_out = Output(1)
