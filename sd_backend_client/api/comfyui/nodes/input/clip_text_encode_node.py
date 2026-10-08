"""A ComfyUI node used to encode text using CLIP."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output


class ClipTextEncodeNode(ComfyNode):
    """A ComfyUI node used to encode text using CLIP."""
    CLASS_TYPE: ClassVar[str] = 'CLIPTextEncode'

    text: str
    clip: Connection = None

    conditioning_out = Output(0)
