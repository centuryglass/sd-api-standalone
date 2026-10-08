"""A ComfyUI node used to load image data."""
from typing import ClassVar, Literal

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Output


class LoadImageNode(ComfyNode):
    """A ComfyUI node used to load image data."""
    CLASS_TYPE: ClassVar[str] = 'LoadImage'

    image: str  # Uploaded image name
    upload: Literal['image'] = 'image'

    image_out = Output(0)
