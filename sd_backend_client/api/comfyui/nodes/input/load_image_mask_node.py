"""A ComfyUI node used to load image mask data."""
from typing import ClassVar, Literal

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Output


class LoadImageMaskNode(ComfyNode):
    """A ComfyUI node used to load one channel of an uploaded image as a mask."""
    CLASS_TYPE: ClassVar[str] = 'LoadImageMask'

    image: str  # Uploaded image name
    channel: Literal['alpha', 'red', 'green', 'blue'] = 'alpha'
    upload: Literal['image'] = 'image'

    mask_out = Output(0)
