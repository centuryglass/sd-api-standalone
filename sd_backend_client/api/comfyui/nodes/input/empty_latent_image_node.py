"""A ComfyUI node used to initialize a batch of latent images."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Output


class EmptyLatentNode(ComfyNode):
    """Creates an empty latent image or image batch."""
    CLASS_TYPE: ClassVar[str] = 'EmptyLatentImage'

    batch_size: int
    width: int
    height: int

    latent_out = Output(0)
