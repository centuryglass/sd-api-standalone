"""A ComfyUI node used to copy latent image data for batch operations."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output


class RepeatLatentNode(ComfyNode):
    """A ComfyUI node used to copy latent image data for batch operations."""
    CLASS_TYPE: ClassVar[str] = 'RepeatLatentBatch'

    amount: int  # Number of repeated copies.
    samples: Connection = None  # latent image data, e.g. from VAEEncodeNode.

    latent_out = Output(0)
