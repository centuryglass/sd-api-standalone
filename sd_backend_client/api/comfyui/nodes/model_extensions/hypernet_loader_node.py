"""A ComfyUI node used to load a hypernetwork model extension."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output


class HypernetLoaderNode(ComfyNode):
    """A ComfyUI node used to load a hypernetwork model extension."""
    CLASS_TYPE: ClassVar[str] = 'HypernetworkLoader'

    hypernetwork_name: str
    strength: float
    model: Connection = None

    model_out = Output(0)
