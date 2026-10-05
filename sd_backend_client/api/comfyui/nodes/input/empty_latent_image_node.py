"""A ComfyUI node used to initialize a batch of latent images."""
from typing import cast, Any
from typing_extensions import TypedDict

from sd_backend_client.util.geometry import Size

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode

NODE_NAME = 'EmptyLatentImage'


class EmptyLatentInputs(TypedDict):
    """Empty latent image loader inputs."""
    batch_size: int
    height: int
    width: int


class EmptyLatentNode(ComfyNode):
    """Creates an empty latent image or image batch."""

    # Output indexes:
    IDX_LATENT = 0

    def __init__(self, batch_size: int, image_size: Size) -> None:
        data: EmptyLatentInputs = {
            'batch_size': batch_size,
            'height': image_size.height(),
            'width': image_size.width()
        }
        super().__init__(NODE_NAME, cast(dict[str, Any], data), set(), 1)
