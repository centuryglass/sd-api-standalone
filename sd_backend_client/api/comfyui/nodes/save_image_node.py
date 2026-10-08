"""A ComfyUI node used to save image data."""
from typing import NotRequired, cast, Any
from typing_extensions import TypedDict

from sd_backend_client.api.comfyui.nodes.comfy_node import NodeConnection, ComfyNode

NODE_NAME = 'SaveImage'

# Prefix for images the package's workflows save on the server. ComfyUI names files `<prefix>_00001_.png`, so an
# empty prefix produces hidden dotfiles.
DEFAULT_FILENAME_PREFIX = 'sd_backend_client'


class SaveImageInputs(TypedDict):
    """SaveImage input parameters."""
    filename_prefix: str
    images: NotRequired[NodeConnection]  # Image source, e.g. VAE decoder


class SaveImageNode(ComfyNode):
    """A ComfyUI node used to save image data."""

    # Connection keys:
    IMAGES = 'images'

    def __init__(self, filename_prefix: str = DEFAULT_FILENAME_PREFIX) -> None:
        data: SaveImageInputs = {
            'filename_prefix': filename_prefix
        }
        super().__init__(NODE_NAME, cast(dict[str, Any], data), {SaveImageNode.IMAGES}, 0)
