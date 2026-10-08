"""A ComfyUI node used to save image data."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection

# Prefix for images the package's workflows save on the server. ComfyUI names files `<prefix>_00001_.png`, so an
# empty prefix produces hidden dotfiles.
DEFAULT_FILENAME_PREFIX = 'sd_backend_client'


class SaveImageNode(ComfyNode):
    """A ComfyUI node used to save image data. It has no outputs."""
    CLASS_TYPE: ClassVar[str] = 'SaveImage'

    filename_prefix: str = DEFAULT_FILENAME_PREFIX
    images: Connection = None  # Image source, e.g. VAE decoder
