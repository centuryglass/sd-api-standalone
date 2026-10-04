"""A ComfyUI node used to resize an image to an exact width and height."""
from typing import NotRequired, Literal, cast
from typing_extensions import TypedDict

from intrapaint_api.api.comfyui.nodes.comfy_node import NodeConnection, ComfyNode

NODE_NAME = 'ImageScale'

DEFAULT_UPSCALE_METHOD = 'lanczos'


class ImageScaleInputs(TypedDict):
    """Resizing parameters."""
    upscale_method: Literal['nearest-exact', 'bilinear', 'area', 'bicubic', 'lanczos']
    width: int
    height: int
    crop: Literal['disabled', 'center']
    image: NotRequired[NodeConnection]


class ImageScaleNode(ComfyNode):
    """A ComfyUI node used to resize an image to an exact size. Unlike `BasicScalingNode`, the target is a pixel size
       and not a multiplier, so the aspect ratio changes if it differs from the source's."""

    # Connection keys:
    IMAGE = 'image'

    # Output indexes:
    IDX_IMAGE = 0

    def __init__(self, width: int, height: int, upscale_method=DEFAULT_UPSCALE_METHOD) -> None:
        connection_params = {ImageScaleNode.IMAGE}
        data: ImageScaleInputs = {
            'width': width,
            'height': height,
            'crop': 'disabled',
            'upscale_method': upscale_method  # type: ignore
        }
        super().__init__(NODE_NAME, cast(dict[str, str], data), connection_params, 1)
