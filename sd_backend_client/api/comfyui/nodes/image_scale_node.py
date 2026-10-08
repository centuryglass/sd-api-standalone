"""A ComfyUI node used to resize an image to an exact width and height."""
from typing import ClassVar, Literal

from sd_backend_client.api.comfyui.nodes.basic_scaling_node import DEFAULT_UPSCALE_METHOD, ImageUpscaleMethod
from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output


class ImageScaleNode(ComfyNode):
    """A ComfyUI node used to resize an image to an exact size. Unlike `BasicScalingNode`, the target is a pixel size
       and not a multiplier, so the aspect ratio changes if it differs from the source's."""
    CLASS_TYPE: ClassVar[str] = 'ImageScale'

    width: int
    height: int
    upscale_method: ImageUpscaleMethod = DEFAULT_UPSCALE_METHOD
    crop: Literal['disabled', 'center'] = 'disabled'
    image: Connection = None

    image_out = Output(0)
