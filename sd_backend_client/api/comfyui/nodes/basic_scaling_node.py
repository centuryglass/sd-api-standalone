"""A ComfyUI node used to scale an image using a basic pixel scaling algorithm."""
from typing import ClassVar, Literal, TypeAlias

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output

ImageUpscaleMethod: TypeAlias = Literal['nearest-exact', 'bilinear', 'area', 'bicubic', 'lanczos']
DEFAULT_UPSCALE_METHOD: ImageUpscaleMethod = 'lanczos'


class BasicScalingNode(ComfyNode):
    """A ComfyUI node used to scale an image using a basic pixel scaling algorithm."""
    CLASS_TYPE: ClassVar[str] = 'ImageScaleBy'

    scale_by: float
    upscale_method: ImageUpscaleMethod = DEFAULT_UPSCALE_METHOD
    image: Connection = None

    image_out = Output(0)
