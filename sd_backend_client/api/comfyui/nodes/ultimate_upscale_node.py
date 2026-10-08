"""A ComfyUI node used to apply the 'Ultimate SD Upscale' script."""
from typing import Any, ClassVar, Optional

from pydantic import Field

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output
from sd_backend_client.api.shared_data.api_datatypes import RedrawMode, SeamFixMode

ULTIMATE_UPSCALE_NODE_NAME = 'UltimateSDUpscale'
ULTIMATE_UPSCALE_NODE_WITHOUT_UPSCALE_MODEL = 'UltimateSDUpscaleNoUpscale'


class UltimateUpscaleNode(ComfyNode):
    """A ComfyUI node used to apply the 'Ultimate SD Upscale' script.

    `use_upscaler` picks the node variant. The variant without an upscale model leaves out `upscale_by`, takes no
    `upscale_model` connection, and names its image input `upscaled_image`.
    """
    CLASS_TYPE: ClassVar[str] = ULTIMATE_UPSCALE_NODE_NAME

    use_upscaler: bool = Field(True, exclude=True)

    upscale_by: Optional[float] = None  # Upscaler variant only.
    seed: int = 0
    steps: int = 20
    cfg: float = 8.0
    sampler_name: str = 'euler'
    scheduler: str = 'normal'
    denoise: float = 0.35
    mode_type: RedrawMode = 'Linear'
    tile_width: int = 512
    tile_height: int = 512
    mask_blur: int = 8
    tile_padding: int = 32
    force_uniform_tiles: bool = False
    tiled_decode: bool = True

    seam_fix_mode: SeamFixMode = 'None'
    seam_fix_denoise: float = 0.35
    seam_fix_width: int = 64
    seam_fix_mask_blur: int = 8
    seam_fix_padding: int = 16

    image: Connection = None  # Image source node
    model: Connection = None  # Usually CheckpointLoaderSimple
    positive: Connection = None  # Usually CLIPTextEncode
    negative: Connection = None  # Usually CLIPTextEncode
    vae: Connection = None
    upscale_model: Connection = None  # Upscaler variant only.

    image_out = Output(0)

    @property
    def class_type(self) -> str:
        return ULTIMATE_UPSCALE_NODE_NAME if self.use_upscaler else ULTIMATE_UPSCALE_NODE_WITHOUT_UPSCALE_MODEL

    def inputs(self) -> dict[str, Any]:
        inputs = super().inputs()
        if not self.use_upscaler:
            inputs.pop('upscale_by', None)
            if 'image' in inputs:
                inputs['upscaled_image'] = inputs.pop('image')
        return inputs
