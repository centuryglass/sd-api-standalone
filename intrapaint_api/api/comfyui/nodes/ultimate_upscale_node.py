"""A ComfyUI node used to apply the 'Ultimate SD Upscale' script."""
from typing import cast, Any, Literal, Optional

from pydantic import BaseModel

from intrapaint_api.api.comfyui.nodes.comfy_node import NodeConnection, ComfyNode

ULTIMATE_UPSCALE_NODE_NAME = 'UltimateSDUpscale'
ULTIMATE_UPSCALE_NODE_WITHOUT_UPSCALE_MODEL = 'UltimateSDUpscaleNoUpscale'


class UltimateUpscaleCoreInputs(BaseModel):
    """Primary inputs, excluding those related to the optional seam fix mode."""
    upscale_by: Optional[float] = None  # Leave out only if the "no upscale" option is chosen.
    seed: int = 0
    steps: int = 20
    cfg: float  = 8.0
    sampler_name: str = 'euler'
    scheduler: str = 'normal'
    denoise: float  = 0.35
    mode_type: Literal['Linear', 'Chess', 'None'] = 'Linear'
    tile_width: int = 512
    tile_height: int = 512
    mask_blur: int = 8
    tile_padding: int =32
    force_uniform_tiles: bool = False
    tiled_decode: bool = True


class SeamFixInputs(BaseModel):
    """Inputs for the optional 'Seam Fix' mode:"""
    seam_fix_mode: Literal['None', 'Band Pass', 'Half Tile', 'Half Tile + Intersections'] = 'None'
    seam_fix_denoise: float = 0.35
    seam_fix_width: int = 64
    seam_fix_mask_blur: int = 8
    seam_fix_padding: int = 16


IMAGE_KEY_WITH_UPSCALER = 'image'
IMAGE_KEY_WITHOUT_UPSCALER = 'upscaled_image'


class UltimateUpscaleInputs(UltimateUpscaleCoreInputs, SeamFixInputs):
    """Full inputs for the "Ultimate SD Upscale" node."""
    image: Optional[NodeConnection] = None # Image source node (upscaler mode only)
    upscaled_image: Optional[NodeConnection] = None  # Image source node (no upscaler mode only)
    model: Optional[NodeConnection] = None  # Usually CheckpointLoaderSimple
    positive: Optional[NodeConnection] = None  # Usually CLIPTextEncode
    negative: Optional[NodeConnection] = None  # Usually CLIPTextEncode
    vae: Optional[NodeConnection] = None
    upscale_model: Optional[NodeConnection] = None  # Leave out only if the "no upscale" option is chosen.


class UltimateUpscaleNode(ComfyNode):
    """A ComfyUI node used to apply the 'Ultimate SD Upscale' script."""

    # Connection keys:
    IMAGE = 'image'
    MODEL = 'model'
    POSITIVE = 'positive'
    NEGATIVE = 'negative'
    VAE = 'vae'
    UPSCALE_MODEL = 'upscale_model'

    # Output indexes:
    IDX_IMAGE = 0

    # noinspection PyTypeChecker
    def __init__(self,
                 core_inputs: UltimateUpscaleCoreInputs, use_upscaler=True,
                 seam_fix_settings: Optional[SeamFixInputs] = None) -> None:
        if seam_fix_settings is None:
            seam_fix_settings = SeamFixInputs()
        data = UltimateUpscaleInputs(**core_inputs.model_dump(), **seam_fix_settings.model_dump())
        connection_params = {
            UltimateUpscaleNode.IMAGE,
            IMAGE_KEY_WITHOUT_UPSCALER,
            UltimateUpscaleNode.MODEL,
            UltimateUpscaleNode.POSITIVE,
            UltimateUpscaleNode.NEGATIVE,
            UltimateUpscaleNode.VAE
        }
        node_name = ULTIMATE_UPSCALE_NODE_WITHOUT_UPSCALE_MODEL
        if use_upscaler:
            node_name = ULTIMATE_UPSCALE_NODE_NAME
            data.upscale_by = core_inputs.upscale_by
            connection_params.add(UltimateUpscaleNode.UPSCALE_MODEL)
        super().__init__(node_name, cast(dict[str, Any], data), connection_params, 1)

    def add_input(self, connected_node: str, output_slot_index: int, input_key: str):
        """Connect one of this node's inputs to another node's output.

        This will check the validity of the input key, but doesn't do anything to validate that the output is correct.
        It will also correct the image key if necessary, since the Ultimate Upscale node uses a slightly different key
        depending on which variant of it is being used.
        """
        if input_key == IMAGE_KEY_WITHOUT_UPSCALER and self.node_name == ULTIMATE_UPSCALE_NODE_NAME:
            input_key = IMAGE_KEY_WITH_UPSCALER
        elif input_key == IMAGE_KEY_WITH_UPSCALER and self.node_name == ULTIMATE_UPSCALE_NODE_WITHOUT_UPSCALE_MODEL:
            input_key = IMAGE_KEY_WITHOUT_UPSCALER
        super().add_input(connected_node, output_slot_index, input_key)
