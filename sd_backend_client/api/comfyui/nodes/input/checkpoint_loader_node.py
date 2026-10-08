"""A ComfyUI node used to load a Stable Diffusion model with an explicit config file, plus its CLIP and VAE models."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Output


class CheckpointLoaderNode(ComfyNode):
    """A ComfyUI node used to load a Stable Diffusion model with an explicit config file, plus its CLIP and VAE
    models."""
    CLASS_TYPE: ClassVar[str] = 'CheckpointLoader'

    ckpt_name: str
    config_name: str

    model_out = Output(0)
    clip_out = Output(1)
    vae_out = Output(2)
