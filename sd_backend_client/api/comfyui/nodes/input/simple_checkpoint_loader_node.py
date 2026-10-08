"""A ComfyUI node used to load a basic Stable Diffusion model, plus its CLIP and VAE models."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Output


class SimpleCheckpointLoaderNode(ComfyNode):
    """Loads a Stable Diffusion model."""
    CLASS_TYPE: ClassVar[str] = 'CheckpointLoaderSimple'

    ckpt_name: str

    model_out = Output(0)
    clip_out = Output(1)
    vae_out = Output(2)
