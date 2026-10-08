"""A ComfyUI node used to control the diffusion sampling process."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output


class KSamplerNode(ComfyNode):
    """Diffusion sampler node."""
    CLASS_TYPE: ClassVar[str] = 'KSampler'

    cfg: float  # Guidance scale
    steps: int
    sampler_name: str
    denoise: float = 1.0  # Denoising strength, 1.0 for txt2img
    scheduler: str = 'normal'
    seed: int = -1
    latent_image: Connection = None  # Image source node, e.g. EmptyLatentImage
    model: Connection = None  # Usually CheckpointLoaderSimple
    negative: Connection = None  # Usually CLIPTextEncode
    positive: Connection = None  # Usually CLIPTextEncode

    latent_out = Output(0)
