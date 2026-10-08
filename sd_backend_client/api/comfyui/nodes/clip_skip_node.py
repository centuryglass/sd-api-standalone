"""A ComfyUI node used to apply the 'CLIP skip' option to prompt conditioning."""
from typing import ClassVar

from pydantic import field_validator

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output


class CLIPSkipNode(ComfyNode):
    """A ComfyUI node used to apply the 'CLIP skip' option to prompt conditioning."""
    CLASS_TYPE: ClassVar[str] = 'CLIPSetLastLayer'

    stop_at_clip_layer: int
    """Last CLIP layer to use, as a negative index. A positive CLIP skip value is negated."""

    clip: Connection = None

    clip_out = Output(0)

    @field_validator('stop_at_clip_layer')
    @classmethod
    def _negate_clip_skip(cls, stop_layer: int) -> int:
        return -stop_layer if stop_layer > 0 else stop_layer
