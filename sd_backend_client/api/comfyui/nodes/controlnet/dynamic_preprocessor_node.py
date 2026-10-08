"""A ComfyUI node used to pre-process image data for ControlNet. Rather than a specific node type, this class can
   stand in for nearly any preprocessor node type, as long as it only has one output connection (type IMAGE) and its
   only connection inputs are an image, a mask, or both."""
from typing import Any

from pydantic import Field, model_validator

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output


class DynamicPreprocessorNode(ComfyNode):
    """A ComfyUI preprocessor node of any type, named by `node_name` with non-connection inputs in `parameters`."""

    node_name: str = Field(exclude=True)
    parameters: dict[str, Any] = Field(exclude=True)
    has_image_input: bool = Field(True, exclude=True)
    """Whether this node takes an image input."""

    has_mask_input: bool = Field(False, exclude=True)
    """Whether this node takes a mask input."""

    image: Connection = None
    mask: Connection = None

    image_out = Output(0)

    @model_validator(mode='after')
    def _check_connections(self) -> 'DynamicPreprocessorNode':
        if self.image is not None and not self.has_image_input:
            raise ValueError(f'{self.node_name} takes no image input')
        if self.mask is not None and not self.has_mask_input:
            raise ValueError(f'{self.node_name} takes no mask input')
        return self

    @property
    def class_type(self) -> str:
        return self.node_name

    def inputs(self) -> dict[str, Any]:
        return {**self.parameters, **super().inputs()}
