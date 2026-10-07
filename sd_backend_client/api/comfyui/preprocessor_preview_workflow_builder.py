"""Creates a minimal ComfyUI workflow used to preview a ControlNet preprocessor."""
from typing import Any, Optional

from sd_backend_client.api.comfyui.comfyui_types import ImageFileReference
from sd_backend_client.api.comfyui.nodes.comfy_node_graph import ComfyNodeGraph
from sd_backend_client.api.comfyui.nodes.controlnet.dynamic_preprocessor_node import DynamicPreprocessorNode
from sd_backend_client.api.comfyui.nodes.input.load_image_mask_node import LoadImageMaskNode
from sd_backend_client.api.comfyui.nodes.input.load_image_node import LoadImageNode
from sd_backend_client.api.comfyui.nodes.save_image_node import SaveImageNode
from sd_backend_client.api.comfyui.workflow_builder_utils import image_ref_to_str
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor, \
    PreprocessorParams


class PreprocessorPreviewWorkflowBuilder:
    """Builds a workflow that runs one ControlNet preprocessor on an uploaded image and saves the result."""

    def __init__(self, preprocessor: ControlNetPreprocessor | PreprocessorParams) -> None:
        """Every parameter takes its default value unless `preprocessor` is a `PreprocessorParams` that sets it."""
        overrides: dict[str, Any] = {}
        if isinstance(preprocessor, PreprocessorParams):
            typedef = preprocessor.typedef
            overrides = preprocessor.parameter_values
        else:
            typedef = preprocessor
        self._preprocessor = typedef
        control_inputs = {parameter.key: parameter.default_value for parameter in typedef.parameters}
        control_inputs.update(overrides)
        self._preprocessor_node = DynamicPreprocessorNode(typedef.name, control_inputs, typedef.has_image_input,
                                                          typedef.has_mask_input)

    def build_workflow(self, source_image: Optional[ImageFileReference],
                       mask: Optional[ImageFileReference] = None) -> ComfyNodeGraph:
        """Use the provided parameters to build a complete workflow graph.

        Raises ValueError if the preprocessor takes an image input and source_image is None."""
        workflow = ComfyNodeGraph()

        if self._preprocessor_node.has_image_input:
            if source_image is None:
                raise ValueError(f'Preprocessor {self._preprocessor.name!r} needs a source image.')
            load_image_node = LoadImageNode(image_ref_to_str(source_image))
            workflow.connect_nodes(self._preprocessor_node, DynamicPreprocessorNode.IMAGE,
                                   load_image_node, LoadImageNode.IDX_IMAGE)
        if mask is not None and self._preprocessor_node.has_mask_input:
            mask_node = LoadImageMaskNode(image_ref_to_str(mask))
            workflow.connect_nodes(self._preprocessor_node, DynamicPreprocessorNode.MASK,
                                   mask_node, LoadImageMaskNode.IDX_MASK)

        # Save preview image:
        filename_prefix = f'{self._preprocessor.name}_preview'
        save_image_node = SaveImageNode(filename_prefix)
        workflow.connect_nodes(save_image_node, SaveImageNode.IMAGES,
                               self._preprocessor_node, DynamicPreprocessorNode.IDX_IMAGE)
        return workflow
