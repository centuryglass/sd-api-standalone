"""Creates a ComfyUI workflow used to apply an upscaling model."""

from typing import Optional

from intrapaint_api.api.comfyui.comfyui_types import ImageFileReference
from intrapaint_api.api.comfyui.nodes.apply_upscaler_node import ApplyUpscalerNode
from intrapaint_api.api.comfyui.nodes.comfy_node import ComfyNode
from intrapaint_api.api.comfyui.nodes.comfy_node_graph import ComfyNodeGraph
from intrapaint_api.api.comfyui.nodes.image_scale_node import ImageScaleNode
from intrapaint_api.api.comfyui.nodes.input.load_image_node import LoadImageNode
from intrapaint_api.api.comfyui.nodes.input.load_upscaler_node import LoadUpscalerNode
from intrapaint_api.api.comfyui.nodes.save_image_node import SaveImageNode
from intrapaint_api.api.comfyui.workflow_builder_utils import image_ref_to_str
from intrapaint_api.util.geometry import Size


def build_basic_upscaling_workflow(source_image: ImageFileReference, upscale_model_name: str,
                                  final_image_size: Optional[Size] = None) -> ComfyNodeGraph:
    """Creates a ComfyUI workflow that applies an upscaling model, then resizes the result to `final_image_size`.

    The model's native scale factor rarely matches the requested size, so the resize step sets the output size.
    When `final_image_size` is None the resize is skipped and the output has the model's native scale."""
    workflow = ComfyNodeGraph()
    load_image_node = LoadImageNode(image_ref_to_str(source_image))
    load_upscaler_node = LoadUpscalerNode(upscale_model_name)
    apply_upscaler_node = ApplyUpscalerNode()
    workflow.connect_nodes(apply_upscaler_node, ApplyUpscalerNode.UPSCALE_MODEL,
                           load_upscaler_node, LoadUpscalerNode.IDX_UPSCALE_MODEL)
    workflow.connect_nodes(apply_upscaler_node, ApplyUpscalerNode.IMAGE,
                           load_image_node, LoadImageNode.IDX_IMAGE)
    output_node: ComfyNode = apply_upscaler_node
    output_idx = ApplyUpscalerNode.IDX_IMAGE
    if final_image_size is not None:
        resize_node = ImageScaleNode(final_image_size.width(), final_image_size.height())
        workflow.connect_nodes(resize_node, ImageScaleNode.IMAGE, output_node, output_idx)
        output_node = resize_node
        output_idx = ImageScaleNode.IDX_IMAGE
    save_image_node = SaveImageNode('')
    workflow.connect_nodes(save_image_node, SaveImageNode.IMAGES, output_node, output_idx)
    return workflow
