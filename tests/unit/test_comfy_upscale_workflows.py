"""Unit tests for the ComfyUI upscaling workflows: graph shape and output sizing, no server needed."""
from typing import Optional

import pytest
from PIL import Image

from intrapaint_api.api.comfyui.basic_upscale_workflow_builder import build_basic_upscaling_workflow
from intrapaint_api.api.comfyui.comfyui_types import ImageFileReference
from intrapaint_api.api.comfyui.latent_upscale_workflow_builder import LatentUpscaleWorkflowBuilder
from intrapaint_api.api.comfyui_webservice import ComfyUiWebservice
from intrapaint_api.util.geometry import Size

SOURCE = ImageFileReference(filename='source.png', subfolder='')
FINAL_SIZE = Size(1536, 1024)


def nodes_of_type(workflow: dict, class_type: str) -> list[dict]:
    """All nodes in a workflow dict with the given class_type."""
    return [node for node in workflow.values() if node['class_type'] == class_type]


def single_node(workflow: dict, class_type: str) -> dict:
    """The one node of a given class_type."""
    matches = nodes_of_type(workflow, class_type)
    assert len(matches) == 1, f'expected exactly one {class_type} node, found {len(matches)}'
    return matches[0]


def source_class(workflow: dict, connection: list) -> str:
    """The class_type of the node a connection points at."""
    return workflow[connection[0]]['class_type']


def build_latent_workflow(ultimate: bool, upscale_model: Optional[str]) -> dict:
    """Build the latent upscale graph for a 3x upscale to FINAL_SIZE."""
    builder = LatentUpscaleWorkflowBuilder(SOURCE, 3.0, FINAL_SIZE, Size(512, 512), ultimate, upscale_model)
    builder.sd_model = 'model.safetensors'
    return builder.build_workflow().get_workflow_dict()


def test_basic_workflow_resizes_model_output_to_target():
    """The basic graph resizes the model output to the exact target size."""
    workflow = build_basic_upscaling_workflow(SOURCE, 'esrgan.pth', FINAL_SIZE).get_workflow_dict()

    resize = single_node(workflow, 'ImageScale')
    assert resize['inputs']['width'] == 1536
    assert resize['inputs']['height'] == 1024
    assert resize['inputs']['crop'] == 'disabled'
    apply_model = single_node(workflow, 'ImageUpscaleWithModel')
    assert source_class(workflow, resize['inputs']['image']) == 'ImageUpscaleWithModel'
    assert source_class(workflow, apply_model['inputs']['image']) == 'LoadImage'
    assert source_class(workflow, apply_model['inputs']['upscale_model']) == 'UpscaleModelLoader'
    save = single_node(workflow, 'SaveImage')
    assert source_class(workflow, save['inputs']['images']) == 'ImageScale'


def test_latent_workflow_without_ultimate_applies_upscale_model_then_resizes():
    """Without Ultimate SD Upscale, the upscale model runs before the resize and VAE encode."""
    workflow = build_latent_workflow(False, 'esrgan.pth')

    resize = single_node(workflow, 'ImageScale')
    assert (resize['inputs']['width'], resize['inputs']['height']) == (1536, 1024)
    apply_model = single_node(workflow, 'ImageUpscaleWithModel')
    assert source_class(workflow, resize['inputs']['image']) == 'ImageUpscaleWithModel'
    assert source_class(workflow, apply_model['inputs']['image']) == 'LoadImage'
    assert source_class(workflow, apply_model['inputs']['upscale_model']) == 'UpscaleModelLoader'
    encode = single_node(workflow, 'VAEEncodeTiled')
    assert source_class(workflow, encode['inputs']['pixels']) == 'ImageScale'
    latent_scale = single_node(workflow, 'LatentUpscale')
    assert (latent_scale['inputs']['width'], latent_scale['inputs']['height']) == (1536, 1024)


def test_latent_workflow_without_ultimate_or_model_scales_latent_only():
    """With no upscale model, only latent scaling sets the size."""
    workflow = build_latent_workflow(False, None)

    assert not nodes_of_type(workflow, 'ImageScale')
    assert not nodes_of_type(workflow, 'ImageUpscaleWithModel')
    assert not nodes_of_type(workflow, 'UpscaleModelLoader')
    encode = single_node(workflow, 'VAEEncodeTiled')
    assert source_class(workflow, encode['inputs']['pixels']) == 'LoadImage'
    latent_scale = single_node(workflow, 'LatentUpscale')
    assert (latent_scale['inputs']['width'], latent_scale['inputs']['height']) == (1536, 1024)


def test_ultimate_workflow_with_model_wires_model_and_uses_multiplier():
    """The Ultimate SD Upscale node receives the model and the multiplier."""
    workflow = build_latent_workflow(True, 'esrgan.pth')

    ultimate = single_node(workflow, 'UltimateSDUpscale')
    assert ultimate['inputs']['upscale_by'] == 3.0
    assert source_class(workflow, ultimate['inputs']['upscale_model']) == 'UpscaleModelLoader'
    assert source_class(workflow, ultimate['inputs']['image']) == 'LoadImage'
    assert not nodes_of_type(workflow, 'ImageScale')


def test_ultimate_workflow_without_model_prescales_by_multiplier():
    """With no model, the Ultimate SD Upscale input is pre-scaled by the multiplier."""
    workflow = build_latent_workflow(True, None)

    scale_by = single_node(workflow, 'ImageScaleBy')
    assert scale_by['inputs']['scale_by'] == 3.0
    assert not nodes_of_type(workflow, 'UpscaleModelLoader')


@pytest.mark.parametrize('size', [(64, 64), (32, 64), (64, 32)])
def test_upscale_rejects_non_enlarging_size(size: tuple[int, int]):
    """upscale() raises ValueError when the target does not exceed the source."""
    # The size check runs before any request, so no server is contacted.
    service = ComfyUiWebservice('http://127.0.0.1:1')
    with pytest.raises(ValueError, match='must exceed'):
        service.upscale(Image.new('RGBA', (64, 64)), *size)
