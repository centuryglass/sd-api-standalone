"""Parity tests: one `DiffusionParams` sent to either backend asks for the same generation.

Each case serializes the same params through `DiffusionRequestBody.to_dict()` (WebUI) and
`DiffusionWorkflowBuilder` (ComfyUI), then checks that every shared field reaches both wire formats with the same
meaning. A field that gains a backend-specific interpretation must fail here.
"""
from typing import Any

import pytest
from PIL import Image

from sd_backend_client.api.comfyui.diffusion_workflow_builder import DiffusionWorkflowBuilder
from sd_backend_client.api.shared_data.diffusion_params import DEFAULT_DENOISING_STRENGTH, DiffusionParams
from sd_backend_client.api.shared_data.sampler_names import (comfyui_sampler_name, comfyui_scheduler_name,
                                                             webui_sampler_name, webui_scheduler_name)
from sd_backend_client.api.webui.diffusion_request_body import DiffusionRequestBody

SOURCE_IMAGE_REF = 'src_image.png [input]'


def _source() -> list[Image.Image]:
    return [Image.new('RGBA', (64, 48), (200, 100, 50, 255))]


CASES: dict[str, dict[str, Any]] = {
    'defaults': {'seed': 1},
    'txt2img': {'prompt': 'a lighthouse', 'negative_prompt': 'fog', 'sd_model_name': 'model.safetensors',
                'sampler': 'dpmpp_2m', 'scheduler': 'karras', 'batch_size': 3, 'steps': 14, 'cfg_scale': 5.5,
                'width': 640, 'height': 384, 'seed': 77},
    'img2img_default_denoise': {'prompt': 'oil painting', 'width': 320, 'height': 256, 'seed': 3,
                                'batch_size': 2},
    'img2img_webui_names': {'prompt': 'pencil sketch', 'sampler': 'DPM++ 2M', 'scheduler': 'Karras', 'seed': 9,
                            'denoising_strength': 0.35, 'width': 128, 'height': 96},
    'legacy_sampler_name_key': {'sampler_name': 'Euler a', 'scheduler': 'ddim_uniform', 'seed': 5},
    'unmapped_names_pass_through': {'sampler': 'Restart', 'scheduler': 'align_your_steps', 'seed': 6},
}
IMG2IMG_CASES = {'img2img_default_denoise', 'img2img_webui_names'}


def _nodes(workflow: dict, class_type: str) -> list[dict]:
    return [node for node in workflow.values() if node['class_type'] == class_type]


def _single(workflow: dict, class_type: str) -> dict:
    nodes = _nodes(workflow, class_type)
    assert len(nodes) == 1, f'expected one {class_type} node, found {len(nodes)}'
    return nodes[0]


def _serialize(case: str) -> tuple[DiffusionParams, dict[str, Any], dict]:
    """Returns the params for a case, their WebUI request dict and their ComfyUI workflow dict."""
    fields = dict(CASES[case])
    if case in IMG2IMG_CASES:
        fields['init_images'] = _source()
    params = DiffusionParams(**fields)
    webui = DiffusionRequestBody(**params.model_dump()).to_dict()
    builder = DiffusionWorkflowBuilder()
    builder.load_diffusion_parameters(params)
    if params.init_images:
        builder.source_image = SOURCE_IMAGE_REF
    return params, webui, builder.build_workflow().get_workflow_dict()


@pytest.mark.parametrize('case', CASES)
def test_core_fields_agree(case):
    """Prompts, steps, CFG, seed and checkpoint reach both backends unchanged."""
    params, webui, workflow = _serialize(case)
    ksampler = _single(workflow, 'KSampler')['inputs']
    assert webui['steps'] == ksampler['steps'] == params.steps
    assert webui['cfg_scale'] == ksampler['cfg'] == params.cfg_scale
    assert webui['seed'] == ksampler['seed'] == params.seed
    assert webui['prompt'] == params.prompt and webui['negative_prompt'] == params.negative_prompt
    encoded = sorted(node['inputs']['text'] for node in _nodes(workflow, 'CLIPTextEncode'))
    assert encoded == sorted([params.prompt, params.negative_prompt])
    checkpoint = _single(workflow, 'CheckpointLoaderSimple')['inputs']['ckpt_name']
    assert checkpoint == params.sd_model_name
    assert webui.get('override_settings', {}).get('sd_model_checkpoint', '') == params.sd_model_name


@pytest.mark.parametrize('case', CASES)
def test_sampler_and_scheduler_agree(case):
    """Both backends get the same sampler and scheduler, each under its own name."""
    params, webui, workflow = _serialize(case)
    ksampler = _single(workflow, 'KSampler')['inputs']
    assert ksampler['sampler_name'] == comfyui_sampler_name(params.sampler)
    assert ksampler['scheduler'] == comfyui_scheduler_name(params.scheduler)
    # Translating ComfyUI's choice to WebUI gives the name WebUI was sent:
    assert webui_sampler_name(ksampler['sampler_name']) == webui['sampler_name']
    assert webui_scheduler_name(ksampler['scheduler']) == webui['scheduler']
    assert 'sampler' not in webui


@pytest.mark.parametrize('case', CASES)
def test_size_batch_and_denoising_agree(case):
    """Both backends produce width x height images in batches of batch_size, with the same denoising strength."""
    params, webui, workflow = _serialize(case)
    assert (webui['width'], webui['height']) == (params.width, params.height)
    assert webui['batch_size'] == params.batch_size
    ksampler = _single(workflow, 'KSampler')['inputs']
    if params.init_images:
        scale = _single(workflow, 'ImageScale')['inputs']
        assert (scale['width'], scale['height']) == (params.width, params.height)
        assert _single(workflow, 'RepeatLatentBatch')['inputs']['amount'] == params.batch_size
        expected_denoise = DEFAULT_DENOISING_STRENGTH if params.denoising_strength is None \
            else params.denoising_strength
        assert webui['denoising_strength'] == ksampler['denoise'] == expected_denoise
    else:
        latent = _single(workflow, 'EmptyLatentImage')['inputs']
        assert (latent['width'], latent['height'], latent['batch_size']) == \
            (params.width, params.height, params.batch_size)
        assert ksampler['denoise'] == 1.0
        assert 'denoising_strength' not in webui


def test_shared_defaults_name_the_same_sampler_on_both_backends():
    """Default params select Euler ancestral with the normal schedule on both backends."""
    _, webui, workflow = _serialize('defaults')
    ksampler = _single(workflow, 'KSampler')['inputs']
    assert (ksampler['sampler_name'], ksampler['scheduler']) == ('euler_ancestral', 'normal')
    assert (webui['sampler_name'], webui['scheduler']) == ('Euler a', 'normal')


def test_legacy_sampler_name_key_sets_sampler():
    """The WebUI request key `sampler_name` is accepted as a constructor alias for `sampler`."""
    assert DiffusionParams(sampler_name='DPM++ 2M').sampler == 'DPM++ 2M'
