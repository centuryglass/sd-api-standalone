"""Unit tests for the A1111/Forge request body — pure dataclass -> wire-dict logic.

``DiffusionRequestBody`` is already a dataclass, and its ``to_dict()`` is what actually
shapes the JSON sent to the WebUI. These tests pin that transform without a server. They
construct the body directly (never ``load_data`` / ``Cache``), so they survive the config
refactor; ``DiffusionRequestBody`` is essentially the target params dataclass for A1111.
"""
from intrapaint_api.api.webui.diffusion_request_body import DiffusionRequestBody
from intrapaint_api.util.visual.image_utils import image_from_base64


def test_to_dict_strips_unset_optional_fields():
    body = DiffusionRequestBody()
    data = body.to_dict()
    # None-valued optionals must not be serialized (the server would reject/misread them).
    assert 'denoising_strength' not in data
    assert 'mask' not in data
    assert 'init_images' not in data
    # Non-None defaults are kept.
    assert data['steps'] == body.steps
    assert data['cfg_scale'] == body.cfg_scale
    assert data['width'] == body.width and data['height'] == body.height


def test_to_dict_keeps_explicitly_set_optionals():
    body = DiffusionRequestBody()
    body.denoising_strength = 0.55
    img_str = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4////fwAJ+wP9KobjigAAAABJRU5ErkJggg=='
    img = image_from_base64(img_str)
    body.mask = img
    data = body.to_dict()
    assert data['denoising_strength'] == 0.55
    assert data['mask'] == img_str


def test_to_dict_passes_through_core_generation_params():
    body = DiffusionRequestBody()
    body.prompt = 'a red apple'
    body.negative_prompt = 'blurry'
    body.seed = 99
    body.steps = 7
    body.sampler_name = 'Euler a'
    data = body.to_dict()
    assert data['prompt'] == 'a red apple'
    assert data['negative_prompt'] == 'blurry'
    assert data['seed'] == 99
    assert data['steps'] == 7
    assert data['sampler_name'] == 'Euler a'


# def test_add_init_image_appends_base64_data_uri():
# TODO: test to_dict serializes correctly with base64 images