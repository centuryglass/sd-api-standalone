"""Integration tests for the shared `DiffusionParams` meanings on live backends.

The metadata tests check `sampler_names`' tables against each server's sampler and scheduler lists, and warn about
table names a server does not list. The generation tests (``--run-generation``) check that img2img resizes the source
to `width` x `height` on both backends and that WebUI applies the requested scheduler.
"""
import warnings

import pytest

from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams
from sd_backend_client.api.shared_data.sampler_names import SAMPLER_WEBUI_NAMES, SCHEDULER_WEBUI_NAMES
from sd_backend_client.errors import SDBackendError

from .comfy_helpers import COMFY_STEPS, build_comfy_params, wait_for_comfy_images
from .helpers import fast_request_body, make_structured_image, save_output

pytestmark = pytest.mark.integration

DEFAULTS = DiffusionParams()
SOURCE_SIZE = (192, 128)
TARGET_SIZE = (256, 320)


def _warn_unlisted(kind: str, server: str, names: set[str], listed: set[str]) -> None:
    missing = sorted(names - listed)
    if missing:
        warnings.warn(f'{server} does not list these mapped {kind} names: {missing}')


def test_comfyui_lists_the_shared_names(comfy_service):
    """ComfyUI lists the default sampler and scheduler; other unlisted table names are reported as warnings."""
    samplers = set(comfy_service.get_sampler_names())
    schedulers = set(comfy_service.get_scheduler_names())
    assert DEFAULTS.sampler in samplers
    assert DEFAULTS.scheduler in schedulers
    _warn_unlisted('sampler', 'ComfyUI', set(SAMPLER_WEBUI_NAMES), samplers)
    _warn_unlisted('scheduler', 'ComfyUI', set(SCHEDULER_WEBUI_NAMES), schedulers)


def test_webui_lists_the_mapped_names(service):
    """WebUI lists the default sampler's WebUI name; other unlisted table names are reported as warnings."""
    samplers = {sampler.name for sampler in service.get_samplers()}
    assert SAMPLER_WEBUI_NAMES[DEFAULTS.sampler] in samplers
    _warn_unlisted('sampler', 'WebUI', set(SAMPLER_WEBUI_NAMES.values()), samplers)
    try:
        schedulers = {scheduler['name'] for scheduler in service.get('/sdapi/v1/schedulers').json()}
    except SDBackendError:
        pytest.skip('WebUI has no /sdapi/v1/schedulers endpoint (A1111 before 1.9)')
    _warn_unlisted('scheduler', 'WebUI', set(SCHEDULER_WEBUI_NAMES.values()), schedulers)


@pytest.mark.generation
def test_webui_img2img_resizes_source_and_applies_scheduler(service, output_dir):
    """WebUI img2img returns a width x height image and reports the requested scheduler."""
    body = fast_request_body('a watercolor painting')
    body.width, body.height = TARGET_SIZE
    body.sampler = 'dpmpp_2m'
    body.scheduler = 'karras'
    source = make_structured_image(*SOURCE_SIZE)
    response = service.img2img(source, request_body=body)
    result = response['images'][0]
    save_output(output_dir, 'shared_params_webui_img2img', result)
    assert result.size == TARGET_SIZE
    info = response['info']
    assert info is not None
    assert info.sampler_name == 'DPM++ 2M'
    assert 'Schedule type: Karras' in info.infotexts[0]


@pytest.mark.generation
def test_comfyui_img2img_resizes_source(comfy_service, comfy_checkpoint, output_dir):
    """ComfyUI img2img returns a width x height image from a source of another size."""
    params = build_comfy_params(comfy_checkpoint, prompt='a watercolor painting')
    params.width, params.height = TARGET_SIZE
    params.steps = COMFY_STEPS
    params.init_images = [make_structured_image(*SOURCE_SIZE)]
    images = wait_for_comfy_images(comfy_service, comfy_service.img2img(params))
    save_output(output_dir, 'shared_params_comfyui_img2img', images[0])
    assert images[0].size == TARGET_SIZE
