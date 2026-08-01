"""Generation integration tests: real diffusion round-trips.

These actually run the model, so they are slow and require a checkpoint to be loaded
on the server. They are opt-in — run with ``--run-generation`` or
``RUN_SD_GENERATION=1`` (see ``conftest.py``). Everything is kept small (tiny canvas,
a handful of steps) so a round-trip is as cheap as possible.
"""
import pytest
from PIL import Image

from .helpers import (configure_fast_cache, fast_request_body, make_mask, make_test_image,
                      save_output, TEST_SIZE)

pytestmark = [pytest.mark.integration, pytest.mark.generation]


def _assert_valid_image(image, expected_size: int = TEST_SIZE):
    assert isinstance(image, Image.Image)
    assert image.mode == 'RGBA'  # decoded images are normalized to RGBA
    # The server may round dimensions to a multiple of 8; allow a little slack.
    assert abs(image.width - expected_size) <= 8
    assert abs(image.height - expected_size) <= 8


def test_txt2img_produces_image(service, output_dir):
    response = service.txt2img(fast_request_body('a red apple on a wooden table'))
    assert len(response['images']) == 1
    _assert_valid_image(response['images'][0])
    save_output(output_dir, 'txt2img', response['images'][0])


def test_txt2img_info_reports_seed(service):
    body = fast_request_body()
    body.seed = 12345
    response = service.txt2img(body)
    info = response['info']
    assert info is not None
    # The echoed generation info should report the seed we requested.
    assert 12345 in info['all_seeds']


def test_txt2img_batch_returns_multiple_images(service, output_dir):
    body = fast_request_body()
    body.n_iter = 2
    response = service.txt2img(body)
    assert len(response['images']) == 2
    for index, image in enumerate(response['images']):
        _assert_valid_image(image)
        save_output(output_dir, f'txt2img_batch_{index}', image)


def test_img2img_round_trip(service, output_dir):
    body = fast_request_body('turn this into a landscape painting')
    body.denoising_strength = 0.6
    source = make_test_image()
    response = service.img2img(source, request_body=body)
    assert len(response['images']) >= 1
    _assert_valid_image(response['images'][0])
    save_output(output_dir, 'img2img', response['images'][0])


def test_img2img_inpaint_with_mask(service, output_dir):
    body = fast_request_body('a bright flower')
    body.denoising_strength = 0.75
    body.inpainting_fill = 1  # original
    body.inpaint_full_res = False
    body.inpainting_mask_invert = 0
    source = make_test_image()
    mask = make_mask()
    response = service.img2img(source, mask=mask, request_body=body)
    assert len(response['images']) >= 1
    _assert_valid_image(response['images'][0])
    save_output(output_dir, 'inpaint_source', source)
    save_output(output_dir, 'inpaint_mask', mask)
    save_output(output_dir, 'inpaint_result', response['images'][0])


def test_upscale_basic(service, output_dir):
    # Uses the plain "extra-single-image" upscaler path (no SD upscaling / ControlNet),
    # driven by the isolated fast cache profile.
    configure_fast_cache()
    source = make_test_image(128, 128)
    response = service.upscale(source, 256, 256)
    assert len(response['images']) == 1
    result = response['images'][0]
    assert isinstance(result, Image.Image)
    assert result.width >= 200 and result.height >= 200
    save_output(output_dir, 'upscale_result', result)


def test_interrogate_returns_caption(service):
    source = make_test_image()
    caption = service.interrogate(source)
    assert isinstance(caption, str)
    assert len(caption) > 0


def test_interrupt_is_accepted(service):
    # With no active job this is a no-op, but the endpoint should still respond cleanly.
    result = service.interrupt()
    assert isinstance(result, dict)
