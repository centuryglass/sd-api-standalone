"""Generation integration tests: real diffusion round-trips.

These actually run the model, so they are slow and require a checkpoint to be loaded
on the server. They are opt-in — run with ``--run-generation`` or
``RUN_SD_GENERATION=1`` (see ``conftest.py``). Everything is kept small (tiny canvas,
a handful of steps) so a round-trip is as cheap as possible.
"""
import pytest
from PIL import Image

from .helpers import (configure_fast_cache, fast_request_body, images_differ, make_mask,
                      make_structured_image, make_test_image, region_mean_diff, save_output, TEST_SIZE)

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
    assert 12345 in info.all_seeds


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
    body.denoising_strength = 0.75  # enough to visibly transform the textured source
    source = make_structured_image(TEST_SIZE, TEST_SIZE)
    response = service.img2img(source, request_body=body)
    assert len(response['images']) >= 1
    result = response['images'][0]
    _assert_valid_image(result)
    save_output(output_dir, 'img2img_source', source)
    save_output(output_dir, 'img2img_result', result)
    assert images_differ(source, result.resize(source.size)) >= 10.0, \
        'img2img returned an image nearly identical to its source'


def test_img2img_inpaint_respects_mask(service, output_dir):
    """Inpainting must change the masked region while preserving the rest.

    A1111's mask polarity is the intuitive one (verified empirically): the white mask
    region is inpainted and the black region is preserved (composited back pixel-for-pixel
    with inpainting_mask_invert=0). Uses a textured source and full denoising so the change
    is unmistakable, and asserts region-aware behavior rather than just "an image came back".
    """
    body = fast_request_body('a bright flower')
    body.denoising_strength = 1.0
    body.inpainting_fill = 1  # original
    body.inpaint_full_res = False
    body.inpainting_mask_invert = 0
    body.mask_blur = 0  # crisp region boundary so per-region diffs are clean
    source = make_structured_image(TEST_SIZE, TEST_SIZE)
    mask = make_mask()  # white center (edited) on black (preserved)
    response = service.img2img(source, mask=mask, request_body=body)
    assert len(response['images']) >= 1
    result = response['images'][0].resize(source.size)
    _assert_valid_image(response['images'][0])
    save_output(output_dir, 'inpaint_source', source)
    save_output(output_dir, 'inpaint_mask', mask)
    save_output(output_dir, 'inpaint_result', response['images'][0])

    center_box = (TEST_SIZE // 4, TEST_SIZE // 4, TEST_SIZE * 3 // 4, TEST_SIZE * 3 // 4)
    border_box = (0, 0, TEST_SIZE, TEST_SIZE // 8)
    edited_diff = region_mean_diff(source, result, center_box)
    preserved_diff = region_mean_diff(source, result, border_box)
    assert edited_diff >= 20.0, f'masked region barely changed (diff {edited_diff:.2f}); inpainting may be a no-op'
    assert preserved_diff <= 5.0, (
        f'inpainting did not respect the mask: the preserved region changed (diff {preserved_diff:.2f}). '
        f'Inspect inpaint_source.png / inpaint_mask.png / inpaint_result.png in {output_dir}.'
    )


def test_upscale_basic(service, output_dir):
    # Uses the plain "extra-single-image" upscaler path (no SD upscaling / ControlNet),
    # driven by the isolated fast cache profile.
    configure_fast_cache()
    source = make_structured_image(128, 128)
    response = service.upscale(source, 256, 256)
    assert len(response['images']) == 1
    result = response['images'][0]
    assert isinstance(result, Image.Image)
    assert result.width >= 200 and result.height >= 200
    save_output(output_dir, 'upscale_source', source)
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
