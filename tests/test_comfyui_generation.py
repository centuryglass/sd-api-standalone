"""Generation integration tests: real diffusion round-trips against ComfyUI.

ComfyUI jobs are asynchronous — each call queues a workflow and returns a ``prompt_id``
that we poll to completion, then download the images (see ``comfy_helpers``). Opt-in
(needs a checkpoint + GPU): run with ``--run-generation`` / ``RUN_SD_GENERATION=1``.
Results are saved under the output dir for visual fidelity inspection.
"""
import pytest
from PIL import Image


from .comfy_helpers import (COMFY_SIZE, build_comfy_params, make_comfy_mask, wait_for_comfy_images)
from .helpers import (images_differ, make_structured_image, make_test_image, region_mean_diff,
                      save_output)

pytestmark = [pytest.mark.integration, pytest.mark.generation]


def _assert_valid_image(image, expected_size: int = COMFY_SIZE):
    assert isinstance(image, Image.Image)
    assert image.mode == 'RGBA'  # decoded images are normalized to RGBA
    assert abs(image.width - expected_size) <= 8
    assert abs(image.height - expected_size) <= 8


def test_txt2img_produces_image(comfy_service, comfy_checkpoint, output_dir):
    params = build_comfy_params(comfy_checkpoint)
    response = comfy_service.txt2img(params)
    images = wait_for_comfy_images(comfy_service, response)
    assert len(images) == 1
    _assert_valid_image(images[0])
    save_output(output_dir, 'comfy_txt2img', images[0])


def test_txt2img_seed_is_reported(comfy_service, comfy_checkpoint):
    params = build_comfy_params(comfy_checkpoint)
    params.seed = 98765
    response = comfy_service.txt2img(params)
    # The client echoes back the seed it actually queued.
    assert response.seed == 98765
    wait_for_comfy_images(comfy_service, response)  # drain the job so it doesn't linger


def test_img2img_round_trip(comfy_service, comfy_checkpoint, output_dir):
    params = build_comfy_params(comfy_checkpoint, prompt='a vivid landscape painting')
    params.denoising_strength = 0.75  # enough to visibly transform the textured source
    source = make_structured_image(COMFY_SIZE, COMFY_SIZE)
    params.init_images = [source]
    params.seed = 7
    response = comfy_service.img2img(params)
    images = wait_for_comfy_images(comfy_service, response)
    assert len(images) >= 1
    result = images[0]
    _assert_valid_image(result)
    save_output(output_dir, 'comfy_img2img_source', source)
    save_output(output_dir, 'comfy_img2img_result', result)
    assert images_differ(source, result) >= 10.0, 'img2img returned an image nearly identical to its source'


def test_inpaint_respects_mask(comfy_service, comfy_checkpoint, output_dir):
    """Inpainting must change the masked (editable) region while preserving the rest.

    Uses a textured source and full denoising so the change is unmistakable, and a mask
    whose transparent center is the editable region (ComfyUI's inverted alpha polarity).
    The assertion is region-aware: the center should change far more than the border,
    which proves the mask is actually being followed rather than the whole image being
    regenerated (or nothing happening at all).
    """
    params = build_comfy_params(comfy_checkpoint, prompt='a photograph of a flower')
    params.denoising_strength = 1.0  # flat/low denoise barely changes inpainted pixels

    source = make_structured_image(COMFY_SIZE, COMFY_SIZE)
    center_box = (COMFY_SIZE // 4, COMFY_SIZE // 4, COMFY_SIZE * 3 // 4, COMFY_SIZE * 3 // 4)
    border_box = (0, 0, COMFY_SIZE, COMFY_SIZE // 8)
    mask = make_comfy_mask(center_box)
    params.init_images = [source]
    params.mask = mask
    params.seed = 7

    response = comfy_service.inpaint(params)
    images = wait_for_comfy_images(comfy_service, response)
    assert len(images) >= 1
    result = images[0]
    _assert_valid_image(result)
    save_output(output_dir, 'comfy_inpaint_source', source)
    save_output(output_dir, 'comfy_inpaint_mask', mask)
    save_output(output_dir, 'comfy_inpaint_result', result)

    edited_diff = region_mean_diff(source, result, center_box)
    preserved_diff = region_mean_diff(source, result, border_box)
    assert edited_diff >= 20.0, f'masked region barely changed (diff {edited_diff:.2f}); inpainting may be a no-op'
    assert edited_diff >= 3 * preserved_diff, (
        f'inpainting did not respect the mask: edited-region diff {edited_diff:.2f} is not clearly greater than '
        f'preserved-region diff {preserved_diff:.2f} (VAE round-trip noise floor). '
        f'Inspect comfy_inpaint_source.png / _mask.png / _result.png in {output_dir}.'
    )


def test_upscale(comfy_service, comfy_checkpoint, output_dir):
    from intrapaint_api.api.comfyui_webservice import ComfyModelType
    from intrapaint_api.config.cache import Cache

    upscale_models = comfy_service.get_models(ComfyModelType.UPSCALING)
    if not upscale_models:
        pytest.skip('No upscale models installed on the ComfyUI server.')
    params = build_comfy_params(comfy_checkpoint)
    # Basic (non-SD) upscaling path: needs a valid upscale model registered in the cache.
    Cache().set(Cache.GENERATOR_SCALING_MODES, upscale_models)
    Cache().set(Cache.SCALING_MODE, upscale_models[0], add_missing_options=True)
    Cache().set(Cache.USE_STABLE_DIFFUSION_UPSCALING, False)

    source = make_structured_image(128, 128)
    response = comfy_service.upscale(source, 256, 256)
    images = wait_for_comfy_images(comfy_service, response)
    assert len(images) >= 1
    result = images[0]
    assert isinstance(result, Image.Image)
    assert result.width >= 200 and result.height >= 200
    save_output(output_dir, 'comfy_upscale_source', source)
    save_output(output_dir, 'comfy_upscale_result', result)


def test_interrupt_is_accepted(comfy_service):
    # With no active job this is a no-op, but the endpoint should respond without raising.
    comfy_service.interrupt()
