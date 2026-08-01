"""ControlNet generation integration tests for the ComfyUI backend.

Like the A1111 suite, this catches ControlNet *silently* doing nothing: the same seed
and prompt are generated with and without a control unit, and the results must differ.
ComfyUI drives ControlNet through the cache (``CONTROLNET_ARGS_*_COMFYUI``) rather than
a request-body script, so the unit is installed via ``comfy_helpers``. All images are
saved under the output dir for visual fidelity inspection.

Opt-in (real diffusion): run with ``--run-generation`` / ``RUN_SD_GENERATION=1``.
"""
import os

import pytest
from PIL import Image

from .comfy_helpers import (clear_comfy_controlnet_cache, configure_comfy_cache, find_canny_model,
                            find_canny_preprocessor, set_comfy_controlnet_unit, wait_for_comfy_images)
from .helpers import images_differ, make_edge_image, save_output

pytestmark = [pytest.mark.integration, pytest.mark.controlnet, pytest.mark.generation]

MIN_MEAN_DIFF = 2.0


@pytest.fixture(scope='module')
def canny(comfy_service):
    """A (preprocessor, model) Canny pairing, skipping if either isn't installed."""
    preprocessor = find_canny_preprocessor(comfy_service)
    model = find_canny_model(comfy_service)
    if preprocessor is None:
        pytest.skip('No Canny preprocessor node installed (comfyui_controlnet_aux?).')
    if model is None:
        pytest.skip('No Canny ControlNet model installed.')
    return preprocessor, model


def test_controlnet_preprocessor_preview(comfy_service, canny, output_dir):
    """The preprocessor-preview workflow should queue and return a processed map."""
    preprocessor, _model = canny
    source = make_edge_image()
    save_output(output_dir, 'comfy_controlnet_input', source)
    response = comfy_service.controlnet_preprocessor_preview(source, source, preprocessor)
    images = wait_for_comfy_images(comfy_service, response)
    assert len(images) >= 1
    preview = images[0]
    assert isinstance(preview, Image.Image)
    save_output(output_dir, f'comfy_controlnet_preview_{preprocessor.name}', preview)
    extrema = preview.convert('L').getextrema()
    assert extrema[0] != extrema[1], 'preprocessor preview is a flat image — preprocessing did nothing'


def test_controlnet_alters_output(comfy_service, comfy_checkpoint, canny, output_dir, tmp_path):
    """Same seed, with vs. without ControlNet, must produce visibly different images."""
    preprocessor, model = canny
    control_image = make_edge_image()
    control_path = os.path.join(tmp_path, 'comfy_control.png')
    control_image.save(control_path)
    save_output(output_dir, 'comfy_controlnet_input', control_image)

    prompt = 'a photograph of a city street, detailed'
    seed = 42

    # Baseline: no ControlNet.
    configure_comfy_cache(comfy_checkpoint, prompt=prompt, edit_mode='Text to Image')
    clear_comfy_controlnet_cache()
    baseline = wait_for_comfy_images(comfy_service, comfy_service.txt2img(control_image, {}, seed=seed))[0]
    save_output(output_dir, 'comfy_controlnet_baseline_no_cn', baseline)

    # Controlled: same seed, Canny unit enabled.
    configure_comfy_cache(comfy_checkpoint, prompt=prompt, edit_mode='Text to Image')
    set_comfy_controlnet_unit(preprocessor, model, control_path)
    controlled = wait_for_comfy_images(comfy_service, comfy_service.txt2img(control_image, {}, seed=seed))[0]
    save_output(output_dir, f'comfy_controlnet_with_{preprocessor.name}', controlled)

    mean_diff = images_differ(baseline, controlled)
    assert mean_diff >= MIN_MEAN_DIFF, (
        f'ControlNet ({preprocessor.name} / {model}) appears to have silently done nothing: '
        f'mean pixel diff {mean_diff:.3f} < {MIN_MEAN_DIFF} vs. the no-ControlNet baseline. '
        f'Compare comfy_controlnet_baseline_no_cn.png and comfy_controlnet_with_{preprocessor.name}.png '
        f'in {output_dir}.'
    )
