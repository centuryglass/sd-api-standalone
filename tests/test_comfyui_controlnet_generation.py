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

from intrapaint_api.api.comfyui.comfyui_diffusion_params import ComfyUIDiffusionParams
from intrapaint_api.api.shared_data.controlnet.controlnet_model import ControlNetModel
from intrapaint_api.api.shared_data.controlnet.controlnet_unit import ControlNetUnit
from .comfy_helpers import find_canny_model, find_canny_preprocessor, wait_for_comfy_images, build_comfy_params
from .helpers import images_differ, make_edge_image, save_output

pytestmark = [pytest.mark.integration, pytest.mark.controlnet, pytest.mark.generation]

MIN_MEAN_DIFF = 2.0


@pytest.fixture(scope='module')
def canny(comfy_service) -> ControlNetUnit:
    """A (preprocessor, model) Canny pairing, skipping if either isn't installed."""
    preprocessor = find_canny_preprocessor(comfy_service)
    model = find_canny_model(comfy_service)
    if preprocessor is None:
        pytest.skip('No Canny preprocessor node installed (comfyui_controlnet_aux?).')
    if model is None:
        pytest.skip('No Canny ControlNet model installed.')
    return ControlNetUnit(model=ControlNetModel(model), preprocessor=preprocessor)


def test_controlnet_preprocessor_preview(comfy_service, canny, output_dir):
    """The preprocessor-preview workflow should queue and return a processed map."""
    controlnet_unit = canny
    source = make_edge_image()
    save_output(output_dir, 'comfy_controlnet_input', source)
    response = comfy_service.controlnet_preprocessor_preview(source, source, controlnet_unit.preprocessor.typedef)
    images = wait_for_comfy_images(comfy_service, response)
    assert len(images) >= 1
    preview = images[0]
    assert isinstance(preview, Image.Image)
    save_output(output_dir, f'comfy_controlnet_preview_{controlnet_unit.preprocessor.typedef.name}', preview)
    extrema = preview.convert('L').getextrema()
    assert extrema[0] != extrema[1], 'preprocessor preview is a flat image — preprocessing did nothing'


def test_controlnet_alters_output(comfy_service, comfy_checkpoint, canny, output_dir, tmp_path):
    """Same seed, with vs. without ControlNet, must produce visibly different images."""
    params = build_comfy_params(comfy_checkpoint)

    controlnet_unit = canny
    control_image = make_edge_image()
    control_path = os.path.join(tmp_path, 'comfy_control.png')
    control_image.save(control_path)
    save_output(output_dir, 'comfy_controlnet_input', control_image)

    params.init_images = [control_image]
    params.sd_model_name = comfy_checkpoint


    params.prompt = 'a photograph of a city street, detailed'
    params.seed = 42

    # Baseline: no ControlNet.
    baseline = wait_for_comfy_images(comfy_service,
                                     comfy_service.txt2img(params))[0]
    save_output(output_dir, 'comfy_controlnet_baseline_no_cn', baseline)

    # Controlled: same seed, Canny unit enabled.
    controlnet_unit.image = control_image
    params.controlnet_units = [controlnet_unit]
    controlled = wait_for_comfy_images(comfy_service,
                                       comfy_service.txt2img(params))[0]

    preprocessor = controlnet_unit.preprocessor.typedef.name
    model = controlnet_unit.model.full_model_name
    save_output(output_dir, f'comfy_controlnet_with_{preprocessor}', controlled)

    mean_diff = images_differ(baseline, controlled)
    assert mean_diff >= MIN_MEAN_DIFF, (
        f'ControlNet ({preprocessor} / {model}) appears to have silently done nothing: '
        f'mean pixel diff {mean_diff:.3f} < {MIN_MEAN_DIFF} vs. the no-ControlNet baseline. '
        f'Compare comfy_controlnet_baseline_no_cn.png and comfy_controlnet_with_{preprocessor}.png '
        f'in {output_dir}.'
    )
