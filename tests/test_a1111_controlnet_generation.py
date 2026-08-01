"""ControlNet generation integration tests.

ControlNet is notorious for *silently* doing nothing when a request is subtly
mis-wired (wrong module/model, bad payload shape, etc.) — the request still returns
200 with a perfectly normal image. To catch that, ``test_controlnet_alters_output``
runs the same seed + prompt twice, with and without a control unit, and asserts the
results actually differ. All images (control input, preprocessor preview, baseline,
controlled) are saved under the output dir for visual fidelity inspection.

Opt-in (real diffusion): run with ``--run-generation`` / ``RUN_SD_GENERATION=1``.
"""
import pytest
from PIL import Image

from .helpers import (add_controlnet_unit, controlnet_unit_dict, fast_request_body,
                      images_differ, make_edge_image, save_output)

pytestmark = [pytest.mark.integration, pytest.mark.controlnet, pytest.mark.generation]

# Same-seed txt2img with vs. without a working control unit should differ by at least
# this mean per-channel value (0-255). Comfortably above noise, well below a real effect.
MIN_MEAN_DIFF = 2.0


def test_controlnet_preprocessor_preview(controlnet_available, controlnet_pairing, output_dir):
    """The /controlnet/detect preview should return a processed (non-empty) map."""
    module, _model = controlnet_pairing
    preprocessor = next(
        (p for p in controlnet_available.get_controlnet_preprocessors() if p.name == module),
        None,
    )
    if preprocessor is None:
        pytest.skip(f'Preprocessor {module!r} not found among available modules.')
    source = make_edge_image()
    save_output(output_dir, 'controlnet_input', source)
    preview = controlnet_available.controlnet_preprocessor_preview(source, None, preprocessor)
    assert isinstance(preview, Image.Image)
    save_output(output_dir, f'controlnet_preview_{module}', preview)
    # A real edge map of a high-contrast image is not a single flat color.
    assert preview.convert('L').getextrema()[0] != preview.convert('L').getextrema()[1], \
        'preprocessor preview is a flat image — preprocessing likely did nothing'


def test_controlnet_alters_output(service, controlnet_available, controlnet_pairing, output_dir):
    """Same seed, with vs. without ControlNet, must produce visibly different images."""
    module, model = controlnet_pairing
    control_image = make_edge_image()
    save_output(output_dir, 'controlnet_input', control_image)

    prompt = 'a photograph of a city street, detailed'
    seed = 42

    baseline_body = fast_request_body(prompt)
    baseline_body.seed = seed
    baseline = service.txt2img(baseline_body)['images'][0]
    save_output(output_dir, 'controlnet_baseline_no_cn', baseline)

    controlled_body = fast_request_body(prompt)
    controlled_body.seed = seed
    add_controlnet_unit(controlled_body, controlnet_unit_dict(module, model, control_image))
    controlled = service.txt2img(controlled_body)['images'][0]
    save_output(output_dir, f'controlnet_with_{module}', controlled)

    mean_diff = images_differ(baseline, controlled)
    assert mean_diff >= MIN_MEAN_DIFF, (
        f'ControlNet ({module} / {model}) appears to have silently done nothing: '
        f'mean pixel diff {mean_diff:.3f} < {MIN_MEAN_DIFF} vs. the no-ControlNet baseline. '
        f'Compare controlnet_baseline_no_cn.png and controlnet_with_{module}.png in {output_dir}.'
    )
