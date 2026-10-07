"""Unit tests for the parameter models' field constraints, which reject bad values at construction and assignment."""
import pytest
from pydantic import ValidationError

from sd_backend_client.api.comfyui.comfyui_diffusion_params import ComfyUIDiffusionParams
from sd_backend_client.api.shared_data.api_datatypes import DiffusionUpscalingParams
from sd_backend_client.api.shared_data.controlnet.controlnet_model import ControlNetModel
from sd_backend_client.api.shared_data.controlnet.controlnet_unit import ControlNetUnit
from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams
from sd_backend_client.api.webui.diffusion_request_body import DiffusionRequestBody

MODEL_NAME = 'control_v11p_sd15_canny [d14c016b]'


@pytest.mark.parametrize('params_class', [DiffusionParams, DiffusionRequestBody, ComfyUIDiffusionParams])
@pytest.mark.parametrize('field, value', [
    ('steps', 0),
    ('batch_size', 0),
    ('width', 0),
    ('height', -8),
    ('cfg_scale', -0.5),
    ('denoising_strength', -0.1),
    ('denoising_strength', 1.1),
])
def test_diffusion_params_reject_out_of_range_values(params_class, field, value):
    """Each bounded field rejects an out-of-range value at construction and on assignment."""
    with pytest.raises(ValidationError, match=field):
        params_class(**{field: value})
    params = params_class()
    with pytest.raises(ValidationError, match=field):
        setattr(params, field, value)


def test_diffusion_params_accept_boundary_values():
    """Values at each bound are accepted, and denoising_strength can be reset to None."""
    params = DiffusionParams(steps=1, batch_size=1, width=1, height=1, cfg_scale=0.0, denoising_strength=0.0)
    params.denoising_strength = 1.0
    params.denoising_strength = None
    assert params.denoising_strength is None


def test_diffusion_params_accept_sizes_that_are_not_multiples_of_8():
    """Sizes off the 8-pixel grid are left for the backend to round."""
    params = DiffusionParams(width=500, height=333)
    assert (params.width, params.height) == (500, 333)


def test_webui_n_iter_must_be_positive():
    """WebUI batch count must be at least 1."""
    with pytest.raises(ValidationError, match='n_iter'):
        DiffusionRequestBody(n_iter=0)


@pytest.mark.parametrize('field, value', [
    ('control_strength', -0.1),
    ('control_strength', 2.1),
    ('control_start', -0.1),
    ('control_end', 1.1),
])
def test_controlnet_unit_rejects_out_of_range_values(field, value):
    """Strength is limited to 0.0-2.0 and start/end to 0.0-1.0."""
    with pytest.raises(ValidationError, match=field):
        ControlNetUnit(**{field: value})
    unit = ControlNetUnit()
    with pytest.raises(ValidationError, match=field):
        setattr(unit, field, value)


def test_controlnet_unit_rejects_start_after_end():
    """control_start may not exceed control_end, at construction or after either is assigned."""
    with pytest.raises(ValidationError, match='control_start'):
        ControlNetUnit(control_start=0.8, control_end=0.2)
    unit = ControlNetUnit(control_start=0.2, control_end=0.5)
    with pytest.raises(ValidationError, match='control_start'):
        unit.control_start = 0.6
    with pytest.raises(ValidationError, match='control_start'):
        unit.control_end = 0.1


def test_controlnet_unit_accepts_equal_start_and_end():
    """A zero-length control range is allowed."""
    unit = ControlNetUnit(control_start=0.5, control_end=0.5)
    assert unit.control_start == unit.control_end


@pytest.mark.parametrize('field, value', [
    ('denoising_strength', 1.5),
    ('step_count', 0),
    ('tile_width', 0),
    ('tile_height', 0),
    ('mask_blur', -1),
    ('tile_padding', -1),
    ('seam_fix_denoise', -0.1),
    ('seam_fix_width', -1),
    ('seam_fix_mask_blur', -1),
    ('seam_fix_padding', -1),
])
def test_upscaling_params_reject_out_of_range_values(field, value):
    """Upscaling tile, blur, padding and denoise values are bounded."""
    with pytest.raises(ValidationError, match=field):
        DiffusionUpscalingParams(**{field: value})
    params = DiffusionUpscalingParams()
    with pytest.raises(ValidationError, match=field):
        setattr(params, field, value)


def test_upscaling_params_validate_nested_diffusion_params():
    """The nested diffusion_params carry their own constraints."""
    with pytest.raises(ValidationError, match='steps'):
        DiffusionUpscalingParams(diffusion_params={'steps': 0})


def test_controlnet_model_is_hashable_and_compares_by_name():
    """Models with the same full name are equal and hash the same."""
    assert ControlNetModel(MODEL_NAME) == ControlNetModel(MODEL_NAME)
    assert len({ControlNetModel(MODEL_NAME), ControlNetModel(MODEL_NAME)}) == 1


def test_controlnet_model_field_accepts_full_model_name():
    """A full model name string validates into a ControlNetModel."""
    unit = ControlNetUnit(model=MODEL_NAME)
    assert isinstance(unit.model, ControlNetModel)
    assert unit.model.full_model_name == MODEL_NAME


def test_controlnet_model_field_rejects_other_types():
    """Values that are neither a model nor a name are rejected."""
    with pytest.raises(ValidationError, match='model'):
        ControlNetUnit(model=42)


def test_controlnet_unit_round_trips_through_json():
    """A unit without an image survives model_dump_json and model_validate_json, with the model as its name."""
    unit = ControlNetUnit(model=ControlNetModel(MODEL_NAME), control_strength=1.5, control_start=0.25,
                          control_end=0.75, low_vram=True)
    json_data = unit.model_dump_json()
    assert f'"model":"{MODEL_NAME}"' in json_data
    assert ControlNetUnit.model_validate_json(json_data) == unit


def test_controlnet_unit_python_dump_keeps_model_instance():
    """Python-mode dumps keep the ControlNetModel instance; only JSON dumps convert it to its name."""
    model = ControlNetModel(MODEL_NAME)
    assert ControlNetUnit(model=model).model_dump()['model'] is model
