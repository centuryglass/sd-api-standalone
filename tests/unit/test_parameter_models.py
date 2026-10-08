"""Request parameter models reject unknown fields, and subclass conversions pass only shared fields."""
import pytest
from pydantic import ValidationError

from sd_backend_client import ControlNetUnit, DiffusionParams, DiffusionUpscalingParams
from sd_backend_client.api.comfyui.comfyui_diffusion_params import ComfyUIDiffusionParams
from sd_backend_client.api.webui.diffusion_request_body import DiffusionRequestBody


def test_parameter_models_reject_unknown_fields():
    """A misspelled field raises instead of silently running with defaults."""
    with pytest.raises(ValidationError):
        DiffusionParams(promt='a fox')
    with pytest.raises(ValidationError):
        ControlNetUnit(strength=0.3)
    with pytest.raises(ValidationError):
        DiffusionUpscalingParams(bogus=1)


def test_webui_body_converts_to_comfyui_params_via_shared_fields():
    """Only fields the target model defines are passed between backend subclasses."""
    body = DiffusionRequestBody(prompt='fox', s_noise=0.5)
    shared = {name: getattr(body, name) for name in DiffusionParams.model_fields.keys()}
    assert ComfyUIDiffusionParams(**shared).prompt == 'fox'
    assert DiffusionRequestBody.from_params(ComfyUIDiffusionParams(prompt='cat', clip_skip=2)).prompt == 'cat'
