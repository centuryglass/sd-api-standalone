"""Unit tests for ``ComfyUiWebservice._prepare_controlnet_data`` unit filtering. No network access."""
from unittest.mock import MagicMock

from intrapaint_api.api.comfyui_webservice import ComfyUiWebservice
from intrapaint_api.api.shared_data.controlnet.controlnet_model import ControlNetModel
from intrapaint_api.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor, PreprocessorParams
from intrapaint_api.api.shared_data.controlnet.controlnet_unit import ControlNetUnit


def _prepare(units: list[ControlNetUnit]) -> MagicMock:
    service = object.__new__(ComfyUiWebservice)  # skip __init__: no connection needed
    builder = MagicMock()
    service._prepare_controlnet_data(builder, units)  # pylint: disable=protected-access
    return builder


def test_model_free_preprocessor_without_model_is_skipped(caplog):
    """ComfyUI needs a control_net model on every unit, so model_free no longer lets a unit through."""
    typedef = ControlNetPreprocessor(name='Canny', model_free=True)
    unit = ControlNetUnit(preprocessor=PreprocessorParams(typedef=typedef))
    builder = _prepare([unit])
    builder.add_controlnet_unit.assert_not_called()
    assert 'Skipping unit 0' in caplog.text


def test_unit_with_model_is_added():
    """A unit with a model is passed to the workflow builder."""
    unit = ControlNetUnit(model=ControlNetModel('canny.safetensors'))
    builder = _prepare([unit])
    builder.add_controlnet_unit.assert_called_once()
    assert builder.add_controlnet_unit.call_args.args[0] == 'canny.safetensors'
