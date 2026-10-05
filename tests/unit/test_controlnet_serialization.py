"""Unit tests for ControlNet unit/preprocessor serialization round-trips.

These pin the serialize/deserialize contract that the refactor relies on: it plans to pass
``list[ControlNetUnit]`` directly instead of stashing serialized JSON in numbered cache slots,
but the units themselves still round-trip through this code. Pure logic, no server.
"""
from PIL import Image

from sd_backend_client.api.shared_data.controlnet.controlnet_model import ControlNetModel
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import (
    ControlNetPreprocessor, ParameterDef, PreprocessorParams)
from sd_backend_client.api.shared_data.controlnet.controlnet_unit import ControlNetUnit
from sd_backend_client.api.webui.controlnet_webui_constants import ControlNetUnitDict


def _make_unit() -> ControlNetUnit:
    unit = ControlNetUnit()
    unit.image = Image.new('RGBA', (1, 1), 'white')
    unit.model = ControlNetModel('control_v11p_sd15_canny [d14c016b]')
    unit.control_strength = 0.75
    unit.control_start = 0.1
    unit.control_end = 0.9
    unit.pixel_perfect = False
    unit.low_vram = True
    return unit


def _assert_units_equal(a: ControlNetUnit, b: ControlNetUnit) -> None:
    assert a.image == b.image
    assert a.model is not None and b.model is not None
    assert a.model.full_model_name == b.model.full_model_name
    assert float(a.control_strength) == float(b.control_strength)
    assert float(a.control_start) == float(b.control_start)
    assert float(a.control_end) == float(b.control_end)
    assert a.pixel_perfect == b.pixel_perfect
    assert a.low_vram == b.low_vram

def test_round_trip():
    original = _make_unit()
    restored = ControlNetUnit.model_validate(original.model_dump())
    _assert_units_equal(original, restored)


def test_from_unit_maps_core_fields_to_webui_keys():
    unit = _make_unit()
    unit_dict = ControlNetUnitDict.from_unit(unit)
    assert unit_dict.model == 'control_v11p_sd15_canny [d14c016b]'
    # Internal names map onto WebUI wire names:
    assert unit_dict.weight == 0.75
    assert unit_dict.guidance_start == 0.1
    assert unit_dict.guidance_end == 0.9
    assert unit_dict.pixel_perfect is False
    assert unit_dict.low_vram is True


def test_from_unit_copies_preprocessor_params_that_match_wire_fields():
    typedef = ControlNetPreprocessor(name='canny', parameters=[
        ParameterDef(key='processor_res', default_value=512, required=True),
        ParameterDef(key='threshold_a', default_value=100.0, required=True),
    ])
    unit = ControlNetUnit(
        model=ControlNetModel('control_v11p_sd15_canny [d14c016b]'),
        preprocessor=PreprocessorParams(typedef=typedef, parameter_values={'processor_res': 768}),
    )
    unit_dict = ControlNetUnitDict.from_unit(unit)
    assert unit_dict.module == 'canny'
    assert unit_dict.processor_res == 768          # explicit value carried through
    assert unit_dict.threshold_a == 100.0          # required default filled then copied


def test_from_unit_handles_missing_preprocessor_and_model():
    unit_dict = ControlNetUnitDict.from_unit(ControlNetUnit())
    assert unit_dict.module == 'None'
    assert unit_dict.model == 'None'
