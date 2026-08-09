"""Unit tests for ControlNet unit/preprocessor serialization round-trips.

These pin the serialize/deserialize contract that the refactor relies on: it plans to pass
``list[ControlNetUnit]`` directly instead of stashing serialized JSON in numbered cache slots,
but the units themselves still round-trip through this code. Pure logic, no server.
"""
from PIL import Image

from intrapaint_api.api.shared_data.controlnet.controlnet_model import ControlNetModel
from intrapaint_api.api.shared_data.controlnet.controlnet_unit import ControlNetUnit


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
