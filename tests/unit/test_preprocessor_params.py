"""Unit tests for PreprocessorParams construction and validation.

These pin the behavior of the ControlNet preprocessor value container that the refactor centers on:
defaults get filled in from the typedef, and provided values are validated for type / range / options.
Pure logic, no server.
"""
import pytest

from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import (
    ControlNetPreprocessor, ParameterDef, PreprocessorParams)


def _preprocessor(*params: ParameterDef) -> ControlNetPreprocessor:
    return ControlNetPreprocessor(name='canny', parameters=list(params))


def test_required_defaults_are_filled_from_typedef():
    typedef = _preprocessor(
        ParameterDef(key='low', default_value=100, required=True),
        ParameterDef(key='high', default_value=200, required=True),
    )
    params = PreprocessorParams(typedef=typedef)
    assert params.parameter_values == {'low': 100, 'high': 200}


def test_optional_params_are_not_auto_filled():
    # Optional params must stay absent unless explicitly set, so a backend can treat them as undefined.
    typedef = _preprocessor(
        ParameterDef(key='required_one', default_value=1, required=True),
        ParameterDef(key='optional_path', default_value='', required=False),
    )
    params = PreprocessorParams(typedef=typedef)
    assert params.parameter_values == {'required_one': 1}
    assert 'optional_path' not in params.parameter_values


def test_falsy_required_defaults_are_still_filled():
    # 0 / '' / False are valid defaults and must populate parameter_values, not be treated as "unset".
    typedef = _preprocessor(
        ParameterDef(key='count', default_value=0, required=True),
        ParameterDef(key='label', default_value='', required=True),
        ParameterDef(key='flag', default_value=False, required=True),
    )
    params = PreprocessorParams(typedef=typedef)
    assert params.parameter_values == {'count': 0, 'label': '', 'flag': False}


def test_parameter_def_requires_a_default():
    with pytest.raises(ValueError):
        ParameterDef(key='low')  # default_value is mandatory


def test_explicit_value_overrides_default():
    typedef = _preprocessor(ParameterDef(key='low', default_value=100))
    params = PreprocessorParams(typedef=typedef, parameter_values={'low': 50})
    assert params.parameter_values['low'] == 50


def test_wrong_type_is_rejected():
    typedef = _preprocessor(ParameterDef(key='low', default_value=100))
    with pytest.raises(ValueError):
        PreprocessorParams(typedef=typedef, parameter_values={'low': 'not-an-int'})


def test_bool_and_int_are_not_interchangeable():
    typedef = _preprocessor(ParameterDef(key='flag', default_value=True))
    with pytest.raises(ValueError):
        PreprocessorParams(typedef=typedef, parameter_values={'flag': 1})


def test_int_value_accepted_for_float_default():
    typedef = _preprocessor(ParameterDef(key='ratio', default_value=1.0))
    params = PreprocessorParams(typedef=typedef, parameter_values={'ratio': 2})
    assert params.parameter_values['ratio'] == 2


def test_value_out_of_range_is_rejected():
    typedef = _preprocessor(ParameterDef(key='low', default_value=100, min_val=0, max_val=255))
    with pytest.raises(ValueError):
        PreprocessorParams(typedef=typedef, parameter_values={'low': 999})


def test_value_not_in_option_list_is_rejected():
    typedef = _preprocessor(
        ParameterDef(key='mode', default_value='a', option_list=['a', 'b', 'c']))
    with pytest.raises(ValueError):
        PreprocessorParams(typedef=typedef, parameter_values={'mode': 'z'})


def test_unexpected_key_is_rejected():
    typedef = _preprocessor(ParameterDef(key='low', default_value=100))
    with pytest.raises(ValueError):
        PreprocessorParams(typedef=typedef, parameter_values={'low': 100, 'bogus': 1})
