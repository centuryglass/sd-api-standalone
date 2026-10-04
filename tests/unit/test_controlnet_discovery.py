"""Offline tests for ControlNet option discovery: preprocessor parsing on both backends, and control type categories.

The inputs are hand-written in the shapes ComfyUI's /object_info and the WebUI's /controlnet/module_list return.
"""
from typing import Any

import pytest

from intrapaint_api.api.comfyui.comfyui_types import NodeInfoResponse
from intrapaint_api.api.comfyui.controlnet_comfyui_utils import get_all_preprocessors as comfy_preprocessors
from intrapaint_api.api.shared_data.controlnet.controlnet_category_builder import ControlNetCategoryBuilder
from intrapaint_api.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor
from intrapaint_api.api.webui.controlnet_webui_constants import ModuleDetail
from intrapaint_api.api.webui.controlnet_webui_utils import get_all_preprocessors as webui_preprocessors

PREPROCESSOR_CATEGORY = 'ControlNet Preprocessors/Line Extractors'


def _node(name: str, required: dict[str, Any], optional: dict[str, Any] | None = None,
          category: str = PREPROCESSOR_CATEGORY, display_name: str | None = None) -> NodeInfoResponse:
    """A NodeInfoResponse in /object_info form, with input_order following the dicts' order."""
    input_types: dict[str, Any] = {'required': required}
    input_order = {'required': list(required)}
    if optional is not None:
        input_types['optional'] = optional
        input_order['optional'] = list(optional)
    return NodeInfoResponse.model_validate({
        'input': input_types, 'input_order': input_order, 'output': ['IMAGE'], 'output_is_list': [False],
        'output_name': ['IMAGE'], 'name': name, 'display_name': display_name, 'description': None,
        'python_module': 'custom_nodes.comfyui_controlnet_aux', 'category': category, 'output_node': False})


def _by_name(preprocessors: list[ControlNetPreprocessor]) -> dict[str, ControlNetPreprocessor]:
    return {preprocessor.name: preprocessor for preprocessor in preprocessors}


# ComfyUI

def test_comfyui_parses_each_parameter_type():
    """INT, FLOAT, BOOLEAN, STRING and combo inputs become ParameterDefs with defaults and ranges."""
    node = _node('AnyLinePreprocessor', {
        'image': ['IMAGE'],
        'resolution': ['INT', {'default': 512, 'min': 64, 'max': 16384, 'step': 64, 'tooltip': 'Output size'}],
        'strength': ['FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01}],
        'safe': ['BOOLEAN', {'default': True}],
        'merge_with_lineart': [['lineart_standard', 'lineart_realistic'], {}],
    }, optional={'mask': ['MASK'], 'label': ['STRING', {'multiline': False}]}, display_name='AnyLine Lineart')
    preprocessor = comfy_preprocessors({'AnyLinePreprocessor': node})[0]

    params = {param.key: param for param in preprocessor.parameters}
    assert list(params) == ['resolution', 'strength', 'safe', 'merge_with_lineart', 'label']
    resolution = params['resolution']
    assert (resolution.default_value, resolution.min_val, resolution.max_val, resolution.step_val) == \
        (512, 64, 16384, 64)
    assert resolution.description == 'Output size'
    assert (params['strength'].default_value, params['strength'].step_val) == (0.5, 0.01)
    assert params['safe'].default_value is True
    assert params['merge_with_lineart'].default_value == 'lineart_standard'
    assert params['label'].default_value == ''
    assert params['label'].required is False and resolution.required is True

    assert preprocessor.has_image_input and preprocessor.has_mask_input
    assert preprocessor.description == 'AnyLine Lineart'
    assert preprocessor.category_name == PREPROCESSOR_CATEGORY


@pytest.mark.xfail(strict=True, reason='https://github.com/centuryglass/sd-api-standalone/issues/34')
def test_comfyui_combo_inputs_keep_their_options():
    """A combo input's choices become the parameter's option_list."""
    node = _node('ModePreprocessor', {'image': ['IMAGE'], 'mode': [['a', 'b'], {}]})
    assert comfy_preprocessors({'ModePreprocessor': node})[0].parameters[0].option_list == ['a', 'b']


def test_comfyui_filters_non_preprocessor_and_unusable_nodes():
    """Only usable nodes in the preprocessor category are kept."""
    nodes = {
        'CannyEdgePreprocessor': _node('CannyEdgePreprocessor', {'image': ['IMAGE']}),
        'KSampler': _node('KSampler', {'model': ['MODEL']}, category='sampling'),
        'ExecuteAllControlNetPreprocessors': _node('ExecuteAllControlNetPreprocessors', {'image': ['IMAGE']}),
        'OddInputPreprocessor': _node('OddInputPreprocessor', {'image': ['IMAGE'], 'model': ['DEPTH_MODEL']}),
    }
    preprocessors = _by_name(comfy_preprocessors(nodes))
    assert list(preprocessors) == ['CannyEdgePreprocessor']
    assert preprocessors['CannyEdgePreprocessor'].description == 'CannyEdge'
    assert preprocessors['CannyEdgePreprocessor'].has_mask_input is False


def test_comfyui_ignores_unhandled_optional_input():
    """An optional input of an unknown type is dropped without discarding the node."""
    node = _node('DepthPreprocessor', {'image': ['IMAGE']}, optional={'model': ['DEPTH_MODEL']})
    preprocessors = comfy_preprocessors({'DepthPreprocessor': node})
    assert [preprocessor.name for preprocessor in preprocessors] == ['DepthPreprocessor']
    assert not preprocessors[0].parameters


# WebUI

def test_webui_preset_parameters_without_module_details():
    """Without API details, parameters come from the preset tables in controlnet_webui_constants."""
    preprocessors = _by_name(webui_preprocessors(['canny', 'inpaint_only', 'depth_marigold', 'reference_only']))

    canny = preprocessors['canny']
    assert [param.key for param in canny.parameters] == \
        ['control_mode', 'resize_mode', 'processor_res', 'threshold_a', 'threshold_b']
    assert canny.parameters[3].description == 'Low Threshold'
    assert canny.model_free is False
    assert [param.key for param in preprocessors['inpaint_only'].parameters] == ['control_mode']
    assert preprocessors['depth_marigold'].parameters[2].default_value == 768
    assert preprocessors['reference_only'].model_free is True


def test_webui_module_details_map_resolution_and_thresholds():
    """API sliders map to processor_res by name and to threshold_a/threshold_b by order."""
    details = {'canny': ModuleDetail.model_validate({'model_free': False, 'sliders': [
        {'name': 'Resolution', 'min': 64, 'max': 2048, 'default': 512, 'step': 8},
        {'name': 'Low Threshold', 'min': 1, 'max': 255, 'default': 100, 'step': 1},
        {'name': 'High Threshold', 'min': 1, 'max': 255, 'default': 200, 'step': 1},
    ]})}
    canny = webui_preprocessors(['canny'], details)[0]

    params = {param.key: param for param in canny.parameters}
    assert list(params) == ['control_mode', 'resize_mode', 'processor_res', 'threshold_a', 'threshold_b']
    assert (params['processor_res'].min_val, params['processor_res'].max_val) == (64, 2048)
    assert params['threshold_a'].description == 'Low Threshold'
    assert params['threshold_b'].default_value == 200


def test_webui_module_details_reject_a_third_threshold():
    """More than two non-resolution sliders is an error."""
    slider = {'name': 'Value', 'min': 0, 'max': 1, 'default': 0, 'step': 1}
    details = {'odd': ModuleDetail.model_validate({'model_free': True, 'sliders': [slider] * 3})}
    with pytest.raises(RuntimeError, match='odd'):
        webui_preprocessors(['odd'], details)


# Control type categories

def test_category_builder_keeps_types_with_available_options():
    """Types keep listed and pattern-matched options that exist, and types with nothing available are dropped."""
    builder = ControlNetCategoryBuilder(['canny', 'CannyEdgePreprocessor', 'depth_midas'],
                                        ['control_v11p_sd15_canny [d14c016b]', 'my_canny_model'])
    control_types = builder.get_control_types()

    canny = control_types['Canny']
    assert canny['module_list'] == ['None', 'canny', 'CannyEdgePreprocessor']
    assert canny['model_list'] == ['None', 'control_v11p_sd15_canny [d14c016b]', 'my_canny_model']
    assert canny['default_option'] == 'canny'
    assert canny['default_model'] == 'control_v11p_sd15_canny [d14c016b]'
    assert 'Depth' not in control_types


def test_category_builder_matches_preprocessors_by_comfyui_category():
    """A ComfyUI preprocessor whose name doesn't match a type's pattern is matched through its category."""
    builder = ControlNetCategoryBuilder(['Zoe-DepthMapPreprocessor', 'MiDaS-NormalMapPreprocessor'],
                                        ['control_depth.safetensors'],
                                        {'MiDaS-NormalMapPreprocessor': 'ControlNet Preprocessors/Depth'})
    assert builder.get_control_types()['Depth']['module_list'] == \
        ['None', 'Zoe-DepthMapPreprocessor', 'MiDaS-NormalMapPreprocessor']


def test_category_builder_prefers_api_type_definitions():
    """API control type definitions are applied before the static ones."""
    api_types = {'control_types': {'Canny': {'module_list': ['none', 'canny'], 'model_list': ['None', 'api_model'],
                                             'default_option': 'canny', 'default_model': 'api_model'}}}
    builder = ControlNetCategoryBuilder(['canny'], ['api_model', 'control_v11p_sd15_canny [d14c016b]'],
                                        None, api_types)  # type: ignore[arg-type]
    assert builder.get_control_types()['Canny']['default_model'] == 'api_model'
