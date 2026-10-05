"""Helper functions for processing ControlNet configuration data for use with the ComfyUI API."""
import logging
from typing import Optional

from sd_backend_client.api.comfyui.comfyui_types import (NodeInfoResponse, CONTROLNET_PREPROCESSOR_CATEGORY,
                                                         IntParamDef, BoolParamDef, FloatParamDef, StrParamDef,
                                                         ParamDef)
from sd_backend_client.api.comfyui.nodes.controlnet.dynamic_preprocessor_node import DynamicPreprocessorNode
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor, ParameterDef

# If a preprocessor name ends in "Preprocessor", we can leave that part out of the display name.
PREPROCESSOR_SUFFIX = 'Preprocessor'

logger = logging.getLogger(__name__)

# These nodes are categorized under ControlNet preprocessors, and appear to have valid inputs and outputs, but
# shouldn't actually be used for various reasons:
INVALID_PREPROCESSOR_NODES = {
    # Selects between preprocessors, presumably for advanced routing purposes that IntraPaint doesn't support:
    'ControlNetPreprocessorSelector',
    # Seems to apply every possible preprocessor, presumably so you can see what all of them would generate:
    'ExecuteAllControlNetPreprocessors',
    # Looks like it's meant for extracting a size, not transforming an image:
    'ImageGenResolutionFromImage',
    # image output is OPTICAL_FLOW type, which we aren't able to handle:
    'Unimatch_OptFlowPreprocessor'
}


def get_all_preprocessors(node_data: dict[str, NodeInfoResponse]) -> list[ControlNetPreprocessor]:
    """Parses available node data to find the complete list of usable ControlNet preprocessiors."""
    none_preprocessor_found = False
    preprocessors: list[ControlNetPreprocessor] = []
    for node_info in node_data.values():
        category = node_info.category
        if CONTROLNET_PREPROCESSOR_CATEGORY not in category:
            continue
        node_name = node_info.name
        if node_name in INVALID_PREPROCESSOR_NODES:
            continue

        # read all parameters:
        parameter_list: list[ParameterDef] = []
        input_lists = node_info.input_order
        input_dicts = node_info.input
        has_image_input = False
        has_mask_input = False

        invalid_input_found = False
        for input_category in ('required', 'optional'):
            if invalid_input_found:
                break
            # input_dicts is an InputTypeDef model (attributes), not a dict; input_lists (input_order) is a dict.
            input_dict = getattr(input_dicts, input_category)  # .required / .optional
            if input_category not in input_lists or input_dict is None:
                continue
            inputs = input_lists[input_category]

            for input_name in inputs:
                if input_name == DynamicPreprocessorNode.IMAGE:
                    has_image_input = True
                    continue
                if input_name == DynamicPreprocessorNode.MASK:
                    has_mask_input = True
                    continue
                input_tuple = input_dict[input_name]
                input_type_or_list = input_tuple[0]
                # Some nodes put a bare value (not a param-def dict) in the second slot; only parse it when it's a dict.
                param_def_dict = input_tuple[1] if len(input_tuple) >= 2 and isinstance(input_tuple[1], dict) else None
                input_param_def = None if param_def_dict is None else ParamDef.model_validate(param_def_dict)

                # Find Parameter init values:
                key = input_name
                if input_param_def is not None and input_param_def.tooltip is not None:
                    param_description = input_param_def.tooltip
                else:
                    param_description = ''
                default_value: Optional[int | float | str]
                min_val: Optional[int | float] = None
                max_val: Optional[int | float] = None
                step_val: Optional[int | float] = None
                options: Optional[list[str | int | float | bool]] = None

                # Re-validate the raw param dict as the specific subclass; casting the base ParamDef wouldn't
                # populate default/min/max/step.
                if input_type_or_list == 'INT':
                    assert input_param_def is not None
                    int_param_def = IntParamDef.model_validate(input_tuple[1])
                    default_value = round(int_param_def.default)
                    min_val = None if int_param_def.min is None else round(int_param_def.min)
                    max_val = None if int_param_def.max is None else round(int_param_def.max)
                    if int_param_def.step is not None:
                        step_val = round(int_param_def.step)
                elif input_type_or_list == 'BOOLEAN':
                    assert input_param_def is not None
                    default_value = BoolParamDef.model_validate(input_tuple[1]).default
                elif input_type_or_list == 'FLOAT':
                    assert input_param_def is not None
                    float_param_def = FloatParamDef.model_validate(input_tuple[1])
                    default_value = float(float_param_def.default)
                    min_val = None if float_param_def.min is None else float(float_param_def.min)
                    max_val = None if float_param_def.max is None else float(float_param_def.max)
                    if float_param_def.step is not None:
                        step_val = float_param_def.step
                elif input_type_or_list == 'STRING':
                    # STRING is the only input type whose server default is nullable; fall back to '' so every
                    # ParameterDef carries a concrete default (see ParameterDef.default_value).
                    string_param_def = None if input_param_def is None else StrParamDef.model_validate(input_tuple[1])
                    default_value = '' if string_param_def is None or string_param_def.default is None \
                        else string_param_def.default
                elif isinstance(input_type_or_list, list):
                    assert len(input_type_or_list) > 0
                    default_value = input_type_or_list[0]
                elif input_category == 'optional':
                    logger.warning(f'"{node_name}" preprocessor: not sure how to handle optional input'
                                   f' {input_name}={input_tuple}, ignoring it.')
                    continue
                else:
                    logger.warning(f'Skipping "{node_name}" preprocessor node: not sure how to handle input'
                                   f' {input_name}={input_tuple}')
                    invalid_input_found = True
                    break
                parameter = ParameterDef(key=key,
                                         default_value=default_value,
                                         description=param_description,
                                         required=input_category=="required",
                                         min_val=min_val,
                                         max_val=max_val,
                                         step_val=step_val,
                                         option_list=options)
                parameter_list.append(parameter)
        if invalid_input_found:
            continue
        if node_info.display_name is not None:
            display_name = node_info.display_name
        else:
            display_name = node_name
        if display_name.endswith(PREPROCESSOR_SUFFIX):
            display_name = display_name[:-len(PREPROCESSOR_SUFFIX)]
        preprocessor = ControlNetPreprocessor(name=node_name, description=display_name, parameters=parameter_list)
        if node_info.description is not None:
            preprocessor.description += "\n" + node_info.description
        preprocessor.category_name = category
        preprocessor.has_image_input = has_image_input
        preprocessor.has_mask_input = has_mask_input
        preprocessors.append(preprocessor)
    return preprocessors
