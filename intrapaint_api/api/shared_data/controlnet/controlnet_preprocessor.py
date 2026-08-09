"""Defines a ControlNet preprocessor's name and parameters, for use with a Stable Diffusion API"""

from typing import Optional, Any, Self

from pydantic import BaseModel, model_validator

class ParameterDef(BaseModel):
    """Defines a ControlNet preprocessor parameter."""
    key: str
    default_value: Optional[str|int|float|bool]
    description: str = ''
    required: bool = False

    option_list: Optional[list[str|int|float|bool]] = None
    """List of valid options. If None, all options are valid."""

    min_val: Optional[int|float] = None
    """Minimum value, ignored if non-numeric. If unset, value has no minimum bounds."""

    max_val: Optional[int|float] = None
    """Maximum value, ignored if non-numeric. If unset, value has no maximum bounds."""

    step_val: Optional[int|float] = None
    """Smallest value change, ignored if non-numeric."""


class ControlNetPreprocessor(BaseModel):
    """Defines a ControlNet preprocessor's name and parameters for use with a Stable Diffusion API."""
    name: str

    category_name: str = ""
    """Category label for grouping with similar preprocessors."""

    description: str = ""
    """Description string provided by a server API."""

    has_image_input: bool = True
    """Whether this preprocessor accepts an image input when converted into a ComfyUI node."""

    has_mask_input: bool = True
    """Whether this preprocessor accepts a mask input when converted into a ComfyUI node."""

    model_free: bool = False
    """Whether this preprocessor can be used without a ControlNet model."""

    parameters: list[ParameterDef] = []
    """Definitions for all additional parameter types the preprocessor accepts."""

class PreprocessorParams(BaseModel):
    """Selected ControlNet Preprocessor and parameter values."""
    typedef: ControlNetPreprocessor
    parameter_values: dict[str, Any]

    @model_validator(mode='before')
    @classmethod
    def build_params_from_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict) and "typedef" in dict:
            typedef = data['typedef']
            if not isinstance(typedef, ControlNetPreprocessor):
                typedef = ControlNetPreprocessor.model_validate(typedef)
            if 'parameter_values' not in data:
                data['parameter_values'] = {}
            param_values = data['parameter_values']
            for param_def in typedef.parameters:
                if param_def.key in param_values:
                    continue
                if param_def.default_value is not None:
                    param_values[param_def.key] = param_def.default_value
                elif param_def.required:
                    raise ValueError(f"Preprocessor {typedef.name}.{param_def.key} required but missing default,"
                                     f" PreprocessorParams must provide parameter_values on construction")

    @model_validator(mode='after')
    def validate_parameters(self) -> Self:
        expected_keys = set()
        for param_def in self.typedef.parameters:
            expected_keys.add(param_def.key)
            if param_def.key in self.parameter_values:
                value = self.parameter_values[param_def.key]
                if type(value) != type(param_def.default_value):
                    raise ValueError(f"Preprocessor {self.typedef.name}.{param_def.key}: expected "
                                     f"{type(param_def.default_value)}, found {type(value)}")
                if param_def.option_list is not None and value not in param_def.option_list:
                    raise ValueError(f"Preprocessor {self.typedef.name}.{param_def.key}: invalid option {value}, expected "
                                     f"selection from {param_def.option_list}")
                if (param_def.min_val is not None or param_def.max_val is not None) and isinstance(value, (int, float)):
                    if param_def.max_val is not None and value > param_def.max_val:
                        raise ValueError( f"Preprocessor {self.typedef.name}.{param_def.key}: invalid value {value}, "
                                          f"expected value <= {param_def.max_val}")
                    if param_def.min_val is not None and value < param_def.min_val:
                        raise ValueError( f"Preprocessor {self.typedef.name}.{param_def.key}: invalid value {value}, "
                                          f"expected value >= {param_def.min_val}")
            elif param_def.required:
                raise ValueError(f"Preprocessor {self.typedef.name}: missing required {param_def.key} param")
        for key in self.parameter_values:
            if key not in expected_keys:
                raise ValueError(f"Preprocessor {self.typedef.name}: unexpected {key} param")
        return self



