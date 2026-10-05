"""Defines a ControlNet preprocessor's name and parameters, for use with a Stable Diffusion API"""

from typing import Optional, Any, Self

from pydantic import BaseModel, model_validator


def _value_type_matches(value: Any, default: Any) -> bool:
    """Whether a provided parameter value is type-compatible with its default.

    `bool` is a subclass of `int`, so it's checked exactly (a bool is never a valid int value and
    vice versa); other ints and floats are treated as interchangeable numbers.
    """
    if isinstance(value, bool) or isinstance(default, bool):
        return isinstance(value, bool) and isinstance(default, bool)
    if isinstance(default, (int, float)):
        return isinstance(value, (int, float))
    return isinstance(value, type(default))


class ParameterDef(BaseModel):
    """Defines a ControlNet preprocessor parameter."""
    key: str
    default_value: str|int|float|bool
    """Reference default for the parameter. Every parameter carries one so there's always a type to validate provided
       values against, but for optional parameters it's only a fallback — it isn't sent unless the caller sets the
       value (see `required`)."""
    description: str = ''
    required: bool = False
    """Whether this parameter's default is auto-applied when the caller leaves it unset. Required parameters are always
       present in `parameter_values`; optional ones are omitted unless explicitly set, so a backend (notably a dynamic
       ComfyUI custom node) can treat an absent optional input as undefined rather than receiving a synthesized
       default it never asked for."""

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
        """Fill in defaults for required parameters left unset by the caller.

        Optional parameters are deliberately left absent unless explicitly provided, so a backend can distinguish
        "use the default" from "leave undefined" (see `ParameterDef.required`).
        """
        if isinstance(data, dict) and "typedef" in data:
            typedef = data['typedef']
            if not isinstance(typedef, ControlNetPreprocessor):
                typedef = ControlNetPreprocessor.model_validate(typedef)
            param_values = data.setdefault('parameter_values', {})
            for param_def in typedef.parameters:
                if param_def.required:
                    param_values.setdefault(param_def.key, param_def.default_value)
        return data

    @model_validator(mode='after')
    def validate_parameters(self) -> Self:
        expected_keys = set()
        for param_def in self.typedef.parameters:
            expected_keys.add(param_def.key)
            if param_def.key not in self.parameter_values:
                continue  # optional parameter left undefined by the caller
            value = self.parameter_values[param_def.key]
            if not _value_type_matches(value, param_def.default_value):
                raise ValueError(f"Preprocessor {self.typedef.name}.{param_def.key}: expected "
                                 f"{type(param_def.default_value).__name__}, found {type(value).__name__}")
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
        for key in self.parameter_values:
            if key not in expected_keys:
                raise ValueError(f"Preprocessor {self.typedef.name}: unexpected {key} param")
        return self



