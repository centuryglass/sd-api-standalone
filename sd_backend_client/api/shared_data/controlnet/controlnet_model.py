"""Provides a common data representation for ControlNet models."""
import re
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

# Misc. regular expressions to use for attempting to extract relevant segments from a full ControlNet model name
# and construct a display name. We can't assume that any of these are present, but when they are they should be safe
# to use.  Patterns are based off of links on https://github.com/Mikubill/sd-webui-controlnet/wiki/Model-download,
# I've tried to make sure all the most common ones are covered.


CONTROL_PREFIX_PATTERN = r'^(?:control[_-]|TTPLANET_Controlnet_|CN-|cn_)'
MODEL_HASH_PATTERN = r' \[([0-9a-f]{8})]$'
MODEL_EXTENSION_PATTERN = r'(\.(?:safetensors|pth|ckpt|bin))$'
CONTROL_CATEGORY_PATTERN = r'^(t2iadapter)_'


def match_segment(pattern):
    """Match a pattern surrounded by '_', '-', or the beginning/end of string on both sides."""
    return f'(?:^|[_-]){pattern}(?:$|[_-])'


MODEL_FORMAT_PATTERN = match_segment(r'(fp16|lora\d*|rank\d*)')
MODEL_SD_VERSION_PATTERN = match_segment(r'(sd15|sd15s2|xl|sd15_lora|sdxl|sdxl_lora|sdxl_unnorm|sdxl_vit-h|sd15_plus)')
MODEL_VERSION_PATTERN = r'[_-]?(v\d+[A-Za-z0-9]*)(?:[_-]|$)'


class ControlNetModel:
    """Provides a common data representation for ControlNet models.

    Equality and hashing use `full_model_name`. Pydantic fields of this type accept an instance or a full model name,
    and serialize to the full model name in JSON.
    """

    def __init__(self, full_model_name: str) -> None:
        self._full_model_name = full_model_name
        model_name = full_model_name
        value_dict = {}

        for pattern in (CONTROL_PREFIX_PATTERN, MODEL_HASH_PATTERN, MODEL_EXTENSION_PATTERN,
                        MODEL_FORMAT_PATTERN, CONTROL_CATEGORY_PATTERN, MODEL_VERSION_PATTERN,
                        MODEL_SD_VERSION_PATTERN):
            while (match := re.search(pattern, model_name)) is not None:
                if match.lastindex is not None and match.lastindex >= 1:
                    if pattern not in value_dict:
                        value_dict[pattern] = match.group(1)
                    else:
                        value_dict[pattern] += f', {match.group(1)}'
                start, end = match.span()
                model_name = model_name[:start] + model_name[end:]

        self._short_name = model_name
        self._hash = value_dict.get(MODEL_HASH_PATTERN, None)
        self._file_extension = value_dict.get(MODEL_EXTENSION_PATTERN, None)
        self._format = value_dict.get(MODEL_FORMAT_PATTERN, None)
        self._category = value_dict.get(CONTROL_CATEGORY_PATTERN, None)
        self._version = value_dict.get(MODEL_VERSION_PATTERN, None)
        self._sd_version = value_dict.get(MODEL_SD_VERSION_PATTERN, None)

        display_name = f'{self._category}:{model_name}' if self._category is not None else model_name
        extra_info = [str(info_str) for info_str in [self._sd_version, self._version, self._format] if info_str is not None]
        if len(extra_info) > 0:
            display_name += f' ({", ".join(extra_info)})'
        self._display_name = display_name

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ControlNetModel):
            return False
        return self.full_model_name == other.full_model_name

    def __hash__(self) -> int:
        return hash(self._full_model_name)

    def __repr__(self) -> str:
        return f'ControlNetModel({self._full_model_name!r})'

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any,
                                     _handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        from_name = core_schema.chain_schema([core_schema.str_schema(),
                                              core_schema.no_info_plain_validator_function(cls)])
        return core_schema.json_or_python_schema(
            json_schema=from_name,
            python_schema=core_schema.union_schema([core_schema.is_instance_schema(cls), from_name]),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda model: model.full_model_name,
                                                                           when_used='json'))

    @property
    def full_model_name(self) -> str:
        """Returns the full model name, used to reference it in API calls."""
        return self._full_model_name

    @property
    def display_name(self) -> str:
        """Returns the model's display name, used to reference it in the UI."""
        return self._display_name
