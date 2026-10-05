"""WebUI API custom script data definitions."""
from typing import Any, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel


class ScriptRequestData(TypedDict):
    """Object format to use when invoking a script in a txt2img/img2img request body. Built and sent by us, so it
       stays a plain TypedDict."""
    args: list[Any]


class ScriptResponseData(BaseModel):
    """Response containing available image generation scripts."""
    txt2img: list[str]
    img2img: list[str]


class ScriptParamDef(BaseModel):
    """Defines a parameter taken by a custom script."""
    label: str
    value: Any = None
    minimum: Optional[int | float] = None
    maximum: Optional[int | float] = None
    step: Optional[int | float] = None
    choices: Optional[list[Any]] = None


class ScriptInfo(BaseModel):
    """Defines the properties of a custom script."""
    name: str
    is_alwayson: bool
    is_img2img: bool
    args: list[ScriptParamDef]
