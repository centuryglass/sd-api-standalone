"""Represents everything needed to define a ControlNet layer, including a preprocessor module and/or a ControlNet model
 (usually both, but not always), and all associated settings.  Values can be serialized and deserialized in both WebUI
 and ComfyUI formats."""
from enum import Enum
from typing import Optional

from PIL import Image
from pydantic import BaseModel, ConfigDict

from intrapaint_api.api.shared_data.controlnet.controlnet_model import ControlNetModel
from intrapaint_api.api.shared_data.controlnet.controlnet_preprocessor import PreprocessorParams

# The `QCoreApplication.translate` context for strings in this file
TR_ID = 'api.controlnet.controlnet_unit'


def _tr(key: str, disambiguation: Optional[str] = None, n: int = -1) -> str:
    """Helper to make `QCoreApplication.translate` more concise."""
    return key


LABEL_TEXT_CONTROL_STRENGTH = _tr('Strength')
TOOLTIP_CONTROL_STRENGTH = _tr('Controls how powerful the ControlNet unit\'s influence is on image generation')

LABEL_TEXT_CONTROL_START = _tr('Starting control step')
TOOLTIP_CONTROL_START = _tr('Step where the ControlNet unit is first activated, as a fraction of the total step count.')

LABEL_TEXT_CONTROL_END = _tr('Ending control step')
TOOLTIP_CONTROL_END = _tr('Step where the ControlNet unit is deactivated, as a fraction of the total step count.')


class ControlKeyType(Enum):
    """Sets the key format used for ControlNet model parameters (start, end, weight/strength)"""
    WEBUI = 0
    COMFYUI = 1


class ControlNetUnit(BaseModel):
    """Represents everything needed to define a ControlNet layer, including a preprocessor module and/or a ControlNet
       model (usually both, but not always), and all associated settings. Values can be serialized and deserialized in
       both WebUI and ComfyUI formats."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Optional[Image.Image] = None

    preprocessor: Optional[PreprocessorParams] = None

    model: Optional[ControlNetModel] = None

    control_strength: float = 1.0
    """ControlNet strength, ranges from 0.0 to 2.0."""

    control_start: float = 0.0
    """Fraction of the total diffusion step count to finish the ControlNet is enabled."""

    control_end: float = 0.0
    """Fraction of the total diffusion step count that should pass before the ControlNet is disabled."""

    low_vram: bool = False
    """WebUI-specific low VRAM toggle"""

    pixel_perfect: bool = True
    """WebUI-specific toggle for exact image resolution matching."""