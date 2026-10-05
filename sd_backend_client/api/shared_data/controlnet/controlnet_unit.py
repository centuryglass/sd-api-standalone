"""Represents everything needed to define a ControlNet layer, including a preprocessor module and/or a ControlNet model
 (usually both, but not always), and all associated settings.  Values can be serialized and deserialized in both WebUI
 and ComfyUI formats."""
from typing import Optional

from PIL import Image
from pydantic import BaseModel, ConfigDict

from sd_backend_client.api.shared_data.controlnet.controlnet_model import ControlNetModel
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import PreprocessorParams

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
    """The point where the ControlNet is enabled, as a fraction of total diffusion step count."""

    control_end: float = 1.0
    """The point where the ControlNet is disabled, as a fraction of total diffusion step count."""

    low_vram: bool = False
    """WebUI-specific low VRAM toggle"""

    pixel_perfect: bool = True
    """WebUI-specific toggle for exact image resolution matching."""