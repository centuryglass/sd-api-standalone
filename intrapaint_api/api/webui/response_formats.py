"""WebUI API response data formats."""
from typing import Any, TypeAlias, Optional, TypedDict

from pydantic import BaseModel


class ProgressStateDict(BaseModel):
    """Defines the "state" section within /sdapi/v1/progress responses."""
    skipped: bool
    interrupted: bool
    stopping_generation: bool
    job: str
    job_count: int
    job_timestamp: str
    job_no: int
    sampling_step: int
    sampling_steps: int


class ProgressResponseBody(BaseModel):
    """WebUI API response format for the /sdapi/v1/progress endpoint."""
    progress: float  # Fraction completed
    eta_relative: float  # Expected time remaining in seconds
    state: ProgressStateDict
    current_image: Optional[str]
    textinfo: Optional[str]


class LoraInfo(BaseModel):
    """Data used to define a LoRA model in WebUI API responses from the /sdapi/v1/loras endpoint."""
    name: str
    alias: str
    path: str  # NOTE: this is an absolute path
    # Open-ended `ss_*` training metadata (often absent / an empty dict); we don't consume it, so keep it opaque
    # rather than modeling the hundreds of possible keys.
    metadata: dict[str, Any] = {}


class ModelInfo(BaseModel):
    """Data used to define Stable Diffusion models in WebUI API responses from the /sdapi/v1/sd-models endpoint."""
    title: str
    model_name: str
    hash: Optional[str] = None  # null for models the server hasn't hashed
    sha256: Optional[str] = None
    filename: str
    config: Optional[str] = None


class VaeInfo(BaseModel):
    """Data used to define Stable Diffusion VAE models in WebUI API responses from the /sdapi/v1/sd-vae endpoint."""
    model_name: str
    filename: str


class SamplerInfo(BaseModel):
    """Data used to define Stable Diffusion samplers in WebUI API responses from the /sdap1/v1/samplers endpoint."""
    name: str
    aliases: list[str]
    options: dict[str, str]


class UpscalerInfo(BaseModel):
    """Data used to define upscalers in WebUI API responses from the /sdapi/v1/upscalers endpoint."""
    name: str
    model_name: Optional[str]
    model_path: Optional[str]
    model_url: Optional[str]
    scale: float


class HypernetworkInfo(BaseModel):
    """Data used to define hypernetwork models in API responses from the /sdapi/v1/hypernetworks endpoint."""
    name: str
    path: str


class LatentUpscalerInfo(BaseModel):
    """Data used to define latent upscalers in API responses from the /sdapi/v1/latent-upscale-modes endpoint."""
    name: str


# Prompt styles are sent as a list of JSON object strings.
PromptStyleRes: TypeAlias = list[str]


class GenerationInfoData(BaseModel):
    """Extra info data returned in a JSON string in txt2img/img2img responses. Sends back provided parameters, defaults
       used for parameters that weren't specified, additional batch output info, seed values used, and other misc.
       information."""
    prompt: str
    all_prompts: list[str]
    negative_prompt: str
    all_negative_prompts: list[str]
    seed: int
    all_seeds: list[int]
    subseed: int
    all_subseeds: list[int]
    subseed_strength: float
    width: int
    height: int
    sampler_name: str
    cfg_scale: float
    batch_size: int
    restore_faces: bool
    sd_model_name: str
    sd_model_hash: str
    sd_vae_name: Optional[str]
    sd_vae_hash: Optional[str]
    seed_resize_from_w: int
    seed_resize_from_h: int
    denoising_strength: Optional[float]
    extra_generation_params: dict[str, Any]
    index_of_first_image: int
    infotexts: list[str]  # A.K.A. metadata
    styles: list[str]
    job_timestamp: str  # int string
    clip_skip: int
    is_using_inpainting_conditioning: bool
    version: str


# These response wrappers are TypedDicts, not validated models: we only read `images` and `info`, while the echoed
# `parameters` and other extra fields vary by fork and endpoint (e.g. /controlnet/detect omits `parameters`). A
# TypedDict documents the shape without a strict runtime schema that would reject those variations.
class Txt2ImgResponse(TypedDict):
    """WebUI API response for a successful /sdapi/v1/txt2img request."""
    images: list[str]  # base64 image list
    parameters: dict[str, Any]  # Opaque echo of the submitted request parameters.
    info: str  # Serialized JSON, parses as GenerationInfoData


class Img2ImgResponse(TypedDict):
    """WebUI API response for a successful /sdapi/v1/img2img request."""
    images: list[str]  # base64 image list
    parameters: dict[str, Any]  # Opaque echo of submitted params.
    info: str  # Serialized JSON, parses as GenerationInfoData


class PromptStyleData(BaseModel):
    """Data used to define prompt styles in API responses from the /sdapi/v1/prompt-style endpoint, after parsing from
       JSON string."""
    name: str
    prompt: str
    negative_prompt: str


class InterrogateResponse(BaseModel):
    """Response format for /sdapi/v1/interrogate API requests. The documentation lists the expected response as a plain
       string, so probably best to make sure responses actually fit this pattern before assuming that they do."""
    caption: str
