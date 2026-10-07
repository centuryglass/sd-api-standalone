"""The submission interface both backend clients implement, so callers can drive either one the same way.

`A1111Webservice` and `ComfyUiWebservice` subclass `Backend`. Each `submit_*` method copies what it needs from its
arguments before returning, and returns a `GenerationHandle` for the queued job. Fields that only the other backend
supports are ignored. Methods outside this class (blocking `txt2img` and `upscale`, model listings, `interrupt`) are
backend-specific and differ in signature and return type.
"""
from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image

from sd_backend_client.api.shared_data.api_datatypes import DiffusionUpscalingParams
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor, \
    PreprocessorParams
from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams
from sd_backend_client.api.shared_data.generation_handle import GenerationHandle

__all__ = ['Backend', 'require_init_image', 'require_mask', 'require_upscale_size']


class Backend(ABC):
    """A Stable Diffusion server client that queues generation jobs and returns a handle for each.

    Every `submit_*` method validates its inputs before contacting the server, raising `ValueError` for a missing
    image or mask, or for an upscale size that is not larger than the image. A returned handle's job runs with its
    arguments as they were at submit time, so the caller may change or reuse them afterwards.
    """

    @abstractmethod
    def submit_txt2img(self, diffusion_params: DiffusionParams) -> GenerationHandle:
        """Queue a text-to-image job and return its handle. Init images and the mask are ignored."""

    @abstractmethod
    def submit_img2img(self, diffusion_params: DiffusionParams) -> GenerationHandle:
        """Queue an image-to-image job on `diffusion_params.init_images[0]` and return its handle.

        A `mask`, if set, limits changes to the masked region, as in `submit_inpaint`.

        Raises
        ------
        ValueError
            If `diffusion_params.init_images` is empty.
        """

    @abstractmethod
    def submit_inpaint(self, diffusion_params: DiffusionParams) -> GenerationHandle:
        """Queue an inpainting job that changes the masked region of `diffusion_params.init_images[0]`.

        Raises
        ------
        ValueError
            If `diffusion_params.init_images` is empty or `diffusion_params.mask` is None.
        """

    @abstractmethod
    def submit_upscale(self, image: Image.Image, width: int, height: int,
                       upscale_params: Optional[DiffusionUpscalingParams] = None) -> GenerationHandle:
        """Queue a job that upscales `image` to `width` x `height` and return its handle.

        The result holds the one upscaled image. The job applies the upscaling model named by
        `upscale_params.upscaling_mode` and resizes to the requested size; with
        `upscale_params.use_stable_diffusion_upscaling` set it also runs a tiled diffusion pass. A basic upscale with
        `upscale_params` None or no `upscaling_mode` uses the server's first upscaling model. A named model the server
        lacks falls back to that first model on WebUI, and raises `ValueError` for a basic upscale on ComfyUI.

        Raises
        ------
        ValueError
            If `width` x `height` does not exceed the size of `image` in at least one dimension.
        """

    @abstractmethod
    def submit_preprocessor_preview(self, image: Image.Image,
                                    preprocessor: ControlNetPreprocessor | PreprocessorParams,
                                    mask: Optional[Image.Image] = None) -> GenerationHandle:
        """Queue a job that runs a ControlNet preprocessor on `image` and return its handle.

        The result holds the one preprocessed control image. A bare `ControlNetPreprocessor` runs with the server's
        parameter defaults, and a `PreprocessorParams` overrides them with its `parameter_values`. `mask` is passed
        only to preprocessors that take one, such as inpainting preprocessors.
        """


def require_init_image(diffusion_params: DiffusionParams, operation: str) -> None:
    """Raise `ValueError` unless `diffusion_params` holds an init image for `operation`."""
    if not diffusion_params.init_images:
        raise ValueError(f'Must set an init image in diffusion_params for {operation}')


def require_mask(diffusion_params: DiffusionParams, operation: str) -> None:
    """Raise `ValueError` unless `diffusion_params` holds a mask for `operation`."""
    if diffusion_params.mask is None:
        raise ValueError(f'Must set a mask in diffusion_params for {operation}')


def require_upscale_size(image: Image.Image, width: int, height: int) -> None:
    """Raise `ValueError` unless `width` x `height` exceeds the size of `image` in at least one dimension."""
    if width <= 0 or height <= 0 or (width <= image.width and height <= image.height):
        raise ValueError(f'Requested size {width}x{height} must exceed the source size {image.width}x{image.height} '
                         'in at least one dimension')
