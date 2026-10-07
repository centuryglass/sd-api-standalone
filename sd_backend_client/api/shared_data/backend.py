"""The submission interface both backend clients implement, so callers can drive either one the same way.

`A1111Webservice` and `ComfyUiWebservice` subclass `Backend`. Each `submit_*` method takes a `DiffusionParams` (or
any subclass), copies what it needs before returning, and returns a `GenerationHandle` for the queued job. Fields
that only the other backend supports are ignored. Methods outside this class (blocking `txt2img`, model listings,
`interrupt`) are backend-specific and differ in signature and return type.
"""
from abc import ABC, abstractmethod

from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams
from sd_backend_client.api.shared_data.generation_handle import GenerationHandle

__all__ = ['Backend', 'require_init_image', 'require_mask']


class Backend(ABC):
    """A Stable Diffusion server client that queues generation jobs and returns a handle for each.

    Every `submit_*` method validates its inputs before contacting the server, raising `ValueError` for a missing
    image or mask. A returned handle's job runs with the parameters as they were at submit time, so the caller may
    change or reuse `diffusion_params` afterwards.
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


def require_init_image(diffusion_params: DiffusionParams, operation: str) -> None:
    """Raise `ValueError` unless `diffusion_params` holds an init image for `operation`."""
    if not diffusion_params.init_images:
        raise ValueError(f'Must set an init image in diffusion_params for {operation}')


def require_mask(diffusion_params: DiffusionParams, operation: str) -> None:
    """Raise `ValueError` unless `diffusion_params` holds a mask for `operation`."""
    if diffusion_params.mask is None:
        raise ValueError(f'Must set a mask in diffusion_params for {operation}')
