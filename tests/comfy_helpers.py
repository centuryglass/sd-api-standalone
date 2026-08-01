"""ComfyUI-specific helpers for the integration tests.

ComfyUI's API is asynchronous and workflow-based: generation calls *queue* a job and
return a ``prompt_id``; you then poll ``check_queue_entry`` until it finishes and
download the resulting image references. These helpers wrap that lifecycle and the
cache setup the workflow builders read from.
"""
import time

from PIL import Image

from intrapaint_api.api.comfyui_webservice import AsyncTaskStatus, ComfyUiWebservice
from intrapaint_api.api.controlnet.controlnet_model import ControlNetModel
from intrapaint_api.api.controlnet.controlnet_preprocessor import ControlNetPreprocessor
from intrapaint_api.api.controlnet.controlnet_unit import ControlKeyType, ControlNetUnit
from intrapaint_api.config.cache import Cache
from intrapaint_api.util.geometry import Size

# ComfyUI SD1.5 defaults kept small/cheap but coherent enough to compare.
COMFY_SIZE = 256
COMFY_STEPS = 8
COMFY_CONTROLNET_KEYS = (Cache.CONTROLNET_ARGS_0_COMFYUI,
                         Cache.CONTROLNET_ARGS_1_COMFYUI,
                         Cache.CONTROLNET_ARGS_2_COMFYUI)


def configure_comfy_cache(checkpoint: str, sampler: str = 'euler', scheduler: str = 'karras',
                          prompt: str = 'a red apple on a wooden table', edit_mode: str = 'Text to Image') -> None:
    """Populate the (isolated) cache with a small, valid ComfyUI generation profile.

    The isolated-cache defaults (`Euler a` / `default`) are A1111 names ComfyUI rejects,
    so real ComfyUI sampler/scheduler names must be supplied.
    """
    cache = Cache()
    # SD_MODEL / SAMPLING_METHOD / SCHEDULER are option-restricted keys whose valid options
    # are normally populated from the server; add_missing_options lets us set real ComfyUI
    # names into the otherwise-empty isolated cache.
    cache.set(Cache.SD_MODEL, checkpoint, add_missing_options=True)
    cache.set(Cache.PROMPT, prompt)
    cache.set(Cache.NEGATIVE_PROMPT, '')
    cache.set(Cache.SAMPLING_STEPS, COMFY_STEPS)
    cache.set(Cache.SAMPLING_METHOD, sampler, add_missing_options=True)
    cache.set(Cache.SCHEDULER, scheduler, add_missing_options=True)
    cache.set(Cache.GUIDANCE_SCALE, 7.0)
    cache.set(Cache.BATCH_SIZE, 1)
    cache.set(Cache.GENERATION_SIZE, Size(COMFY_SIZE, COMFY_SIZE))
    cache.set(Cache.SEED, 1)
    cache.set(Cache.EDIT_MODE, edit_mode)
    clear_comfy_controlnet_cache()


def make_comfy_mask(editable_box: tuple[int, int, int, int], size: int = COMFY_SIZE) -> Image.Image:
    """Build a ComfyUI inpainting mask that encodes the editable region in the ALPHA channel.

    ComfyUI's LoadImageMask reads channel='alpha', so a fully-opaque RGB mask carries no
    spatial information. ComfyUI's polarity is also inverted from the intuitive one
    (verified empirically): **transparent (alpha=0) is the region that gets inpainted, and
    opaque (alpha=255) is preserved**. So the editable box gets alpha=0 and the rest 255.
    """
    mask = Image.new('RGBA', (size, size), (255, 255, 255, 255))  # opaque => preserved
    editable = Image.new('RGBA', (editable_box[2] - editable_box[0], editable_box[3] - editable_box[1]),
                         (255, 255, 255, 0))  # transparent => inpainted
    mask.paste(editable, (editable_box[0], editable_box[1]))
    return mask


def clear_comfy_controlnet_cache() -> None:
    """Reset all three cached ComfyUI ControlNet slots to disabled units."""
    disabled = ControlNetUnit(ControlKeyType.COMFYUI).serialize()
    cache = Cache()
    for key in COMFY_CONTROLNET_KEYS:
        cache.set(key, disabled)


def set_comfy_controlnet_unit(preprocessor: ControlNetPreprocessor, model_name: str, image_path: str,
                              strength: float = 1.0) -> None:
    """Install a single enabled ControlNet unit into the first cached ComfyUI slot.

    ``image_path`` must be a path to an image file on disk; the client uploads it when
    the workflow is built (see ComfyUiWebservice._prepare_controlnet_data).
    """
    clear_comfy_controlnet_cache()
    unit = ControlNetUnit(ControlKeyType.COMFYUI)
    unit.enabled = True
    unit.preprocessor = preprocessor
    unit.model = ControlNetModel(model_name)
    unit.image_string = image_path
    unit.control_strength.value = strength
    Cache().set(Cache.CONTROLNET_ARGS_0_COMFYUI, unit.serialize())


def find_canny_preprocessor(service: ComfyUiWebservice):
    """Return a Canny-style preprocessor object, or None if none is installed."""
    preprocessors = service.get_controlnet_preprocessors()
    exact = next((p for p in preprocessors if p.name == 'CannyEdgePreprocessor'), None)
    if exact is not None:
        return exact
    return next((p for p in preprocessors if 'canny' in p.name.lower()), None)


def find_canny_model(service: ComfyUiWebservice):
    """Return a Canny ControlNet model filename, or None if none is installed."""
    models = service.get_controlnet_models()
    return next((m for m in models if 'canny' in m.lower()), None)


def wait_for_comfy_images(service: ComfyUiWebservice, response, timeout: float = 240.0, poll: float = 1.0):
    """Block until a queued ComfyUI job finishes, returning its downloaded PIL images.

    Raises on submission errors, task failure, or timeout.
    """
    if 'prompt_id' not in response:
        raise RuntimeError(f'ComfyUI rejected the workflow: {response.get("node_errors") or response}')
    prompt_id = response['prompt_id']
    task_number = response['number']
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        progress = service.check_queue_entry(prompt_id, task_number)
        status = progress['status']
        if status == AsyncTaskStatus.FINISHED:
            return service.download_images(progress['outputs']['images'])
        if status == AsyncTaskStatus.FAILED:
            raise RuntimeError(f'ComfyUI task {prompt_id} failed during execution')
        time.sleep(poll)
    raise TimeoutError(f'ComfyUI task {prompt_id} did not finish within {timeout}s')
