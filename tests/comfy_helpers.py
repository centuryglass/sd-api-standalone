"""ComfyUI-specific helpers for the integration tests.

ComfyUI's API is asynchronous and workflow-based: generation calls *queue* a job and
return a ``prompt_id``; you then poll ``check_queue_entry`` until it finishes and
download the resulting image references. These helpers wrap that lifecycle and the
cache setup the workflow builders read from.
"""
import time

from PIL import Image

from intrapaint_api.api.comfyui.comfyui_diffusion_params import ComfyUIDiffusionParams
from intrapaint_api.api.comfyui_webservice import AsyncTaskStatus, ComfyUiWebservice

# ComfyUI SD1.5 defaults kept small/cheap but coherent enough to compare.
COMFY_SIZE = 256
COMFY_STEPS = 8


def build_comfy_params(checkpoint: str, sampler: str = 'euler', scheduler: str = 'karras',
                          prompt: str = 'a red apple on a wooden table') -> ComfyUIDiffusionParams:
    """Create small, valid ComfyUI generation parameter set."""
    return ComfyUIDiffusionParams(sd_model_name=checkpoint,
                                    prompt=prompt,
                                    steps=COMFY_STEPS,
                                    sampler=sampler,
                                    scheduler=scheduler,
                                    cfg_scale=7.0,
                                    batch_size=1,
                                    width=COMFY_SIZE,
                                    height=COMFY_SIZE,
                                    seed=1)



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
    if response.prompt_id is None:
        raise RuntimeError(f'ComfyUI rejected the workflow: {response.node_errors or response}')
    prompt_id = response.prompt_id
    task_number = response.number
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        progress = service.check_queue_entry(prompt_id, task_number)
        status = progress.status
        if status == AsyncTaskStatus.FINISHED:
            if progress.outputs is None or progress.outputs.images is None:
                images = []
            else:
                images = progress.outputs.images
            return service.download_images(images)
        if status == AsyncTaskStatus.FAILED:
            raise RuntimeError(f'ComfyUI task {prompt_id} failed during execution')
        time.sleep(poll)
    raise TimeoutError(f'ComfyUI task {prompt_id} did not finish within {timeout}s')
