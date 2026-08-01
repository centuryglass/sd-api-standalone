"""Small shared helpers for the integration tests."""
import os

from PIL import Image, ImageChops, ImageDraw

from intrapaint_api.api.webui.controlnet_webui_constants import CONTROLNET_SCRIPT_KEY
from intrapaint_api.api.webui.diffusion_request_body import DiffusionRequestBody
from intrapaint_api.config.cache import Cache
from intrapaint_api.util.geometry import Size
from intrapaint_api.util.visual.image_utils import image_to_base64

# Keep generation cheap: tiny canvas, minimal steps, single image.
TEST_SIZE = 256
TEST_STEPS = 4


def make_test_image(width: int = TEST_SIZE, height: int = TEST_SIZE,
                    color: tuple[int, int, int, int] = (128, 64, 200, 255)) -> Image.Image:
    """A solid-color RGBA image usable as an img2img / upscale / interrogate source."""
    return Image.new('RGBA', (width, height), color)


def make_mask(width: int = TEST_SIZE, height: int = TEST_SIZE) -> Image.Image:
    """A mask with a white square in the middle (region to inpaint) on black."""
    mask = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    quarter_w, quarter_h = width // 4, height // 4
    white = Image.new('RGBA', (width - 2 * quarter_w, height - 2 * quarter_h), (255, 255, 255, 255))
    mask.paste(white, (quarter_w, quarter_h))
    return mask


def make_edge_image(width: int = TEST_SIZE, height: int = TEST_SIZE) -> Image.Image:
    """A high-contrast image with clear edges, so a Canny/edge preprocessor has something to detect.

    (A solid color has no edges and would make ControlNet a silent no-op regardless of wiring.)
    """
    image = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([width // 6, height // 6, width * 5 // 6, height * 5 // 6], outline=(0, 0, 0, 255), width=6)
    draw.ellipse([width // 3, height // 3, width * 2 // 3, height * 2 // 3], outline=(0, 0, 0, 255), width=6)
    draw.line([0, 0, width, height], fill=(0, 0, 0, 255), width=4)
    draw.line([0, height, width, 0], fill=(0, 0, 0, 255), width=4)
    return image


def images_differ(a: Image.Image, b: Image.Image) -> float:
    """Mean per-pixel absolute difference between two images (0.0 == identical).

    Used to catch ControlNet silently doing nothing: same seed + prompt with vs. without
    a control unit should produce visibly different images when ControlNet actually applies.
    """
    a_rgb = a.convert('RGB')
    b_rgb = b.convert('RGB').resize(a_rgb.size)
    diff = ImageChops.difference(a_rgb, b_rgb)
    histogram = diff.histogram()
    # Weighted mean across the three channel histograms.
    total = sum(value * (index % 256) for index, value in enumerate(histogram))
    pixel_count = a_rgb.width * a_rgb.height * 3
    return total / pixel_count if pixel_count else 0.0


def controlnet_unit_dict(module: str, model: str, control_image: Image.Image,
                        weight: float = 1.0) -> dict:
    """A minimal ControlNet unit payload for alwayson_scripts, mirroring what the WebUI API expects."""
    return {
        'enabled': True,
        'module': module,
        'model': model,
        'weight': weight,
        'image': image_to_base64(control_image, include_prefix=True),
        'resize_mode': 'Crop and Resize',
        'processor_res': TEST_SIZE,
        'guidance_start': 0.0,
        'guidance_end': 1.0,
        'control_mode': 0,
        'pixel_perfect': True,
    }


def add_controlnet_unit(body: DiffusionRequestBody, unit: dict) -> None:
    """Attach a ControlNet unit to a request body's alwayson_scripts."""
    if body.alwayson_scripts is None:
        body.alwayson_scripts = {}
    body.alwayson_scripts[CONTROLNET_SCRIPT_KEY] = {'args': [unit]}


def save_output(output_dir: str, name: str, image: Image.Image) -> str:
    """Save an image under output_dir as a PNG and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{name}.png')
    image.convert('RGBA').save(path)
    return path


def fast_request_body(prompt: str = 'a red apple on a table') -> DiffusionRequestBody:
    """A minimal, fast request body that does not depend on the shared config defaults."""
    body = DiffusionRequestBody()
    body.prompt = prompt
    body.negative_prompt = ''
    body.steps = TEST_STEPS
    body.cfg_scale = 4.0
    body.width = TEST_SIZE
    body.height = TEST_SIZE
    body.batch_size = 1
    body.n_iter = 1
    body.seed = 1
    body.save_images = False
    body.send_images = True
    return body


def configure_fast_cache() -> None:
    """Point the (isolated) cache at a small, fast generation profile."""
    cache = Cache()
    cache.set(Cache.PROMPT, 'a red apple on a table')
    cache.set(Cache.NEGATIVE_PROMPT, '')
    cache.set(Cache.SAMPLING_STEPS, TEST_STEPS)
    cache.set(Cache.BATCH_SIZE, 1)
    cache.set(Cache.BATCH_COUNT, 1)
    cache.set(Cache.GENERATION_SIZE, Size(TEST_SIZE, TEST_SIZE))
    # "Lanczos" is a built-in upscaler present on every WebUI build (see test_get_upscalers).
    cache.set(Cache.SCALING_MODE, 'Lanczos')
