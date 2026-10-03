"""Unit tests for the A1111/Forge request body — pure dataclass -> wire-dict logic.

``DiffusionRequestBody`` is already a dataclass, and its ``to_dict()`` is what actually
shapes the JSON sent to the WebUI. These tests pin that transform without a server. They
construct the body directly (never ``load_data`` / ``Cache``), so they survive the config
refactor; ``DiffusionRequestBody`` is essentially the target params dataclass for A1111.
"""
import base64
import io

from PIL import Image

from intrapaint_api.api.webui.diffusion_request_body import DiffusionRequestBody
from intrapaint_api.util.visual.image_utils import BASE_64_PREFIX, image_from_base64, image_to_base64


def test_to_dict_strips_unset_optional_fields():
    body = DiffusionRequestBody()
    data = body.to_dict()
    # None-valued optionals must not be serialized (the server would reject/misread them).
    assert 'denoising_strength' not in data
    assert 'mask' not in data
    assert 'init_images' not in data
    # Non-None defaults are kept.
    assert data['steps'] == body.steps
    assert data['cfg_scale'] == body.cfg_scale
    assert data['width'] == body.width and data['height'] == body.height


def test_to_dict_keeps_explicitly_set_optionals():
    body = DiffusionRequestBody()
    body.denoising_strength = 0.55
    body.mask = Image.new('RGBA', (2, 2), (255, 0, 0, 255))
    data = body.to_dict()
    assert data['denoising_strength'] == 0.55
    assert data['mask'].startswith(BASE_64_PREFIX)


def _emitted_mask_pixels(mask: Image.Image) -> list[int]:
    """Serialize `mask` through to_dict() and return the grayscale values of the emitted mask, row by row."""
    body = DiffusionRequestBody()
    body.mask = mask
    emitted = image_from_base64(body.to_dict()['mask'])
    pixels = [emitted.getpixel((x, y)) for y in range(emitted.height) for x in range(emitted.width)]
    # image_from_base64 normalizes to RGBA; an opaque L-mode source decodes with R == G == B and alpha 255.
    assert all(pixel[3] == 255 and pixel[0] == pixel[1] == pixel[2] for pixel in pixels)
    return [pixel[0] for pixel in pixels]


def test_to_dict_sends_alpha_mask_as_opaque_grayscale():
    """A colored alpha mask is sent with brightness equal to its alpha."""
    # Opaque red where selected, transparent elsewhere, one half-selected pixel.
    mask = Image.new('RGBA', (3, 1), (0, 0, 0, 0))
    mask.putpixel((0, 0), (255, 0, 0, 255))
    mask.putpixel((1, 0), (255, 0, 0, 128))
    assert _emitted_mask_pixels(mask) == [255, 128, 0]


def test_to_dict_sends_fully_opaque_colored_mask_as_all_white():
    """An opaque colored mask reads as fully masked."""
    # A selection covering the whole image is opaque everywhere; brightness comes from alpha, not the highlight color.
    mask = Image.new('RGBA', (2, 1), (255, 0, 0, 255))
    assert _emitted_mask_pixels(mask) == [255, 255]


def test_to_dict_keeps_grayscale_mask_luminance():
    """An L-mode mask is sent with its own luminance."""
    grayscale = Image.new('L', (3, 1), 0)
    grayscale.putpixel((0, 0), 255)
    grayscale.putpixel((1, 0), 100)
    assert _emitted_mask_pixels(grayscale) == [255, 100, 0]


def test_to_dict_keeps_grayscale_mask_luminance_after_rgba_normalization():
    """An opaque grayscale mask keeps its luminance after RGBA normalization; reading alpha would make it all white."""
    # An L-mode mask that went through image_from_base64 is RGBA with alpha 255 everywhere.
    grayscale = Image.new('L', (3, 1), 0)
    grayscale.putpixel((0, 0), 255)
    grayscale.putpixel((1, 0), 100)
    normalized = image_from_base64(image_to_base64(grayscale))
    assert normalized.mode == 'RGBA'
    assert _emitted_mask_pixels(normalized) == [255, 100, 0]


def test_to_dict_emits_single_channel_mask():
    """The mask is encoded as a single-channel L-mode PNG."""
    body = DiffusionRequestBody()
    body.mask = Image.new('RGBA', (2, 2), (255, 0, 0, 255))
    encoded = body.to_dict()['mask'][len(BASE_64_PREFIX):]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as emitted:
        assert emitted.mode == 'L'


def test_to_dict_passes_through_core_generation_params():
    body = DiffusionRequestBody()
    body.prompt = 'a red apple'
    body.negative_prompt = 'blurry'
    body.seed = 99
    body.steps = 7
    body.sampler_name = 'Euler a'
    data = body.to_dict()
    assert data['prompt'] == 'a red apple'
    assert data['negative_prompt'] == 'blurry'
    assert data['seed'] == 99
    assert data['steps'] == 7
    assert data['sampler_name'] == 'Euler a'


# def test_add_init_image_appends_base64_data_uri():
# TODO: test to_dict serializes correctly with base64 images