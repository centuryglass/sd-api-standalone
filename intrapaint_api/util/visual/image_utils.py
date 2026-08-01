"""Helpers for converting between PIL images and the byte/base64 formats image generation backends expect.

This is a slimmed, Qt-free replacement for IntraPaint's much larger image_utils module: the standalone API only
needs to encode images for upload and decode them from responses.
"""
import base64
import io
from typing import Union

from PIL import Image

BASE_64_PREFIX = 'data:image/png;base64,'


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG byte data."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def image_from_bytes(data: bytes) -> Image.Image:
    """Load a PIL image from raw encoded image bytes (PNG, JPEG, etc.), normalized to RGBA."""
    image = Image.open(io.BytesIO(data))
    image.load()
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    return image


def image_to_base64(image: Union[Image.Image, str], include_prefix: bool = False) -> str:
    """Convert a PIL image or an image file path to a base64 string.

    When given a file path the raw file bytes are encoded as-is; when given a PIL image it is re-encoded as PNG.
    """
    if isinstance(image, str):
        with open(image, 'rb') as file:
            image_str = base64.b64encode(file.read()).decode('utf-8')
    else:
        image_str = base64.b64encode(image_to_png_bytes(image)).decode('utf-8')
    if include_prefix:
        return BASE_64_PREFIX + image_str
    return image_str


def image_from_base64(image_str: str) -> Image.Image:
    """Return a PIL image (RGBA) from base64-encoded string data, tolerating an optional data-URI prefix."""
    if image_str.startswith(BASE_64_PREFIX):
        image_str = image_str[len(BASE_64_PREFIX):]
    data = base64.b64decode(image_str)
    return image_from_bytes(data)
