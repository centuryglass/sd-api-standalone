"""Unit tests for the Qt-free image and geometry helpers.

These replaced Qt (QImage/QSize) during extraction and underpin every request the clients
build, so their round-trip behavior is worth pinning independently of any backend. Pure,
fast, no server.
"""
from PIL import Image

from sd_backend_client.util.geometry import Size
from sd_backend_client.util.visual.image_utils import (BASE_64_PREFIX, image_from_base64, image_from_bytes,
                                                    image_to_base64, image_to_png_bytes)


def test_base64_round_trip_preserves_pixels():
    original = Image.new('RGBA', (5, 3), (12, 34, 56, 255))
    encoded = image_to_base64(original)
    decoded = image_from_base64(encoded)
    assert decoded.size == (5, 3)
    assert decoded.mode == 'RGBA'
    assert decoded.getpixel((0, 0)) == (12, 34, 56, 255)


def test_base64_prefix_is_optional_on_both_ends():
    original = Image.new('RGBA', (4, 4), (200, 100, 50, 255))
    with_prefix = image_to_base64(original, include_prefix=True)
    without_prefix = image_to_base64(original, include_prefix=False)
    assert with_prefix.startswith(BASE_64_PREFIX)
    assert not without_prefix.startswith(BASE_64_PREFIX)
    # Decoding tolerates the data-URI prefix either way.
    assert image_from_base64(with_prefix).getpixel((0, 0)) == (200, 100, 50, 255)
    assert image_from_base64(without_prefix).getpixel((0, 0)) == (200, 100, 50, 255)


def test_image_from_bytes_normalizes_to_rgba():
    rgb = Image.new('RGB', (6, 6), (10, 20, 30))
    decoded = image_from_bytes(image_to_png_bytes(rgb))
    assert decoded.mode == 'RGBA'
    assert decoded.getpixel((0, 0)) == (10, 20, 30, 255)


def test_size_width_height_accessors():
    size = Size(384, 256)
    assert size.width() == 384
    assert size.height() == 256


def test_size_equality():
    assert Size(100, 200) == Size(100, 200)
    assert Size(100, 200) != Size(200, 100)
