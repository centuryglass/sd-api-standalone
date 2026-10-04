"""Offline unit tests for `ComfyUiWebservice.upload_mask` polarity.

`DiffusionParams.mask` means opaque/white = change. ComfyUI's `LoadImageMask` (channel `alpha`) treats transparent
pixels as the region to change, so `upload_mask` must send alpha = 255 - luminance. These tests capture the PNG bytes
handed to the HTTP layer.
"""
import io
from typing import Any

from PIL import Image

from intrapaint_api.api.comfyui.comfyui_types import ImageFileReference
from intrapaint_api.api.comfyui_webservice import ComfyEndpoints, ComfyUiWebservice


class _FakeResponse:
    """Minimal stand-in for the upload endpoint's HTTP response."""

    def json(self) -> dict[str, Any]:
        """Return a valid upload response body."""
        return {'name': 'mask_src.png', 'subfolder': '', 'type': 'input'}


def _upload(mask: Image.Image) -> tuple[Image.Image, str]:
    """Upload `mask` through a service with a fake `post`, returning the decoded uploaded PNG and the endpoint."""
    service = ComfyUiWebservice('http://localhost:8188')
    sent: dict[str, Any] = {}

    def fake_post(endpoint: str, **kwargs: Any) -> _FakeResponse:
        sent['endpoint'] = endpoint
        sent['files'] = kwargs['files']
        return _FakeResponse()

    service.post = fake_post  # type: ignore[method-assign]
    service.upload_mask(mask, ImageFileReference(filename='src.png', subfolder='', type='input'))
    _name, data, _type = next(iter(sent['files'].values()))
    return Image.open(io.BytesIO(data)).convert('RGBA'), sent['endpoint']


def _half_mask(opaque_gray: bool) -> Image.Image:
    """4x2 mask whose left half marks the change region."""
    if opaque_gray:
        mask = Image.new('RGBA', (4, 2), (0, 0, 0, 255))
        mask.paste(Image.new('RGBA', (2, 2), (255, 255, 255, 255)), (0, 0))
    else:
        mask = Image.new('RGBA', (4, 2), (255, 255, 255, 0))
        mask.paste(Image.new('RGBA', (2, 2), (255, 255, 255, 255)), (0, 0))
    return mask


def _assert_left_transparent(uploaded: Image.Image) -> None:
    assert uploaded.size == (4, 2)
    assert uploaded.getpixel((0, 0))[3] == 0
    assert uploaded.getpixel((1, 1))[3] == 0
    assert uploaded.getpixel((2, 0))[3] == 255
    assert uploaded.getpixel((3, 1))[3] == 255


def test_white_on_black_mask_uploads_white_as_transparent() -> None:
    """A white-on-black mask changes the white region, which must upload transparent."""
    uploaded, endpoint = _upload(_half_mask(opaque_gray=True))
    assert endpoint == ComfyEndpoints.MASK_UPLOAD
    _assert_left_transparent(uploaded)


def test_alpha_mask_uploads_opaque_as_transparent() -> None:
    """An alpha mask changes its opaque region, which must upload transparent."""
    uploaded, _endpoint = _upload(_half_mask(opaque_gray=False))
    _assert_left_transparent(uploaded)


def test_grayscale_values_are_inverted_into_alpha() -> None:
    """Intermediate grayscale levels map to alpha = 255 - level."""
    uploaded, _endpoint = _upload(Image.new('L', (1, 1), 64))
    assert uploaded.getpixel((0, 0))[3] == 255 - 64
