"""Offline unit tests for `ComfyUiWebservice.upload_mask` polarity and naming.

`DiffusionParams.mask` means opaque/white = change. ComfyUI's `LoadImageMask` (channel `alpha`) treats transparent
pixels as the region to change, so `upload_mask` must send alpha = 255 - luminance. These tests capture the PNG bytes
handed to the HTTP layer.
"""
import io
import re
from typing import Any
from unittest.mock import patch

from PIL import Image

from sd_backend_client.api.comfyui.comfyui_types import ImageFileReference
from sd_backend_client.api.comfyui_webservice import ComfyEndpoints, ComfyUiWebservice


class _FakeResponse:
    """Minimal stand-in for the upload endpoint's HTTP response."""

    def json(self) -> dict[str, Any]:
        """Return a valid upload response body."""
        return {'name': 'mask_src.png', 'subfolder': '', 'type': 'input'}


def _upload_sent(mask: Image.Image, ref_filename: str = 'src.png') -> dict[str, Any]:
    """Upload `mask` through a service with a fake `post`, returning the endpoint and files it was handed."""
    service = ComfyUiWebservice('http://localhost:8188')
    sent: dict[str, Any] = {}

    def fake_post(endpoint: str, **kwargs: Any) -> _FakeResponse:
        sent['endpoint'] = endpoint
        sent['files'] = kwargs['files']
        return _FakeResponse()

    with patch.object(service, 'post', fake_post):
        service.upload_mask(mask, ImageFileReference(filename=ref_filename, subfolder='', type='input'))
    return sent


def _upload(mask: Image.Image) -> tuple[Image.Image, str]:
    """Upload `mask`, returning the decoded uploaded PNG and the endpoint."""
    sent = _upload_sent(mask)
    _name, data, _type = next(iter(sent['files'].values()))
    return Image.open(io.BytesIO(data)).convert('RGBA'), sent['endpoint']


def _uploaded_name(mask: Image.Image, ref_filename: str = 'src.png') -> str:
    name: str = next(iter(_upload_sent(mask, ref_filename)['files'].values()))[0]
    return name


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


def test_mask_name_covers_mask_and_reference_image() -> None:
    """The server saves the reference image with the mask applied, so changing either changes the upload name."""
    name = _uploaded_name(_half_mask(opaque_gray=True))
    assert re.fullmatch(r'mask_[0-9a-f]{32}_src\.png', name)
    assert _uploaded_name(_half_mask(opaque_gray=True)) == name
    assert _uploaded_name(_half_mask(opaque_gray=True), 'other.png') != name
    assert _uploaded_name(Image.new('L', (4, 2), 64)) != name
