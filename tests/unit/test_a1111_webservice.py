"""Unit tests for the request bodies A1111Webservice builds inline, with `post` replaced so no server is needed."""
from typing import Any

from PIL import Image

from intrapaint_api.api.a1111_webservice import A1111Webservice
from intrapaint_api.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor
from intrapaint_api.util.visual.image_utils import image_from_base64, image_to_base64


class _FakeResponse:
    """Minimal stand-in for a successful `requests.Response` carrying one image."""
    status_code = 200

    def json(self) -> dict[str, Any]:
        """Return a response body holding a single 1x1 image."""
        return {'images': [image_to_base64(Image.new('RGBA', (1, 1)))]}


def test_preprocessor_preview_sends_mask_as_opaque_grayscale(monkeypatch):
    """The preview request's mask is converted with mask_to_grayscale, like the img2img mask."""
    service = A1111Webservice('http://unused.invalid')
    sent_bodies: list[dict[str, Any]] = []

    def fake_post(_endpoint: str, body: dict[str, Any], *_args: Any, **_kwargs: Any) -> _FakeResponse:
        sent_bodies.append(body)
        return _FakeResponse()

    monkeypatch.setattr(service, 'post', fake_post)
    image = Image.new('RGBA', (2, 1), (10, 20, 30, 255))
    mask = Image.new('RGBA', (2, 1), (0, 0, 0, 0))
    mask.putpixel((0, 0), (255, 0, 0, 255))
    service.controlnet_preprocessor_preview(image, mask, ControlNetPreprocessor(name='inpaint_only+lama'))

    input_images = sent_bodies[0]['controlnet_input_images']
    assert len(input_images) == 2
    emitted_mask = image_from_base64(input_images[1])
    assert emitted_mask.getpixel((0, 0)) == (255, 255, 255, 255)
    assert emitted_mask.getpixel((1, 0)) == (0, 0, 0, 255)
