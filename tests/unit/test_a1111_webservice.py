"""Unit tests for the request bodies A1111Webservice builds inline, with `post` replaced so no server is needed."""
from typing import Any

from PIL import Image

from intrapaint_api.api.a1111_webservice import A1111Webservice
from intrapaint_api.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor
from intrapaint_api.api.webui.diffusion_request_body import DiffusionRequestBody
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


def _service_with_stubbed_post(monkeypatch) -> tuple[A1111Webservice, list[tuple[str, dict[str, Any]]]]:
    """A service whose `post` records (endpoint, body) and whose dispatcher skips the server idle wait."""
    service = A1111Webservice('http://unused.invalid')
    sent: list[tuple[str, dict[str, Any]]] = []

    def fake_post(endpoint: str, body: dict[str, Any], *_args: Any, **_kwargs: Any) -> _FakeResponse:
        sent.append((endpoint, body))
        return _FakeResponse()

    monkeypatch.setattr(service, 'post', fake_post)
    service._generation_dispatcher._wait_for_idle = False  # pylint: disable=protected-access
    return service, sent


def test_submit_txt2img_reuses_body_with_distinct_task_ids(monkeypatch):
    """Submitting one body repeatedly gives each job its own task id and leaves the caller's body unchanged."""
    service, sent = _service_with_stubbed_post(monkeypatch)
    body = DiffusionRequestBody()
    handles = [service.submit_txt2img(body) for _ in range(3)]
    for handle in handles:
        handle.wait()
    assert body.force_task_id is None
    task_ids = [handle.task_id for handle in handles]
    assert len(set(task_ids)) == 3
    assert sorted(sent_body['force_task_id'] for _, sent_body in sent) == sorted(task_ids)


def test_submit_txt2img_snapshots_body_at_submit_time(monkeypatch):
    """Changing the caller's body after submit does not change the job's request."""
    service, sent = _service_with_stubbed_post(monkeypatch)
    body = DiffusionRequestBody(steps=7)
    handle = service.submit_txt2img(body)
    body.steps = 99
    handle.wait()
    assert sent[0][1]['steps'] == 7


def test_submit_img2img_reuse_with_different_images(monkeypatch):
    """Reusing one body for several img2img submissions sends each call's own image and leaves the body unchanged."""
    service, sent = _service_with_stubbed_post(monkeypatch)
    body = DiffusionRequestBody()
    colors = [(255, 0, 0, 255), (0, 255, 0, 255)]
    handles = [service.submit_img2img(Image.new('RGBA', (2, 2), color), None, body) for color in colors]
    for handle in handles:
        handle.wait()
    assert body.init_images is None
    assert body.mask is None
    assert body.force_task_id is None
    emitted = sorted(image_from_base64(sent_body['init_images'][0]).getpixel((0, 0)) for _, sent_body in sent)
    assert emitted == sorted(colors)
    assert len({sent_body['force_task_id'] for _, sent_body in sent}) == 2


def test_img2img_does_not_modify_request_body(monkeypatch):
    """The blocking img2img leaves the caller's body without the image and mask it was called with."""
    service, sent = _service_with_stubbed_post(monkeypatch)
    body = DiffusionRequestBody()
    service.img2img(Image.new('RGBA', (2, 2)), Image.new('RGBA', (2, 2)), body)
    assert body.init_images is None
    assert body.mask is None
    assert len(sent[0][1]['init_images']) == 1
    assert 'mask' in sent[0][1]
