"""Offline tests for ComfyUiWebservice's HTTP requests and response parsing, against a fake `requests.Session`.

The response bodies follow the shapes ComfyUI's /history, /queue, /object_info and /upload endpoints return.
"""
# pylint: disable=protected-access
import io
import json
from typing import Any, Optional
from urllib.parse import parse_qsl, urlsplit
from unittest.mock import MagicMock

import pytest
from PIL import Image

from sd_backend_client.api.comfyui.comfyui_types import ImageFileReference
from sd_backend_client.api.comfyui.comfyui_diffusion_params import ComfyUIDiffusionParams
from sd_backend_client.api.comfyui_webservice import AsyncTaskStatus, ComfyEndpoints, ComfyModelType, ComfyUiWebservice
from sd_backend_client.util.visual.image_utils import image_to_png_bytes

PROMPT_ID = '7d4c0f0e-6a3b-4c1e-9f6e-0a1b2c3d4e5f'
OTHER_ID = '11111111-2222-3333-4444-555555555555'


class _FakeSession:
    """Routes GET/POST by URL path to canned responses and records each request."""

    def __init__(self, routes: Optional[dict[tuple[str, str], Any]] = None) -> None:
        self.routes: dict[tuple[str, str], Any] = routes if routes is not None else {}
        self.requests: list[dict[str, Any]] = []
        self.auth = None

    def _respond(self, method: str, address: str, kwargs: dict[str, Any]) -> MagicMock:
        url = urlsplit(address)
        query = dict(parse_qsl(url.query, keep_blank_values=True))
        self.requests.append({'method': method, 'path': url.path, 'query': query, **kwargs})
        body = self.routes.get((method, url.path), {})
        if isinstance(body, BaseException):
            raise body
        response = MagicMock()
        response.status_code = 200
        if isinstance(body, bytes):
            response.content = body
        else:
            response.json.return_value = json.loads(json.dumps(body))
        return response

    def get(self, address: str, **kwargs: Any) -> MagicMock:
        """Record a GET and return its canned response."""
        return self._respond('GET', address, kwargs)

    def post(self, address: str, **kwargs: Any) -> MagicMock:
        """Record a POST and return its canned response."""
        return self._respond('POST', address, kwargs)

    def of(self, method: str, path: str) -> list[dict[str, Any]]:
        """Recorded requests with the given method and path."""
        return [request for request in self.requests if request['method'] == method and request['path'] == path]


def _service(routes: Optional[dict[tuple[str, str], Any]] = None) -> tuple[ComfyUiWebservice, _FakeSession]:
    service = ComfyUiWebservice('http://127.0.0.1:8188')
    session = _FakeSession(routes)
    service._session = session  # type: ignore[assignment]
    return service, session


def _history_entry(status_str: str = 'success', completed: bool = True,
                   outputs: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """One /history/<prompt_id> entry, keyed by prompt id."""
    if outputs is None:
        outputs = {'9': {'images': [{'filename': 'ComfyUI_00001_.png', 'subfolder': '', 'type': 'output'}]}}
    return {PROMPT_ID: {
        'prompt': [7, PROMPT_ID, {}, {'client_id': 'abc'}, ['9']],
        'outputs': outputs,
        'status': {'status_str': status_str, 'completed': completed,
                   'messages': [['execution_start', {'prompt_id': PROMPT_ID, 'timestamp': 1700000000000}]]},
    }}


def _queue(running: list[tuple[int, str]], pending: list[tuple[int, str]]) -> dict[str, Any]:
    """A /queue response holding (number, prompt_id) entries."""
    return {'queue_running': [[number, prompt_id, {}, {}, ['9']] for number, prompt_id in running],
            'queue_pending': [[number, prompt_id, {}, {}, ['9']] for number, prompt_id in pending]}


# check_queue_entry

def test_finished_history_entry_collects_images_from_every_output_node():
    """A completed entry is FINISHED, with the image references of all output nodes."""
    outputs = {'9': {'images': [{'filename': 'a.png', 'subfolder': '', 'type': 'output'}]},
               '12': {'images': [{'filename': 'b.png', 'subfolder': 'sub', 'type': 'output'}]}}
    service, session = _service({('GET', f'/history/{PROMPT_ID}'): _history_entry(outputs=outputs)})
    progress = service.check_queue_entry(PROMPT_ID, 7)

    assert progress.status is AsyncTaskStatus.FINISHED
    assert progress.outputs is not None
    assert [(ref.filename, ref.subfolder) for ref in progress.outputs.images] == [('a.png', ''), ('b.png', 'sub')]
    assert not session.of('GET', ComfyEndpoints.QUEUE)


def test_error_history_entry_is_failed():
    """An entry whose status_str is 'error' is FAILED."""
    service, _ = _service({('GET', f'/history/{PROMPT_ID}'): _history_entry('error', False, outputs={})})
    assert service.check_queue_entry(PROMPT_ID, 7).status is AsyncTaskStatus.FAILED


def test_running_entry_is_active():
    """A prompt missing from history but in queue_running is ACTIVE."""
    service, _ = _service({('GET', ComfyEndpoints.QUEUE): _queue([(7, PROMPT_ID)], [])})
    assert service.check_queue_entry(PROMPT_ID, 7).status is AsyncTaskStatus.ACTIVE


def test_incomplete_history_entry_falls_through_to_queue():
    """A history entry not yet completed is resolved from the queue."""
    service, _ = _service({('GET', f'/history/{PROMPT_ID}'): _history_entry(completed=False, outputs={}),
                           ('GET', ComfyEndpoints.QUEUE): _queue([(7, PROMPT_ID)], [])})
    assert service.check_queue_entry(PROMPT_ID, 7).status is AsyncTaskStatus.ACTIVE


def test_pending_entry_index_counts_lower_numbers():
    """A pending prompt's index is the count of pending entries with a lower number, in any listed order."""
    service, _ = _service({('GET', ComfyEndpoints.QUEUE): _queue(
        [(3, OTHER_ID)], [(9, 'later'), (7, PROMPT_ID), (5, 'earlier-a'), (4, 'earlier-b')])})
    progress = service.check_queue_entry(PROMPT_ID, 7)
    assert progress.status is AsyncTaskStatus.PENDING
    assert progress.index == 2


def test_unknown_entry_is_not_found():
    """A prompt in neither history nor queue is NOT_FOUND."""
    service, _ = _service({('GET', ComfyEndpoints.QUEUE): _queue([(3, OTHER_ID)], [(9, 'later')])})
    assert service.check_queue_entry(PROMPT_ID, 7).status is AsyncTaskStatus.NOT_FOUND


# Queue control

def test_interrupt_with_task_deletes_it_then_interrupts():
    """interrupt(task_id) deletes the task from the queue and then posts /interrupt."""
    service, session = _service()
    service.interrupt(PROMPT_ID)
    assert [request['path'] for request in session.requests] == [ComfyEndpoints.QUEUE, ComfyEndpoints.INTERRUPT]
    assert session.requests[0]['json'] == {'clear': None, 'delete': [PROMPT_ID]}


def test_remove_from_queue_does_not_interrupt():
    """remove_from_queue only posts the queue deletion."""
    service, session = _service()
    service.remove_from_queue(PROMPT_ID)
    assert [request['path'] for request in session.requests] == [ComfyEndpoints.QUEUE]
    assert session.requests[0]['json']['delete'] == [PROMPT_ID]


# Option discovery

KSAMPLER_INFO = {'KSampler': {
    'input': {'required': {
        'model': ['MODEL'],
        'seed': ['INT', {'default': 0, 'min': 0, 'max': 18446744073709551615}],
        'sampler_name': [['euler', 'dpmpp_2m'], {}],
        'scheduler': [['normal', 'karras'], {}],
    }},
    'input_order': {'required': ['model', 'seed', 'sampler_name', 'scheduler']},
    'output': ['LATENT'], 'output_is_list': [False], 'output_name': ['LATENT'], 'name': 'KSampler',
    'display_name': 'KSampler', 'description': '', 'python_module': 'nodes', 'category': 'sampling',
    'output_node': False,
}}


def test_sampler_and_scheduler_names_come_from_ksampler_info_fetched_once():
    """Sampler and scheduler options are read from /object_info/KSampler, which is cached."""
    service, session = _service({('GET', '/object_info/KSampler'): KSAMPLER_INFO})
    assert service.get_sampler_names() == ['euler', 'dpmpp_2m']
    assert service.get_scheduler_names() == ['normal', 'karras']
    assert len(session.of('GET', '/object_info/KSampler')) == 1


def test_is_node_available_checks_for_node_key():
    """is_node_available is true only when /object_info/<name> returns that node."""
    service, _ = _service({('GET', '/object_info/KSampler'): KSAMPLER_INFO})
    assert service.is_node_available('KSampler')
    assert not service.is_node_available('UltimateSDUpscale')


def test_get_models_requests_the_model_type_folder():
    """get_models lists one model folder."""
    service, session = _service({('GET', '/models/loras'): ['detail.safetensors']})
    assert service.get_models(ComfyModelType.LORA) == ['detail.safetensors']
    assert session.requests[0]['timeout'] is not None


def test_preprocessor_discovery_skips_nodes_that_fail_validation():
    """Nodes in /object_info that don't match NodeInfoResponse are skipped, and the result is cached."""
    canny = {
        'input': {'required': {'image': ['IMAGE'],
                               'low_threshold': ['INT', {'default': 100, 'min': 1, 'max': 255}]}},
        'input_order': {'required': ['image', 'low_threshold']},
        'output': ['IMAGE'], 'output_is_list': [False], 'output_name': ['IMAGE'], 'name': 'CannyEdgePreprocessor',
        'display_name': 'Canny Edge', 'description': '', 'python_module': 'custom_nodes.controlnet_aux',
        'category': 'ControlNet Preprocessors/Line Extractors', 'output_node': False,
    }
    service, session = _service({('GET', ComfyEndpoints.OBJECT_INFO): {'CannyEdgePreprocessor': canny,
                                                                      'BrokenNode': {'name': 'BrokenNode'}}})
    names = [preprocessor.name for preprocessor in service.get_controlnet_preprocessors()]
    assert names == ['CannyEdgePreprocessor']
    service.get_controlnet_preprocessors()
    assert len(session.of('GET', ComfyEndpoints.OBJECT_INFO)) == 1


# Uploads and downloads

def _upload_response(name: str = 'src_image.png') -> dict[str, str]:
    return {'name': name, 'subfolder': 'IntraPaint', 'type': 'input'}


def test_upload_image_sends_png_multipart_and_caches_by_content():
    """upload_image posts a PNG as multipart form data, and the same image again reuses the first reference."""
    service, session = _service({('POST', ComfyEndpoints.IMG_UPLOAD): _upload_response()})
    image = Image.new('RGBA', (3, 2), (1, 2, 3, 255))
    reference = service.upload_image(image)
    again = service.upload_image(image.copy())

    assert reference == again == ImageFileReference(filename='src_image.png', subfolder='IntraPaint', type='input')
    uploads = session.of('POST', ComfyEndpoints.IMG_UPLOAD)
    assert len(uploads) == 1
    assert uploads[0]['data'] == {'type': 'input', 'subfolder': 'IntraPaint', 'overwrite': '1'}
    name, data, content_type = uploads[0]['files']['image']
    assert (name, content_type) == ('src_image.png', 'image/png')
    assert Image.open(io.BytesIO(data)).size == (3, 2)


def test_upload_image_adds_png_extension_to_name():
    """A name without .png gets one."""
    service, session = _service({('POST', ComfyEndpoints.IMG_UPLOAD): _upload_response('control_0.png')})
    service.upload_image(Image.new('RGBA', (1, 1)), 'control_0')
    assert session.requests[0]['files']['image'][0] == 'control_0.png'


def test_upload_image_rejects_non_base64_string():
    """A string that is neither a file path nor base64 raises ValueError."""
    service, _ = _service()
    with pytest.raises(ValueError, match='base64'):
        service.upload_image('not base64!')


def test_download_images_sends_reference_as_query_and_skips_failures():
    """Each reference becomes /view query parameters, and an image that fails to download is skipped."""
    png = image_to_png_bytes(Image.new('RGB', (2, 2), (255, 0, 0)))
    service, session = _service({('GET', ComfyEndpoints.VIEW_IMAGE): png})
    refs = [ImageFileReference(filename='a.png', subfolder='', type='output'),
            ImageFileReference(filename='b.png', subfolder='sub')]
    images = service.download_images(refs)

    assert len(images) == 2
    assert images[0].mode == 'RGBA'
    assert [request['query'] for request in session.requests] == [
        {'filename': 'a.png', 'subfolder': '', 'type': 'output'},
        {'filename': 'b.png', 'subfolder': 'sub'}]

    session.routes[('GET', ComfyEndpoints.VIEW_IMAGE)] = RuntimeError('404: not found')
    assert not service.download_images(refs)


# Generation requests

def test_txt2img_posts_workflow_with_client_id_and_returns_seed():
    """txt2img queues the built workflow under this client's id and reports the seed it used."""
    service, session = _service({('GET', '/models/configs'): [],
                                 ('POST', ComfyEndpoints.PROMPT): {'prompt_id': PROMPT_ID, 'number': 7,
                                                                   'node_errors': {}}})
    params = ComfyUIDiffusionParams(prompt='a cat', sd_model_name='model.safetensors', seed=1234,
                                    denoising_strength=0.4)
    response = service.txt2img(params)

    assert (response.prompt_id, response.number, response.seed) == (PROMPT_ID, 7, 1234)
    body = session.of('POST', ComfyEndpoints.PROMPT)[0]['json']
    assert body['client_id'] == service._client_id
    samplers = [node for node in body['prompt'].values() if node['class_type'] == 'KSampler']
    assert len(samplers) == 1
    assert samplers[0]['inputs']['seed'] == 1234
    assert samplers[0]['inputs']['denoise'] == 1.0


def test_img2img_uploads_init_image_and_wires_it_into_the_workflow():
    """img2img uploads the init image and loads the uploaded reference in the graph."""
    service, session = _service({('GET', '/models/configs'): [],
                                 ('POST', ComfyEndpoints.IMG_UPLOAD): _upload_response(),
                                 ('POST', ComfyEndpoints.PROMPT): {'prompt_id': PROMPT_ID, 'number': 1}})
    params = ComfyUIDiffusionParams(sd_model_name='model.safetensors', init_images=[Image.new('RGBA', (8, 8))])
    service.img2img(params)

    assert len(session.of('POST', ComfyEndpoints.IMG_UPLOAD)) == 1
    prompt = session.of('POST', ComfyEndpoints.PROMPT)[0]['json']['prompt']
    load_nodes = [node for node in prompt.values() if node['class_type'] == 'LoadImage']
    assert [node['inputs']['image'] for node in load_nodes] == ['IntraPaint/src_image.png']


def test_img2img_and_inpaint_require_their_inputs():
    """img2img needs an init image; inpaint also needs a mask. Neither contacts the server when they are missing."""
    service, session = _service()
    with pytest.raises(ValueError, match='init image'):
        service.img2img(ComfyUIDiffusionParams())
    with pytest.raises(ValueError, match='mask'):
        service.inpaint(ComfyUIDiffusionParams(init_images=[Image.new('RGBA', (8, 8))]))
    assert not session.requests


# Websocket progress parsing

@pytest.mark.parametrize('message, expected', [
    ('{"type": "progress", "data": {"value": 5, "max": 20}}', 25.0),
    ('{"type": "status", "data": {"status": {}}}', None),
    ('{"type": "progress", "data": {}}', None),
    ('not json', None),
])
def test_parse_percentage_from_websocket_message(message: str, expected: Optional[float]):
    """Only progress messages with value and max produce a percentage."""
    assert ComfyUiWebservice.parse_percentage_from_websocket_message(message) == expected
