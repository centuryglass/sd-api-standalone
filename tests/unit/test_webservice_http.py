"""Offline tests for WebService's request building and error mapping, with `requests.Session` mocked out."""
# pylint: disable=protected-access,missing-function-docstring
from unittest.mock import MagicMock

import pytest
import requests

from sd_backend_client.api.webservice import DEFAULT_REQUEST_TIMEOUT, JSON_DATA_TYPE, MULTIPART_FORM_DATA_TYPE, \
    WebService


def _response(status: int = 200, text: str = '') -> MagicMock:
    res = MagicMock()
    res.status_code = status
    res.text = text
    return res


def _service(status: int = 200, text: str = '') -> WebService:
    service = WebService('http://host:7860')
    service._session = MagicMock()
    service._session.get.return_value = _response(status, text)
    service._session.post.return_value = _response(status, text)
    return service


def test_get_joins_endpoint_and_passes_timeout_and_headers():
    service = _service()
    service.get('/sdapi/v1/options', timeout=12, headers={'X-Test': '1'})
    service._session.get.assert_called_once_with('http://host:7860/sdapi/v1/options', params=None, timeout=12,
                                                 headers={'X-Test': '1'})


def test_get_passes_url_params_to_requests():
    service = _service()
    service.get('/view', url_params={'filename': 'a.png', 'type': 'output'})
    assert service._session.get.call_args.args[0] == 'http://host:7860/view'
    assert service._session.get.call_args.kwargs['params'] == {'filename': 'a.png', 'type': 'output'}


def test_url_params_with_reserved_characters_are_encoded():
    """The real Session prepares the request, so this checks the URL that would go on the wire."""
    service = WebService('http://host:7860')
    prepared: list[requests.PreparedRequest] = []

    def fake_send(request: requests.PreparedRequest, **_kwargs) -> MagicMock:
        prepared.append(request)
        return _response()

    service._session.send = fake_send  # type: ignore[method-assign]
    service.get('/sd_extra_networks/thumb', url_params={'filename': 'my lora&v2#1?.png'})
    assert prepared[0].url == 'http://host:7860/sd_extra_networks/thumb?filename=my+lora%26v2%231%3F.png'


@pytest.mark.parametrize('url', ['http://host:7860/', 'http://host:7860//'])
def test_trailing_slashes_are_stripped_from_the_server_url(url: str):
    service = WebService(url)
    service._session = MagicMock()
    service._session.get.return_value = _response()
    service.get('/sdapi/v1/options')
    assert service.server_url == 'http://host:7860'
    assert service._session.get.call_args.args[0] == 'http://host:7860/sdapi/v1/options'


def test_requests_without_a_timeout_use_the_constructor_default():
    service = _service()
    service.get('/x')
    service.post('/y', {})
    assert service._session.get.call_args.kwargs['timeout'] == DEFAULT_REQUEST_TIMEOUT
    assert service._session.post.call_args.kwargs['timeout'] == DEFAULT_REQUEST_TIMEOUT


def test_constructor_timeout_is_overridable():
    service = WebService('http://host:7860', request_timeout=None)
    service._session = MagicMock()
    service._session.get.return_value = _response()
    service.get('/x')
    assert service._session.get.call_args.kwargs['timeout'] is None
    service.get('/x', timeout=3)
    assert service._session.get.call_args.kwargs['timeout'] == 3


@pytest.mark.parametrize('url, expected', [
    ('http://host:8188', 'ws://host:8188'),
    ('https://host', 'wss://host'),
    ('https://host:443/comfy/', 'wss://host:443/comfy'),
    ('HTTPS://host', 'wss://host'),
])
def test_websocket_url_maps_scheme_and_keeps_path(url: str, expected: str):
    assert WebService(url).websocket_url == expected


def test_post_json_body_is_sent_as_json():
    service = _service()
    service.post('/prompt', {'prompt': {}}, timeout=5)
    service._session.post.assert_called_once_with('http://host:7860/prompt', params=None, timeout=5, headers={},
                                                  json={'prompt': {}})


def test_post_multipart_sends_files_and_form_fields():
    service = _service()
    files = {'image': ('a.png', b'png-bytes', 'image/png')}
    service.post('/upload/image', {'type': 'input'}, body_format=MULTIPART_FORM_DATA_TYPE, files=files)
    kwargs = service._session.post.call_args.kwargs
    assert kwargs['files'] == files
    assert kwargs['data'] == {'type': 'input'}
    assert 'json' not in kwargs


def test_post_without_body_sends_no_json():
    service = _service()
    service.post('/interrupt', None)
    kwargs = service._session.post.call_args.kwargs
    assert kwargs['data'] is None
    assert 'json' not in kwargs


def test_post_other_format_sends_raw_data():
    service = _service()
    service.post('/raw', 'payload', body_format='text/plain')
    assert service._session.post.call_args.kwargs['data'] == 'payload'


def test_unsupported_method_raises_value_error():
    with pytest.raises(ValueError, match='PUT'):
        _service()._send('/x', 'PUT', None, JSON_DATA_TYPE)


def test_error_status_raises_with_status_and_body():
    service = _service(500, 'CUDA out of memory')
    with pytest.raises(RuntimeError, match='500: CUDA out of memory'):
        service.get('/x')


def test_error_status_is_returned_when_throw_on_failure_is_off():
    service = _service(404, 'missing')
    assert service.get('/x', throw_on_failure=False).status_code == 404


def test_connection_error_is_wrapped_with_endpoint():
    service = _service()
    service._session.get.side_effect = requests.exceptions.ConnectionError('refused')
    with pytest.raises(RuntimeError, match='Error connecting to /x') as error:
        service.get('/x')
    assert isinstance(error.value.__cause__, requests.exceptions.ConnectionError)


def test_timeout_is_wrapped_with_endpoint():
    service = _service()
    service._session.post.side_effect = requests.exceptions.ReadTimeout('read timed out')
    with pytest.raises(RuntimeError, match='Error connecting to /x: read timed out'):
        service.post('/x', {})


def test_fail_on_auth_error_raises_without_reauthenticating():
    service = _service(401)
    service._handle_auth_error = MagicMock()
    with pytest.raises(RuntimeError, match='401'):
        service.get('/x', fail_on_auth_error=True)
    service._handle_auth_error.assert_not_called()


def test_fail_on_auth_error_without_throw_returns_401_response():
    service = _service(401)
    service._handle_auth_error = MagicMock()
    assert service.get('/x', fail_on_auth_error=True, throw_on_failure=False).status_code == 401
    service._handle_auth_error.assert_not_called()


def test_base_class_has_no_auth_handler():
    with pytest.raises(NotImplementedError):
        _service(401).get('/x')


def test_set_auth_and_disconnect_act_on_the_session():
    service = _service()
    service.set_auth(('user', 'pass'))
    assert service._session.auth == ('user', 'pass')
    service.disconnect()
    service._session.close.assert_called_once()
