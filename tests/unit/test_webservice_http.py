"""Offline tests for WebService's request building and error mapping, with `requests.Session` mocked out."""
# pylint: disable=protected-access,missing-function-docstring
from unittest.mock import MagicMock

import pytest
import requests

from intrapaint_api.api.webservice import JSON_DATA_TYPE, MULTIPART_FORM_DATA_TYPE, WebService


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
    service._session.get.assert_called_once_with('http://host:7860/sdapi/v1/options', timeout=12,
                                                 headers={'X-Test': '1'})


def test_get_appends_url_params_as_query_string():
    service = _service()
    service.get('/view', url_params={'filename': 'a.png', 'type': 'output'})
    assert service._session.get.call_args.args[0] == 'http://host:7860/view?filename=a.png&type=output'


def test_post_json_body_is_sent_as_json():
    service = _service()
    service.post('/prompt', {'prompt': {}}, timeout=5)
    service._session.post.assert_called_once_with('http://host:7860/prompt', timeout=5, headers={},
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
