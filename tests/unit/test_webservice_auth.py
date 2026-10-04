"""Offline tests that the 401 retry paths in WebService and A1111Webservice terminate."""
# pylint: disable=protected-access,missing-function-docstring
from unittest.mock import MagicMock

import pytest

from intrapaint_api.api.a1111_webservice import A1111Webservice, AuthError, MAX_LOGIN_ATTEMPTS
from intrapaint_api.api.webservice import WebService


def _response(status: int) -> MagicMock:
    res = MagicMock()
    res.status_code = status
    res.ok = 200 <= status < 300
    res.text = ''
    return res


class _CountingProvider:
    def __init__(self, credentials=('user', 'password')):
        self.calls = 0
        self.credentials = credentials

    def __call__(self):
        self.calls += 1
        return self.credentials


def test_wrong_credentials_raise_after_bounded_attempts():
    provider = _CountingProvider()
    service = A1111Webservice('http://test', credentials_provider=provider)
    service._session.get = MagicMock(return_value=_response(401))
    service._session.post = MagicMock(return_value=_response(401))
    with pytest.raises(AuthError, match='401'):
        service.get_config()
    assert provider.calls == MAX_LOGIN_ATTEMPTS
    assert service._session.auth is None


def test_accepted_credentials_are_installed_and_request_retried():
    provider = _CountingProvider()
    service = A1111Webservice('http://test', credentials_provider=provider)

    def fake_get(_address, **_kwargs):
        return _response(200 if service._session.auth == ('user', 'password') else 401)

    service._session.get = MagicMock(side_effect=fake_get)
    service.get_config()
    assert provider.calls == 1
    assert service._session.auth == ('user', 'password')


def test_provider_returning_none_aborts():
    service = A1111Webservice('http://test', credentials_provider=lambda: None)
    service._session.get = MagicMock(return_value=_response(401))
    with pytest.raises(AuthError):
        service.get_config()


def test_missing_provider_raises():
    service = A1111Webservice('http://test')
    service._session.get = MagicMock(return_value=_response(401))
    with pytest.raises(AuthError, match='credentials_provider'):
        service.get_config()


def test_send_retries_once_when_auth_handler_succeeds_but_api_still_401():
    class _Service(WebService):
        handled = 0

        def _handle_auth_error(self):
            self.handled += 1

    service = _Service('http://test')
    service._session.get = MagicMock(return_value=_response(401))
    with pytest.raises(AuthError, match='401'):
        service.get('/x')
    assert service.handled == 1
    assert service._session.get.call_count == 2
