"""Offline replay of the server responses `scripts/capture_fixtures.py` records into `fixtures/recorded/`.

`replay` swaps a client's `get` for a lookup into one recording, so tests can call the client's real discovery and
parsing methods without a server.
"""
import json
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import pytest

from sd_backend_client.api.webservice import WebService
from sd_backend_client.errors import ServerError

RECORDED_DIR = Path(__file__).parent / 'fixtures' / 'recorded'


def recordings(backend: str) -> list[Any]:
    """pytest params for every recording of one backend, with the file name as the test id."""
    params = []
    for path in sorted(RECORDED_DIR.glob('*.json')):
        recording = json.loads(path.read_text(encoding='utf-8'))
        if recording['meta']['backend'] == backend:
            params.append(pytest.param(recording, id=path.stem))
    return params


def endpoint_key(endpoint: str, url_params: Optional[dict[str, str]] = None) -> str:
    """The key a response is recorded under. Must match `endpoint_key` in scripts/capture_fixtures.py."""
    if not url_params:
        return endpoint
    return f'{endpoint}?{urlencode(sorted(url_params.items()))}'


class RecordedResponse:
    """Stand-in for a successful `requests.Response` carrying one recorded JSON body."""
    status_code = 200
    ok = True

    def __init__(self, body: Any) -> None:
        self._body = body

    def json(self) -> Any:
        """Return a fresh copy of the recorded body, as each `Response.json()` call parses anew."""
        return json.loads(json.dumps(self._body))


def replay(service: WebService, recording: dict[str, Any]) -> WebService:
    """Replace service.get with a lookup into the recording; an unrecorded endpoint fails the test."""
    responses: dict[str, Any] = recording['responses']
    errors: dict[str, int] = recording.get('errors', {})

    def recorded_get(endpoint: str, timeout: Optional[float] = None, url_params: Optional[dict[str, str]] = None,
                     **_kwargs: Any) -> RecordedResponse:
        del timeout
        key = endpoint_key(endpoint, url_params)
        if key in errors:
            raise ServerError(errors[key], '', endpoint, 'GET')
        if key not in responses:
            pytest.fail(f'{key} is not in the recording; re-run scripts/capture_fixtures.py')
        return RecordedResponse(responses[key])

    service.get = recorded_get  # type: ignore[method-assign, assignment]
    return service
