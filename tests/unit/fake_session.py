"""A fake `requests.Session` for offline client tests: canned responses routed by method and URL path."""
import json
from typing import Any, Optional
from urllib.parse import parse_qsl, urlsplit
from unittest.mock import MagicMock

import requests


class CannedStatus:
    """A canned failure response: a status code and a raw text body."""

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class FakeSession:
    """Routes GET/POST by URL path to canned responses and records each request.

    A route whose value is callable is called for each request, so a test can change the response between calls.
    """

    def __init__(self, routes: Optional[dict[tuple[str, str], Any]] = None) -> None:
        self.routes: dict[tuple[str, str], Any] = routes if routes is not None else {}
        self.requests: list[dict[str, Any]] = []
        self.auth = None

    def _respond(self, method: str, address: str, kwargs: dict[str, Any]) -> MagicMock:
        url = urlsplit(requests.Request(method, address, params=kwargs.get('params')).prepare().url)
        query = dict(parse_qsl(url.query, keep_blank_values=True))
        self.requests.append({'method': method, 'path': url.path, 'query': query, **kwargs})
        body = self.routes.get((method, url.path), {})
        if callable(body):
            body = body()
        if isinstance(body, BaseException):
            raise body
        response = MagicMock()
        response.status_code = 200
        if isinstance(body, CannedStatus):
            response.status_code = body.status_code
            response.text = body.text
        elif isinstance(body, bytes):
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

    def close(self) -> None:
        """Do nothing, as a closed fake keeps answering."""

    def of(self, method: str, path: str) -> list[dict[str, Any]]:
        """Recorded requests with the given method and path."""
        return [request for request in self.requests if request['method'] == method and request['path'] == path]
