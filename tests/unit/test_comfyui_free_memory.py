"""Unit test for ComfyUiWebservice.free_memory's request body, with the HTTP session mocked."""
import json
from typing import Any
from unittest.mock import MagicMock

import requests

from sd_backend_client.api.comfyui_webservice import ComfyUiWebservice


def test_free_memory_sends_json_serializable_body(monkeypatch):
    """free_memory posts a body that `requests` can serialize with json.dumps."""
    service = ComfyUiWebservice('http://unused.invalid')
    sent: list[str] = []

    def fake_post(_session: Any, _address: str, **kwargs: Any) -> MagicMock:
        sent.append(json.dumps(kwargs['json']))
        response = MagicMock()
        response.status_code = 200
        return response

    monkeypatch.setattr(requests.Session, 'post', fake_post)
    service.free_memory()

    assert json.loads(sent[0]) == {'unload_models': True, 'free_memory': True}
