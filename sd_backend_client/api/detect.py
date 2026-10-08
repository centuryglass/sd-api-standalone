"""Picks the client for a server from its URL alone, so callers can start without knowing which backend runs there.

ComfyUI is recognized by `/system_stats` and WebUI (A1111 or Forge) by `/sdapi/v1/options`. ComfyUI is probed first
because it has no auth, while WebUI's probe can start the `credentials_provider` login flow.
"""
from typing import Any, Callable, Optional

import requests

from sd_backend_client.api.a1111_webservice import A1111Webservice
from sd_backend_client.api.comfyui_webservice import ComfyEndpoints, ComfyUiWebservice
from sd_backend_client.api.shared_data.backend import Backend
from sd_backend_client.api.webservice import DEFAULT_REQUEST_TIMEOUT
from sd_backend_client.errors import UnexpectedResponseError

__all__ = ['connect_to_backend']


def connect_to_backend(url: str,
                       credentials_provider: Optional[Callable[[], Optional[tuple[str, str]]]] = None,
                       request_timeout: Optional[float] = DEFAULT_REQUEST_TIMEOUT) -> Backend:
    """Return a `ComfyUiWebservice` or `A1111Webservice` for the server at `url`, whichever it answers as.

    The returned client has its constructor's defaults apart from these arguments. Construct a client directly to
    set other options.

    Parameters
    ----------
    url: str
        Base URL of the server.
    credentials_provider: optional callable
        Passed to `A1111Webservice`, which calls it if the server requires authentication (see its constructor).
        ComfyUI ignores it.
    request_timeout: float, optional, default=DEFAULT_REQUEST_TIMEOUT
        Timeout in seconds for each probe and for the returned client's non-generation requests.

    Raises
    ------
    BackendConnectionError
        If nothing answers at `url`.
    BackendTimeoutError
        If the server does not answer a probe in time.
    AuthError
        If the server is a WebUI that requires authentication and the credentials are missing or rejected.
    UnexpectedResponseError
        If the server answers neither probe. A WebUI started without `--api` answers this way.
    """
    comfyui = ComfyUiWebservice(url, request_timeout)
    stats = comfyui.get(ComfyEndpoints.SYSTEM_STATS, fail_on_auth_error=True, throw_on_failure=False)
    if 'system' in _json_object(stats):
        return comfyui
    comfyui.disconnect()

    webui = A1111Webservice(url, credentials_provider=credentials_provider, request_timeout=request_timeout)
    options = webui.get(A1111Webservice.Endpoints.OPTIONS, throw_on_failure=False)
    if _json_object(options):
        return webui
    webui.disconnect()
    raise UnexpectedResponseError(f'{url} is not a ComfyUI or WebUI server: {ComfyEndpoints.SYSTEM_STATS} returned '
                                  f'status {stats.status_code} and {A1111Webservice.Endpoints.OPTIONS} returned status '
                                  f'{options.status_code}. Start WebUI with --api to enable its API.')


def _json_object(response: requests.Response) -> dict[str, Any]:
    """`response`'s body if it is a successful JSON object, else an empty dict."""
    if response.status_code != 200:
        return {}
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}
