"""
Basic interface for classes used to access HTTP web services. 

Provides basic session management, auth access, and functions for making GET and POST requests.
"""
from typing import Optional, Any
import secrets
from urllib.parse import urlsplit, urlunsplit
import requests

from sd_backend_client.errors import AuthError, BackendConnectionError, BackendTimeoutError, ServerError

JSON_DATA_TYPE = 'application/json'
MULTIPART_FORM_DATA_TYPE = 'multipart/form-data'
DEFAULT_REQUEST_TIMEOUT = 30.0


class WebService:
    """
    WebService establishes a connection to a URL, handles basic auth, and provides convenience methods for GET and
    POST requests.
    """

    def __init__(self, url: str, request_timeout: Optional[float] = DEFAULT_REQUEST_TIMEOUT):
        """__init__.

        Parameters
        ----------
        url : str
            Base URL of the webservice. Trailing slashes are stripped, since endpoints start with '/'.
        request_timeout : float, optional, default=DEFAULT_REQUEST_TIMEOUT
            Timeout in seconds for requests that don't pass their own. None waits indefinitely.
        """
        self._server_url = url.rstrip('/')
        self._request_timeout = request_timeout
        self._session = requests.Session()
        self._auth = None
        self._session_hash = secrets.token_hex(5)

    @property
    def server_url(self) -> str:
        """Returns the server URL, without a trailing slash."""
        return self._server_url

    @property
    def request_timeout(self) -> Optional[float]:
        """Timeout in seconds for requests that don't pass their own, or None to wait indefinitely."""
        return self._request_timeout

    @property
    def websocket_url(self) -> str:
        """The server URL with its scheme mapped to a websocket scheme: https to wss, anything else to ws."""
        url = urlsplit(self._server_url)
        scheme = 'wss' if url.scheme == 'https' else 'ws'
        return urlunsplit((scheme, url.netloc, url.path, '', ''))

    def set_auth(self, auth):
        """Set session authentication.

        Parameters
        ----------
        auth : (str, str)
            A (username, password) pair, or any other auth format supported by the requests library and the particular
            webservice being targeted.
        """
        self._session.auth = auth

    def get(self,
            endpoint: str,
            timeout: Optional[float] = None,
            url_params: Optional[dict[str, str]] = None,
            headers: Optional[dict[str, str]] = None,
            fail_on_auth_error: bool = False,
            throw_on_failure: bool = True) -> requests.Response:
        """Sends an HTTP GET request to the webservice

        Parameters
        ----------
        endpoint : str
            String appended to the end of the service's base URL.
        timeout : float, optional
            Request timeout period in seconds. None uses the service's request_timeout.
        url_params : dict, optional
            Any URL parameters to send with the request.
        headers : dict, optional
            Any headers that should be explicitly set on the request.
        fail_on_auth_error : bool, default=false
            Whether a 401: unauthorized response should raise AuthError instead of starting the auth flow.
        throw_on_failure : bool, default=true
            Whether other responses with failure statuses should raise ServerError.

        Returns
        -------
        Response
            The response returned by the webservice.

        Raises
        ------
        BackendConnectionError
            If the server could not be reached.
        BackendTimeoutError
            If the server did not respond within the timeout.
        AuthError
            If the server rejected authentication.
        ServerError
            If the server returned another failure status and throw_on_failure is set.
        """
        return self._send(endpoint, 'GET', None, None, timeout, url_params, headers, None,
                          fail_on_auth_error, throw_on_failure)

    def post(self,
             endpoint: str,
             body: Any,
             body_format: Optional[str] = 'application/json',
             timeout: Optional[float] = None,
             url_params: Optional[dict[str, str]] = None,
             headers: Optional[dict[str, str]] = None,
             files: Optional[dict[str, tuple[str, bytes, str]]] = None,
             fail_on_auth_error: bool = False,
             throw_on_failure: bool = True) -> requests.Response:
        """Sends an HTTP POST request to the webservice

        Parameters
        ----------
        endpoint : str
            String appended to the end of the service's base URL.
        body: any
            Data to send to the webservice. Any format supported by the request library is accepted, but it should
            be one that's valid for the body_format parameter used.
        body_format: Optional[str], default='application/json'
            Request content format to use.
        timeout : float, optional
            Request timeout period in seconds. None uses the service's request_timeout.
        url_params : dict[str, str], optional
            Any URL parameters to send with the request.
        headers : dict[str, str], optional
            Any headers that should be explicitly set on the request.
        files: dict[str, tuple[str, bytes, str]], optional
            Files that should be sent with form data, to be used with body type 'multipart/form-data'. Tuple format is
            (filename, file_bytes, file_type_str).
        fail_on_auth_error : bool, default=false
            Whether a 401: unauthorized response should raise AuthError instead of starting the auth flow.
        throw_on_failure : bool, default=true
            Whether other responses with failure statuses should raise ServerError.

        Returns
        -------
        Response
            The response returned by the webservice.

        Raises
        ------
        BackendConnectionError
            If the server could not be reached.
        BackendTimeoutError
            If the server did not respond within the timeout.
        AuthError
            If the server rejected authentication.
        ServerError
            If the server returned another failure status and throw_on_failure is set.
        """
        if body is None:
            body_format = None
        return self._send(endpoint, 'POST', body, body_format, timeout, url_params, headers, files,
                          fail_on_auth_error, throw_on_failure)

    def _send(self,
              endpoint: str,
              method: str,
              body,
              body_format: Optional[str] = JSON_DATA_TYPE,
              timeout: Optional[float] = None,
              url_params: Optional[dict[str, str]] = None,
              headers: Optional[dict[str, str]] = None,
              files: Optional[dict[str, tuple[str, bytes, str]]] = None,
              fail_on_auth_error: bool = False,
              throw_on_failure: bool = True,
              _auth_retried: bool = False) -> requests.Response:
        address = self._build_address(endpoint)
        if headers is None:
            headers = {}
        request_timeout = self._request_timeout if timeout is None else timeout
        try:
            if method == 'GET':
                res = self._session.get(address, params=url_params, timeout=request_timeout, headers=headers)
            elif method == 'POST':
                if body_format == JSON_DATA_TYPE:
                    res = self._session.post(address, params=url_params, timeout=request_timeout, headers=headers,
                                             json=body)
                elif body_format == MULTIPART_FORM_DATA_TYPE and files is not None:
                    res = self._session.post(address, params=url_params, timeout=request_timeout, headers=headers,
                                             files=files, data=body)
                else:
                    res = self._session.post(address, params=url_params, timeout=request_timeout, headers=headers,
                                             data=body)
            else:
                raise ValueError(f'HTTP method {method} not supported')
        except requests.exceptions.Timeout as err:
            raise BackendTimeoutError(f'{method} {endpoint} timed out after {request_timeout}s: {err}') from err
        except requests.exceptions.RequestException as err:
            raise BackendConnectionError(f'Error connecting to {address}: {err}') from err
        if res.status_code == 401:
            if fail_on_auth_error and throw_on_failure:
                raise AuthError(f'{method} {endpoint} failed with status 401: unauthorized')
            if not fail_on_auth_error:
                if _auth_retried:
                    raise AuthError(f'HTTP method {method} to {endpoint} still failed with status 401 after '
                                    'authenticating. Check the credentials and the server\'s auth settings.')
                self._handle_auth_error()
                return self._send(endpoint,
                                  method,
                                  body,
                                  body_format,
                                  timeout,
                                  url_params,
                                  headers,
                                  files,
                                  fail_on_auth_error,
                                  throw_on_failure,
                                  _auth_retried=True)
        elif res.status_code != 200 and throw_on_failure:
            raise ServerError(res.status_code, res.text, endpoint, method)
        return res

    def disconnect(self) -> None:
        """Close the session and clear auth.  Do not use the webservice after calling this."""
        self._session.close()
        self._auth = None

    def _handle_auth_error(self):
        raise NotImplementedError('Authentication is not implemented')

    def _build_address(self, endpoint: str) -> str:
        """Joins the server URL and an endpoint path. Query parameters are passed to `requests` separately, which
        encodes them."""
        return f'{self._server_url}{endpoint}'
