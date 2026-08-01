"""Authentication integration tests.

These validate the ``/login`` flow and the ``credentials_provider`` callback wiring.
Behavior differs depending on whether the server was launched with ``--api-auth``:
the ``auth_enforced`` fixture probes that, and each test asserts the appropriate
outcome (or skips) for both configurations, so the suite is correct against an
authenticated *and* an unauthenticated instance.
"""
import pytest

from intrapaint_api.api.a1111_webservice import A1111Webservice, AuthError

pytestmark = pytest.mark.integration


def test_login_with_valid_credentials(api_url, credentials, auth_enforced):
    if not auth_enforced:
        pytest.skip('Server does not enforce API auth (no --api-auth); login is a no-op.')
    if credentials is None:
        pytest.fail('Server enforces auth but SD_UNAME / SD_PASS are not set.')
    client = A1111Webservice(api_url)
    try:
        res = client.login(*credentials)
        assert res.status_code == 200, f'valid login rejected ({res.status_code}): {res.text}'
    except RuntimeError as err:
        pytest.skip(f'No WebUI server reachable at {api_url}: {err}')
    finally:
        client.disconnect()


def test_login_with_bad_credentials_is_rejected(api_url, auth_enforced):
    if not auth_enforced:
        pytest.skip('Server does not enforce API auth (no --api-auth); /login accepts anything.')
    client = A1111Webservice(api_url)
    try:
        res = client.login('definitely-not-a-real-user', 'definitely-not-a-real-password')
        # login() only raises on a transport error, not on an HTTP failure status.
        assert res.status_code != 200, 'bad credentials were accepted'
    except RuntimeError as err:
        pytest.skip(f'No WebUI server reachable at {api_url}: {err}')
    finally:
        client.disconnect()


def test_credentials_provider_drives_authenticated_requests(api_url, credentials, auth_enforced):
    """A configured provider should let protected calls succeed and be invoked only on a 401."""
    calls = []

    def provider():
        calls.append(True)
        return credentials

    client = A1111Webservice(api_url, credentials_provider=provider)
    try:
        samplers = client.get_samplers()
        assert isinstance(samplers, list)
    except AuthError as err:
        pytest.fail(f'authenticated request failed with provided credentials: {err}')
    except RuntimeError as err:
        pytest.skip(f'No WebUI server reachable at {api_url}: {err}')
    finally:
        client.disconnect()
    if auth_enforced:
        assert calls, 'provider was never invoked despite the server enforcing auth'
    else:
        assert not calls, 'provider was invoked even though the server does not enforce auth'


def test_missing_provider_raises_autherror_on_protected_server(api_url, auth_enforced):
    """With no provider and auth enabled, a 401 must surface as AuthError (not a hang)."""
    if not auth_enforced:
        pytest.skip('Server does not enforce API auth; there is no 401 to handle.')
    client = A1111Webservice(api_url, credentials_provider=None)
    try:
        with pytest.raises(AuthError):
            client.get_samplers()
    except RuntimeError as err:
        pytest.skip(f'No WebUI server reachable at {api_url}: {err}')
    finally:
        client.disconnect()
