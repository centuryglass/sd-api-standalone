"""Read-only metadata integration tests against a live A1111 / Forge / reForge / Forge Neo WebUI.

These are fast and non-destructive: they only query the server for its capabilities,
models, and current state. They exercise the ``get_*`` accessors on
:class:`A1111Webservice` and validate the shapes documented in
``sd_backend_client/api/webui/response_formats.py``.
"""
import pytest

from sd_backend_client.errors import ServerError

pytestmark = pytest.mark.integration


def test_get_config_returns_options_dict(service):
    config = service.get_config()
    assert isinstance(config, dict)
    # Every A1111/Forge build exposes the active checkpoint under this key.
    assert 'sd_model_checkpoint' in config


def test_get_samplers(service):
    samplers = service.get_samplers()
    assert isinstance(samplers, list) and len(samplers) > 0
    for sampler in samplers:
        assert isinstance(sampler.name, str) and sampler.name
        assert isinstance(sampler.aliases, list)


def test_get_upscalers(service):
    upscalers = service.get_upscalers()
    assert isinstance(upscalers, list) and len(upscalers) > 0
    names = [u.name for u in upscalers]
    # "None" and "Lanczos" are built in and always present.
    assert 'Lanczos' in names
    for upscaler in upscalers:
        assert upscaler.name
        assert upscaler.scale is not None


def test_get_models(service):
    models = service.get_models()
    assert isinstance(models, list) and len(models) > 0, 'no checkpoints installed on the server'
    for model in models:
        assert model.title
        assert model.model_name


def test_get_vae_with_forge_fallback(service):
    # get_vae() falls back to the Forge /sdapi/v1/sd-modules endpoint when
    # /sdapi/v1/sd-vae is unavailable; either way we should get a list back.
    vae_models = service.get_vae()
    assert isinstance(vae_models, list)


def test_get_loras(service):
    loras = service.get_loras()
    assert isinstance(loras, list)
    for lora in loras:
        assert lora.name


def test_get_hypernetworks(service):
    # Optional endpoint: Forge Neo drops hypernetwork support.
    try:
        hypernetworks = service.get_hypernetworks()
    except ServerError as err:
        if err.status_code != 404:
            raise
        pytest.skip(f'/sdapi/v1/hypernetworks not exposed by this build: {err}')
    assert isinstance(hypernetworks, list)
    assert all(isinstance(name, str) for name in hypernetworks)


def test_get_latent_upscale_modes(service):
    modes = service.get_latent_upscale_modes()
    assert isinstance(modes, list) and len(modes) > 0
    assert all(isinstance(name, str) for name in modes)


def test_get_styles(service):
    # Returned as a list of JSON strings; may be empty if the user saved none.
    styles = service.get_styles()
    assert isinstance(styles, list)


def test_get_scripts(service):
    scripts = service.get_scripts()
    assert isinstance(scripts.txt2img, list)
    assert isinstance(scripts.img2img, list)


def test_get_script_info(service):
    info = service.get_script_info()
    assert isinstance(info, list)
    if info:
        assert info[0].name


def test_progress_check_shape(service):
    progress = service.progress_check()
    assert isinstance(progress.progress, (int, float))
    assert progress.state is not None
    assert isinstance(progress.state.sampling_steps, int)


def test_refresh_checkpoints(service):
    res = service.refresh_checkpoints()
    assert res.status_code == 200
