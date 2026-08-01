"""Read-only metadata integration tests against a live A1111 / Forge / ReForge WebUI.

These are fast and non-destructive: they only query the server for its capabilities,
models, and current state. They exercise the ``get_*`` accessors on
:class:`A1111Webservice` and validate the shapes documented in
``intrapaint_api/api/webui/response_formats.py``.
"""
import pytest

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
        assert 'name' in sampler and isinstance(sampler['name'], str)
        assert 'aliases' in sampler


def test_get_upscalers(service):
    upscalers = service.get_upscalers()
    assert isinstance(upscalers, list) and len(upscalers) > 0
    names = [u['name'] for u in upscalers]
    # "None" and "Lanczos" are built in and always present.
    assert 'Lanczos' in names
    for upscaler in upscalers:
        assert 'name' in upscaler
        assert 'scale' in upscaler


def test_get_models(service):
    models = service.get_models()
    assert isinstance(models, list) and len(models) > 0, 'no checkpoints installed on the server'
    for model in models:
        assert 'title' in model
        assert 'model_name' in model


def test_get_vae_with_forge_fallback(service):
    # get_vae() falls back to the Forge /sdapi/v1/sd-modules endpoint when
    # /sdapi/v1/sd-vae is unavailable; either way we should get a list back.
    vae_models = service.get_vae()
    assert isinstance(vae_models, list)


def test_get_loras(service):
    loras = service.get_loras()
    assert isinstance(loras, list)
    for lora in loras:
        assert 'name' in lora


def test_get_hypernetworks(service):
    hypernetworks = service.get_hypernetworks()
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
    assert 'txt2img' in scripts
    assert 'img2img' in scripts
    assert isinstance(scripts['txt2img'], list)
    assert isinstance(scripts['img2img'], list)


def test_get_script_info(service):
    info = service.get_script_info()
    assert isinstance(info, list)
    if info:
        assert 'name' in info[0]


def test_progress_check_shape(service):
    progress = service.progress_check()
    assert 'progress' in progress
    assert 'state' in progress
    assert isinstance(progress['progress'], (int, float))
    assert 'sampling_steps' in progress['state']


def test_refresh_checkpoints(service):
    res = service.refresh_checkpoints()
    assert res.status_code == 200
