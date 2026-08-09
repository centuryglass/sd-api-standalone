"""ControlNet integration tests.

The ControlNet endpoints come from the ``sd-webui-controlnet`` extension. Forks vary
in *which* routes they expose — e.g. ReForge keeps ``/controlnet/model_list``,
``/module_list`` and ``/control_types`` but drops ``/controlnet/version`` and
``/controlnet/settings`` (both 404). So we:

* gate the whole module on ``model_list`` (the reliable "extension present" signal),
  skipping everything if that 404s, and
* let the individual optional-endpoint tests (version, settings) skip on their own
  404 rather than dragging the rest down with them.
"""
import pytest

pytestmark = pytest.mark.integration

# The `controlnet_available` fixture lives in conftest.py (shared with the generation tests).


def test_controlnet_version(controlnet_available):
    # Optional endpoint: present on upstream sd-webui-controlnet, dropped by some forks.
    try:
        version = controlnet_available.get_controlnet_version()
    except RuntimeError as err:
        pytest.skip(f'/controlnet/version not exposed by this build: {err}')
    assert isinstance(version, int)
    assert version > 0


def test_controlnet_models(controlnet_available):
    models = controlnet_available.get_controlnet_models()
    assert isinstance(models.model_list, list) and len(models.model_list) > 0
    # "None" is always offered as the no-op model choice.
    assert any(name.lower() == 'none' for name in models.model_list)


def test_controlnet_modules(controlnet_available):
    modules = controlnet_available.get_controlnet_modules()
    assert isinstance(modules.module_list, list) and len(modules.module_list) > 0
    # "none" is always an available preprocessor module.
    assert any(name.lower() == 'none' for name in modules.module_list)


def test_controlnet_control_types(controlnet_available):
    control_types = controlnet_available.get_controlnet_control_types()
    assert 'control_types' in control_types
    assert isinstance(control_types['control_types'], dict) and len(control_types['control_types']) > 0


def test_controlnet_settings(controlnet_available):
    # Optional endpoint: dropped by some forks (ReForge returns 404).
    try:
        settings = controlnet_available.get_controlnet_settings()
    except RuntimeError as err:
        pytest.skip(f'/controlnet/settings not exposed by this build: {err}')
    assert isinstance(settings, dict)


def test_controlnet_preprocessors_parse(controlnet_available):
    preprocessors = controlnet_available.get_controlnet_preprocessors()
    assert isinstance(preprocessors, list) and len(preprocessors) > 0
    names = [p.name for p in preprocessors]
    assert any(name.lower() == 'none' for name in names)


def test_controlnet_type_categories(controlnet_available):
    categories = controlnet_available.get_controlnet_type_categories()
    assert isinstance(categories, dict) and len(categories) > 0
