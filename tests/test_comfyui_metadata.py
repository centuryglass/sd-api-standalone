"""Read-only metadata integration tests against a live ComfyUI backend.

Fast and non-destructive: they query the server for its capabilities, models, and node
graph, exercising the ``get_*`` accessors on :class:`ComfyUiWebservice`. ComfyUI has no
authentication, so there is no login/credentials story to test here.
"""
import pytest

from intrapaint_api.api.comfyui_webservice import ComfyModelType

pytestmark = pytest.mark.integration


def test_system_stats(comfy_service):
    stats = comfy_service.get_system_stats()
    assert stats.system is not None
    assert stats.devices is not None
    assert isinstance(stats.devices, list) and len(stats.devices) > 0


def test_get_model_types(comfy_service):
    model_types = comfy_service.get_model_types()
    assert isinstance(model_types, list) and len(model_types) > 0
    # The checkpoints folder is always part of a working install.
    assert ComfyModelType.CHECKPOINT.value in model_types


def test_get_sd_checkpoints(comfy_service):
    checkpoints = comfy_service.get_sd_checkpoints()
    assert isinstance(checkpoints, list) and len(checkpoints) > 0, 'no checkpoints installed'
    assert all(isinstance(name, str) for name in checkpoints)


def test_get_vae_models(comfy_service):
    assert isinstance(comfy_service.get_vae_models(), list)


def test_get_lora_models(comfy_service):
    assert isinstance(comfy_service.get_lora_models(), list)


def test_get_controlnet_models(comfy_service):
    assert isinstance(comfy_service.get_controlnet_models(), list)


def test_get_hypernetwork_models(comfy_service):
    assert isinstance(comfy_service.get_hypernetwork_models(), list)


def test_get_upscale_models(comfy_service):
    assert isinstance(comfy_service.get_models(ComfyModelType.UPSCALING), list)


def test_get_embeddings(comfy_service):
    assert isinstance(comfy_service.get_embeddings(), list)


def test_get_extensions(comfy_service):
    extensions = comfy_service.get_extensions()
    assert isinstance(extensions, list)


def test_get_sampler_names(comfy_service):
    samplers = comfy_service.get_sampler_names()
    assert isinstance(samplers, list) and len(samplers) > 0
    # "euler" is a core ComfyUI sampler present on every build.
    assert 'euler' in samplers


def test_get_scheduler_names(comfy_service):
    schedulers = comfy_service.get_scheduler_names()
    assert isinstance(schedulers, list) and len(schedulers) > 0
    assert 'normal' in schedulers or 'karras' in schedulers


def test_is_node_available(comfy_service):
    # KSampler is a core node; a made-up name should not be reported as available.
    assert comfy_service.is_node_available('KSampler') is True
    assert comfy_service.is_node_available('NotARealComfyNode_zzz') is False


def test_get_queue_info(comfy_service):
    queue = comfy_service.get_queue_info()
    assert queue.queue_running is not None
    assert queue.queue_pending is not None
    assert isinstance(queue.queue_running, list)
    assert isinstance(queue.queue_pending, list)
