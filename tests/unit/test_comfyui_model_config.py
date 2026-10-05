"""Unit tests for ComfyUI model-config auto-detection in ``ComfyUiWebservice._build_diffusion_body``."""
import logging
from collections.abc import Sequence
from unittest.mock import patch

import pytest

from sd_backend_client.api.comfyui.comfyui_diffusion_params import ComfyUIDiffusionParams
from sd_backend_client.api.comfyui_webservice import ComfyUiWebservice

CONFIGS = ('v1-inference.yaml', 'README', 'sd_xl.v1.yaml')


def _config_for(params: ComfyUIDiffusionParams, configs: Sequence[str] = CONFIGS) -> str | None:
    service = ComfyUiWebservice('http://localhost:8188')
    with patch.object(ComfyUiWebservice, 'get_models', return_value=list(configs)):
        return service._build_diffusion_body(params).model_config_path  # pylint: disable=protected-access


def test_dotless_config_entry_does_not_raise() -> None:
    """A config entry with no extension is skipped instead of raising ValueError."""
    assert _config_for(ComfyUIDiffusionParams(sd_model_name='m.safetensors')) is None


def test_config_matches_model_name_without_extension() -> None:
    """The config is matched on the name with only its final extension removed."""
    assert _config_for(ComfyUIDiffusionParams(sd_model_name='v1-inference.ckpt')) == 'v1-inference.yaml'


def test_dotted_model_name_matches_dotted_config() -> None:
    """A model name containing dots matches the config sharing that full stem."""
    assert _config_for(ComfyUIDiffusionParams(sd_model_name='sd_xl.v1.safetensors')) == 'sd_xl.v1.yaml'


def test_explicit_config_in_list_is_kept() -> None:
    """An explicit config the server lists is used as given."""
    params = ComfyUIDiffusionParams(sd_model_name='sd_xl.v1.safetensors', sd_model_config='v1-inference.yaml')
    assert _config_for(params) == 'v1-inference.yaml'


def test_missing_explicit_config_warns(caplog: pytest.LogCaptureFixture) -> None:
    """An explicit config absent from the server list logs a warning and falls back to name matching."""
    params = ComfyUIDiffusionParams(sd_model_name='sd_xl.v1.safetensors', sd_model_config='missing.yaml')
    with caplog.at_level(logging.WARNING):
        assert _config_for(params) == 'sd_xl.v1.yaml'
    assert 'missing.yaml' in caplog.text
