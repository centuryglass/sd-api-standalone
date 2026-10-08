"""Offline tests for `build_webui_result`: sorting a WebUI response's images into generated images, seeds and maps.

Responses go through `A1111Webservice._decode_image_response`, so the `info` parsing is covered too.
"""
# pylint: disable=protected-access
import logging
from typing import Any, Optional

from PIL import Image

from sd_backend_client.api.a1111_webservice import A1111Webservice
from sd_backend_client.api.shared_data.generation_handle import GenerationResult
from sd_backend_client.api.webui.response_formats import GenerationInfoData
from sd_backend_client.api.webui.webui_generation_handle import build_webui_result
from sd_backend_client.util.visual.image_utils import image_to_base64

from .webui_responses import generation_info, generation_response

TASK_ID = 'task(txt2img-ABCDEFG)'


def _image(value: int) -> Image.Image:
    """A small image whose first pixel identifies it."""
    return Image.new('RGBA', (2, 2), (value, value, value, 255))


def _ids(images: list[Image.Image]) -> list[int]:
    return [image.getpixel((0, 0))[0] for image in images]


def _result(image_ids: list[int], info: Optional[dict[str, Any]]) -> GenerationResult:
    """Decode a response holding images `image_ids` and `info`, then build its result."""
    encoded = [image_to_base64(_image(value)) for value in image_ids]
    body = generation_response(encoded, info) if info is not None else {'images': encoded}
    response = A1111Webservice._decode_image_response(body)
    return build_webui_result(response['images'], response['info'], TASK_ID)


def test_single_image_reports_its_seed():
    """One generated image comes back with its seed and the parsed info as raw_info."""
    result = _result([1], generation_info([1234]))
    assert _ids(result.images) == [1]
    assert result.seeds == [1234] and result.seed == 1234
    assert not result.control_maps and not result.extra
    assert isinstance(result.raw_info, GenerationInfoData)
    assert result.task_id == TASK_ID


def test_controlnet_detect_maps_are_split_from_the_generated_images():
    """Images after the last seeded one are detect maps, so `images` holds only the generated ones."""
    result = _result([1, 2, 50, 51], generation_info([10, 11]))
    assert _ids(result.images) == [1, 2]
    assert result.seeds == [10, 11]
    assert _ids(result.control_maps) == [50, 51]


def test_batch_grid_moves_to_extra():
    """A leading batch grid, marked by index_of_first_image, goes to extra['grid'] rather than `images`."""
    result = _result([99, 1, 2, 50], generation_info([10, 11], index_of_first_image=1))
    assert _ids(result.images) == [1, 2]
    assert result.seeds == [10, 11]
    assert _ids(result.control_maps) == [50]
    assert _ids([result.extra['grid']]) == [99]


def test_response_without_info_keeps_every_image_and_no_seeds():
    """Basic upscaling and preprocessor previews carry no info, so every image counts and no seed is reported."""
    result = _result([1], None)
    assert _ids(result.images) == [1]
    assert not result.seeds and result.seed is None and result.raw_info is None


def test_counts_that_do_not_fit_the_layout_keep_every_image(caplog):
    """More seeds than images can't be aligned, so every image counts as generated and a warning is logged."""
    with caplog.at_level(logging.WARNING):
        result = _result([1], generation_info([10, 11]))
    assert _ids(result.images) == [1]
    assert not result.seeds and result.seed == 10
    assert 'treating every image as generated' in caplog.text
