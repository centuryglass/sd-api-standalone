"""Builders for WebUI generation response bodies, shaped like A1111's `/sdapi/v1/txt2img` and `img2img` output."""
import json
from typing import Any


def generation_info(all_seeds: list[int], index_of_first_image: int = 0) -> dict[str, Any]:
    """The decoded `info` of a WebUI generation response for a batch with `all_seeds`.

    Field names and value types follow `Processed.js()` in A1111's `modules/processing.py`.
    """
    count = len(all_seeds)
    return {
        'prompt': 'a red apple', 'all_prompts': ['a red apple'] * count,
        'negative_prompt': '', 'all_negative_prompts': [''] * count,
        'seed': all_seeds[0], 'all_seeds': all_seeds,
        'subseed': 7, 'all_subseeds': [7 + i for i in range(count)], 'subseed_strength': 0.0,
        'width': 8, 'height': 8, 'sampler_name': 'Euler a', 'cfg_scale': 7.0, 'steps': 20, 'batch_size': count,
        'restore_faces': False, 'face_restoration_model': None,
        'sd_model_name': 'model', 'sd_model_hash': '6ce0161689', 'sd_vae_name': None, 'sd_vae_hash': None,
        'seed_resize_from_w': -1, 'seed_resize_from_h': -1, 'denoising_strength': None,
        'extra_generation_params': {'Schedule type': 'Automatic'},
        'index_of_first_image': index_of_first_image, 'infotexts': ['a red apple\nSteps: 20'] * count,
        'styles': [], 'job_timestamp': '20261008031200', 'clip_skip': 1,
        'is_using_inpainting_conditioning': False, 'version': 'v1.10.1',
    }


def generation_response(images: list[str], info: dict[str, Any]) -> dict[str, Any]:
    """A WebUI generation response body holding base64 `images`, with `info` serialized as the server sends it."""
    return {'images': images, 'parameters': {}, 'info': json.dumps(info)}
