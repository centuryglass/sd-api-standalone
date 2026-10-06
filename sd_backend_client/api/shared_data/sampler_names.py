"""Shared sampler and scheduler names, and their translation to each backend's own names.

`DiffusionParams.sampler` and `DiffusionParams.scheduler` use ComfyUI's KSampler identifiers as the shared vocabulary.
ComfyUI takes them unchanged, and WebUI requests translate them through `SAMPLER_WEBUI_NAMES` and
`SCHEDULER_WEBUI_NAMES`. WebUI names are also accepted on ComfyUI and translated back. A name found in neither table
is sent unchanged, so it works only on a backend that defines it.
"""

SAMPLER_WEBUI_NAMES: dict[str, str] = {
    'euler': 'Euler',
    'euler_ancestral': 'Euler a',
    'heun': 'Heun',
    'dpm_2': 'DPM2',
    'dpm_2_ancestral': 'DPM2 a',
    'lms': 'LMS',
    'dpm_fast': 'DPM fast',
    'dpm_adaptive': 'DPM adaptive',
    'dpmpp_2s_ancestral': 'DPM++ 2S a',
    'dpmpp_sde': 'DPM++ SDE',
    'dpmpp_2m': 'DPM++ 2M',
    'dpmpp_2m_sde': 'DPM++ 2M SDE',
    'dpmpp_3m_sde': 'DPM++ 3M SDE',
    'ddpm': 'DDPM',
    'lcm': 'LCM',
    'ddim': 'DDIM',
    'uni_pc': 'UniPC',
}
"""Shared (ComfyUI) sampler name -> WebUI sampler name, as WebUI's `/sdapi/v1/samplers` lists it."""

SCHEDULER_WEBUI_NAMES: dict[str, str] = {
    'normal': 'normal',
    'karras': 'karras',
    'exponential': 'exponential',
    'sgm_uniform': 'sgm_uniform',
    'simple': 'simple',
    'ddim_uniform': 'ddim',
    'beta': 'beta',
    'kl_optimal': 'kl_optimal',
}
"""Shared (ComfyUI) scheduler name -> WebUI scheduler name.

WebUI accepts a separate scheduler from A1111 1.9 on, and the `normal`, `simple`, `ddim` and `beta` schedulers from
A1111 1.10 on. Older servers ignore the field.
"""

_SAMPLER_SHARED_NAMES = {webui_name: shared_name for shared_name, webui_name in SAMPLER_WEBUI_NAMES.items()}
_SCHEDULER_SHARED_NAMES = {webui_name: shared_name for shared_name, webui_name in SCHEDULER_WEBUI_NAMES.items()}

# WebUI's scheduler display labels, accepted on ComfyUI as well as the names above.
_SCHEDULER_SHARED_NAMES.update({
    'Normal': 'normal',
    'Karras': 'karras',
    'Exponential': 'exponential',
    'SGM Uniform': 'sgm_uniform',
    'Simple': 'simple',
    'DDIM': 'ddim_uniform',
    'Beta': 'beta',
    'KL Optimal': 'kl_optimal',
})


def webui_sampler_name(sampler: str) -> str:
    """Returns the WebUI name for a shared or WebUI sampler name, or `sampler` unchanged if it has none."""
    return SAMPLER_WEBUI_NAMES.get(comfyui_sampler_name(sampler), sampler)


def webui_scheduler_name(scheduler: str) -> str:
    """Returns the WebUI name for a shared or WebUI scheduler name, or `scheduler` unchanged if it has none."""
    return SCHEDULER_WEBUI_NAMES.get(comfyui_scheduler_name(scheduler), scheduler)


def comfyui_sampler_name(sampler: str) -> str:
    """Returns the ComfyUI name for a shared or WebUI sampler name, or `sampler` unchanged if it has none."""
    return _SAMPLER_SHARED_NAMES.get(sampler, sampler)


def comfyui_scheduler_name(scheduler: str) -> str:
    """Returns the ComfyUI name for a shared or WebUI scheduler name, or `scheduler` unchanged if it has none."""
    return _SCHEDULER_SHARED_NAMES.get(scheduler, scheduler)
