# Backend integration tests

Live-API integration tests for the two backend clients:

- `intrapaint_api.api.a1111_webservice.A1111Webservice` (A1111 / Forge / ReForge)
- `intrapaint_api.api.comfyui_webservice.ComfyUiWebservice` (ComfyUI)

They talk to a **real, running** server rather than mocking HTTP. Point the suite at
whichever backend is up — the other backend's tests skip themselves automatically, so
`pytest tests/` is safe with only one (or neither) running.

## Requirements

- A running A1111/Forge/ReForge WebUI (`--api` enabled) **and/or** a running ComfyUI.
- `pip install -r requirements.txt` plus `pytest`.

## Environment

| Variable         | Default                  | Purpose                                        |
|------------------|--------------------------|------------------------------------------------|
| `SD_API_URL`     | `http://127.0.0.1:7860`  | A1111 base URL (overrides host+port).          |
| `SD_API_PORT`    | `7860`                   | A1111 port only, when `SD_API_URL` is unset.   |
| `SD_UNAME`       | —                        | A1111 username, if the server uses `--api-auth`.|
| `SD_PASS`        | —                        | A1111 password, if the server uses `--api-auth`.|
| `COMFYUI_API_URL`| `http://127.0.0.1:8188`  | ComfyUI base URL (overrides host+port).        |
| `COMFYUI_API_PORT`| `8188`                  | ComfyUI port only, when `COMFYUI_API_URL` unset.|
| `SD_TEST_OUTPUT_DIR` | `tests/output/`      | Where generation tests save PNGs for inspection.|

(ComfyUI has no authentication, so there is no ComfyUI equivalent of `SD_UNAME`/`SD_PASS`.)

Config isolation: `conftest.py` points `Cache` / `AppConfig` at a throwaway temp dir
before anything else instantiates them, so tests never read or clobber the shared
IntraPaint per-user config.

## Running

```bash
# Fast, read-only metadata + auth tests for whichever backend(s) are up:
pytest tests/

# Include the slow generation tests (real diffusion; needs a checkpoint loaded):
pytest tests/ --run-generation
# or
RUN_SD_GENERATION=1 pytest tests/
```

## What's covered

### A1111 / Forge / ReForge

- **`test_a1111_metadata.py`** — read-only `get_*` accessors (config, samplers,
  upscalers, models, VAE with Forge fallback, LoRAs, hypernetworks, styles, scripts,
  progress, checkpoint refresh).
- **`test_a1111_auth.py`** — `/login` flow and the `credentials_provider` callback.
  Auto-detects whether the server enforces `--api-auth` (`auth_enforced` fixture) and
  asserts the right outcome either way.
- **`test_a1111_controlnet.py`** — ControlNet extension endpoints. Gated on
  `/controlnet/model_list` (the reliable "installed" signal); the fork-specific optional
  routes `/controlnet/version` and `/controlnet/settings` skip individually if a build
  (e.g. ReForge) drops them.
- **`test_a1111_generation.py`** *(opt-in)* — real round-trips: txt2img (single/batch,
  seed echo), img2img, inpaint-with-mask, basic upscale, interrogate, interrupt.
- **`test_a1111_controlnet_generation.py`** *(opt-in)* — ControlNet diffusion. Includes a
  **silent-failure detector**: the same seed + prompt is generated with and without a
  control unit, and the test fails unless the two images actually differ.

### ComfyUI

ComfyUI's API is asynchronous and workflow-based (jobs are queued by `prompt_id` and
polled to completion); `comfy_helpers.py` wraps that lifecycle and the cache setup the
workflow builders read from.

- **`test_comfyui_metadata.py`** — read-only accessors (system stats, model types,
  checkpoints, VAE/LoRA/ControlNet/hypernetwork/upscale models, embeddings, extensions,
  sampler & scheduler names, `is_node_available`, queue info).
- **`test_comfyui_generation.py`** *(opt-in)* — real round-trips: txt2img (seed echo),
  img2img, inpaint-with-mask, upscale, interrupt.
- **`test_comfyui_controlnet_generation.py`** *(opt-in)* — Canny preprocessor preview
  plus the same **silent-failure detector** A/B (with vs. without a control unit).

### Saved outputs

The generation tests write their results to the output dir (default `tests/output/`,
git-ignored) so you can eyeball fidelity — including the ControlNet input, the
`/controlnet/detect` preview, and the with/without-ControlNet baseline pair.

## Graceful degradation

The suite is designed to stay green across environments: it **skips** (never errors)
when the server is unreachable, when auth is disabled, when ControlNet is absent, or
when the generation flag is off.
