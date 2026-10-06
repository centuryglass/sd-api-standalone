# Backend integration tests

Live-API integration tests for the two backend clients:

- `sd_backend_client.api.a1111_webservice.A1111Webservice` (A1111 / Forge / ReForge)
- `sd_backend_client.api.comfyui_webservice.ComfyUiWebservice` (ComfyUI)

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

## Running

```bash
# Pure-logic unit tests only (no server, no GPU — always runnable, e.g. in CI):
pytest tests/unit/

# Fast, read-only metadata + auth tests for whichever backend(s) are up (+ unit tests):
pytest tests/

# Include the slow generation tests (real diffusion; needs a checkpoint loaded):
pytest tests/ --run-generation
# or
RUN_SD_GENERATION=1 pytest tests/
```

## Unit tests (`tests/unit/`)

Backend-independent tests of the pure "functional core": the code that turns parameters
into request bodies / node graphs, plus serialization and the image/geometry helpers. They
need **no running server** and no GPU, are deterministic, and drive the builders by setting
attributes directly.

- **`test_comfy_workflow_builder.py`** — `DiffusionWorkflowBuilder.build_workflow()` graph
  structure (checkpoint/KSampler params, prompt encoding, txt2img vs img2img latent source,
  CLIP-skip insertion, batch validation).
- **`test_diffusion_request_body.py`** — `DiffusionRequestBody.to_dict()` None-stripping and
  field passthrough.
- **`test_controlnet_serialization.py`** — `ControlNetUnit` / preprocessor serialize↔deserialize
  round-trips (both WebUI and ComfyUI key formats).
- **`test_image_and_geometry.py`** — base64/PNG round-trips, RGBA normalization, `Size`.
- **`test_param_parity.py`** - the same `DiffusionParams` serialized for both backends asks for the same
  generation: sampler and scheduler names, size, batch size, denoising strength and checkpoint.

These pin behavior ahead of the planned config-decoupling refactor: they cover exactly the
logic that survives it, so a regression that silently changes an emitted request will fail
here (fast, offline) rather than only showing up in a live generation.

## What's covered (integration)

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

### Both backends

- **`test_shared_params.py`** - each server lists the names in `sampler_names`' tables (unlisted ones are
  warnings); *(opt-in)* img2img resizes a source to `width` x `height` on both backends, and WebUI applies the
  requested scheduler.

### Saved outputs

The generation tests write their results to the output dir (default `tests/output/`,
git-ignored) so you can eyeball fidelity — including the ControlNet input, the
`/controlnet/detect` preview, and the with/without-ControlNet baseline pair.

## Graceful degradation

The suite is designed to stay green across environments: it **skips** (never errors)
when the server is unreachable, when auth is disabled, when ControlNet is absent, or
when the generation flag is off.
