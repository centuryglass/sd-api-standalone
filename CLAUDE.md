# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`intrapaint_api` is a standalone, headless Stable Diffusion backend client extracted from
[IntraPaint](https://github.com/centuryglass/IntraPaint)'s `src/api`. It talks to **ComfyUI**,
**Forge/Automatic1111 WebUI**, and their **ControlNet** extensions, building the request bodies /
node graphs each backend expects. Everything lives under the `intrapaint_api` package and imports
only from within it plus third-party libs.

Two hard constraints define this codebase (see README "Design notes"):
- **No Qt / PySide6, no cv2 / numpy.** Images are plain `PIL.Image` (normalized to RGBA); sizes use
  the minimal `Size` class in `util/geometry.py`. The `_tr()` translation helpers are no-op shims.
- **No config layer.** Generation parameters are passed explicitly as pydantic models, never read
  from a `Cache`/`AppConfig` singleton. When adding a parameter, thread it through the pydantic model
  — do not reintroduce hidden global config.

Requires **Python 3.11+** (uses `match` and `X | Y` type aliases).

## Commands

```bash
pip install -r requirements.txt   # pillow, requests, platformdirs, websocket-client, pydantic
pip install pytest                # test-only, not in requirements.txt

# Offline pure-logic unit tests — no server, no GPU, deterministic (use these by default):
pytest tests/unit/
pytest tests/unit/test_comfy_workflow_builder.py::TEST_NAME   # single test

# Live integration tests against whatever backend(s) are up; the rest skip themselves:
pytest tests/                     # fast read-only metadata/auth tests + unit tests
pytest tests/ --run-generation    # add slow real-diffusion round-trips (needs a loaded checkpoint)
RUN_SD_GENERATION=1 pytest tests/ # equivalent env-var form
```

Point the suite at servers via env vars (defaults in parentheses): `SD_API_URL`
(`http://127.0.0.1:7860`), `COMFYUI_API_URL` (`http://127.0.0.1:8188`), `SD_UNAME`/`SD_PASS` for
A1111 `--api-auth`. `tests/local.env` (gitignored) is a convenience `source`-able file for
credentials. See `tests/README.md` for the full matrix. There is no lint or build step configured.

## Architecture

The package is a **functional core / imperative shell**: pure builders turn pydantic params into
wire payloads, and thin HTTP clients send them. The two backends' APIs are shaped very differently,
which is the main thing to understand.

### Shared, backend-agnostic layer — `api/shared_data/`
- `diffusion_params.py` — `DiffusionParams`, the pydantic base holding parameters common to both
  backends. Each backend **subclasses** it: `DiffusionRequestBody` (WebUI) and
  `ComfyUIDiffusionParams` add backend-specific fields.
- `api_datatypes.py` — `DiffusionUpscalingParams` and related shared models.
- `controlnet/` — `ControlNetUnit` + `ControlNetModel` + `ControlNetPreprocessor`, all
  backend-agnostic. You attach units to `diffusion_params.controlnet_units`; each backend serializes
  them into its own form. Serialization round-trips are pinned by `tests/unit/test_controlnet_serialization.py`.

### `api/webservice.py` — `WebService` base
Session/auth/GET/POST helper both clients extend. On a 401 it calls `_handle_auth_error()`
(subclass-overridden) then retries the request. `A1111Webservice` implements it via a
`credentials_provider` callback (replacing IntraPaint's Qt login dialog); ComfyUI has no auth.

### WebUI client — `api/a1111_webservice.py` + `api/webui/` (SYNCHRONOUS)
`A1111Webservice.txt2img(body)` / `img2img(image, mask, body)` block until the image is ready and
return `{'images': list[PIL.Image], 'info': ...}`. `api/webui/` holds the request/response pydantic
formats; `diffusion_request_body.py`'s `DiffusionRequestBody.to_dict()` strips `None`s and is where
ControlNet units get injected into `alwayson_scripts`.

### ComfyUI client — `api/comfyui_webservice.py` + `api/comfyui/` (ASYNCHRONOUS + node graphs)
ComfyUI is a queue: `ComfyUiWebservice.txt2img(params)` returns immediately with a `prompt_id`; you
poll `check_queue_entry(prompt_id, number)` until FINISHED, then `download_images(...)`. Progress is
also streamed over a websocket (`websocket-client`).

The request body is a **node graph**, not JSON fields. This is the most involved part of the repo:
- `comfyui/diffusion_workflow_builder.py` — `DiffusionWorkflowBuilder`. Configure it by setting
  attributes, then `build_workflow()` returns a `ComfyNodeGraph`. It decides checkpoint/KSampler
  wiring, prompt encoding, txt2img-vs-img2img latent source, CLIP-skip insertion, batching, etc.
  Its graph structure is pinned by `tests/unit/test_comfy_workflow_builder.py`.
- `comfyui/nodes/` — one class per ComfyUI node type (`ComfyNode` subclasses), grouped into
  `input/`, `vae/`, `model_extensions/`, `controlnet/`. Each node declares its valid input keys and
  output count. `nodes/comfy_node_graph.py` (`ComfyNodeGraph`) assigns integer string keys, wires
  connections as `(node_key, output_slot)` tuples, and emits the final `{key: {class_type, inputs}}`
  workflow dict. Node keys start at 3 and both `ComfyNode` and `ComfyNodeGraph` implement
  `__deepcopy__` because builders clone partial graphs.
- Separate builders exist for upscaling / preprocessor-preview workflows
  (`basic_upscale_workflow_builder.py`, `latent_upscale_workflow_builder.py`,
  `preprocessor_preview_workflow_builder.py`).

### `util/`
`geometry.py` (`Size`), `shared_constants.py`, `api_defaults.py`, `singleton.py` (a `Singleton`
metaclass), and `visual/image_utils.py` (base64/PNG ↔ `PIL.Image`, RGBA normalization — the seam
where Qt `QImage` used to be).

## Working in this repo

- Prefer `tests/unit/` for verifying builder/serialization changes — they're offline, deterministic,
  and cover exactly the request-shaping logic. A change that silently alters an emitted request
  should fail there, not only in a live generation.
- The ControlNet generation integration tests contain a **silent-failure detector**: they generate
  the same seed+prompt with and without a control unit and fail unless the outputs differ. Keep that
  invariant in mind when touching ControlNet serialization.
- Status: under active refactoring. Core txt2img / img2img / inpainting / ControlNet paths work
  end-to-end on both backends; the ComfyUI node classes' pydantic migration and tiled upscaling are
  still in progress.
