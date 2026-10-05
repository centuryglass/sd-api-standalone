# sd-backend-client

A standalone, self-contained copy of IntraPaint's Stable Diffusion backend client
(`src/api`). It talks to **ComfyUI**, **Forge/Automatic1111 WebUI**, and their
**ControlNet** extensions, and builds the request bodies / node graphs those backends expect.

This package was extracted from [IntraPaint](https://github.com/centuryglass/IntraPaint) so it can
be dropped into other projects without pulling in the rest of the editor. Everything lives under the
`sd_backend_client` package and imports only from within it (plus third-party libraries).

**No Qt / PySide6 dependency.** Images are plain [Pillow](https://python-pillow.org/) `PIL.Image`
objects and the package has no GUI requirements — it runs headless.

## Status

This package is under active refactoring. The core txt2img / img2img / inpainting / ControlNet paths
work end-to-end against both backends (covered by the integration tests), but some areas — notably the
ComfyUI node classes' migration to pydantic and the tiled-upscaling workflow — are still in progress.

My eventual goal is to provide an API-agnostic interface, allowing complex image generation requests
to pass to either ComfyUI or Stable-Diffusion-WebUI without needing to care which interface is
actually available. To accomplish this, response data formats still need to be unified, and work
towards queued vs. blocking request handling still needs to be fully completed and tested.

## Installation

```
pip install -e .
```

This installs the package (and its dependencies) in editable mode, so `sd_backend_client` is importable
from anywhere without manually managing `sys.path`.

Requires **Python 3.11+** (uses `match` statements and PEP 604 / `X | Y` type aliases).
Runtime dependencies: `pillow`, `requests`, `websocket-client`, `pydantic`, `typing-extensions`.

## Layout

```
sd_backend_client/
  api/
    webservice.py            # base HTTP/session helper
    a1111_webservice.py      # Forge / A1111 WebUI client (synchronous)
    comfyui_webservice.py    # ComfyUI client (async queue + polling)
    shared_data/             # backend-agnostic pydantic models
      diffusion_params.py        # DiffusionParams: shared generation parameters
      api_datatypes.py           # DiffusionUpscalingParams and friends
      controlnet/                # ControlNetUnit / PreprocessorParams / ControlNet model
    comfyui/                 # ComfyUI node-graph + workflow builders (+ ComfyUIDiffusionParams)
    webui/                   # WebUI request/response formats (+ DiffusionRequestBody)
  util/                    # shared constants, Size, PIL image helpers
```

## Usage

Generation parameters are passed explicitly as [pydantic](https://docs.pydantic.dev/) models — there
is no hidden config/cache layer. Images passed in and returned are `PIL.Image` objects; no
`QApplication` or event loop is needed.

`DiffusionParams` (in `shared_data`) holds the parameters common to both backends. Each backend
extends it: `DiffusionRequestBody` (WebUI) and `ComfyUIDiffusionParams` (ComfyUI) add
backend-specific fields.

### Forge / A1111 WebUI (synchronous)

```python
from sd_backend_client.api.a1111_webservice import A1111Webservice
from sd_backend_client.api.webui.diffusion_request_body import DiffusionRequestBody

service = A1111Webservice('http://localhost:7860')
body = DiffusionRequestBody(prompt='a corgi astronaut, detailed',
                            steps=30, cfg_scale=7.0, width=512, height=512)
result = service.txt2img(body)     # blocks until the image is generated
images = result['images']          # list[PIL.Image]
```

For img2img / inpainting, pass the source image (and, for inpainting, a mask) to
`service.img2img(image, mask, body)`.

### ComfyUI (asynchronous)

ComfyUI generation is queued: the call returns immediately with a prompt id, and you poll for
completion, then download the results.

```python
from sd_backend_client.api.comfyui_webservice import ComfyUiWebservice
from sd_backend_client.api.comfyui.comfyui_diffusion_params import ComfyUIDiffusionParams

service = ComfyUiWebservice('http://localhost:8188')
params = ComfyUIDiffusionParams(sd_model_name='deliberate_v3.safetensors',
                                prompt='a corgi astronaut, detailed',
                                steps=20, sampler='euler', scheduler='karras')
response = service.txt2img(params)                 # queues a job
# poll service.check_queue_entry(response.prompt_id, response.number) until it reports FINISHED,
# then service.download_images(progress.outputs.images) to get the PIL images.
```

Set `init_images` (and `mask`) on the params for img2img / inpainting.

### ControlNet

ControlNet units are backend-agnostic: build a `ControlNetUnit` (with an optional `ControlNetModel`
and a `PreprocessorParams`) and add it to `diffusion_params.controlnet_units`. Each backend serializes
them into the form it expects — WebUI's `alwayson_scripts` ControlNet args, or ComfyUI ControlNet
nodes.

### Authentication (A1111 / Forge)

IntraPaint prompted for credentials with a Qt login dialog. The standalone client instead accepts a
`credentials_provider` callback, invoked when the server requires auth; it returns a
`(username, password)` pair to try, or `None` to abort:

```python
from sd_backend_client.api.a1111_webservice import A1111Webservice

service = A1111Webservice('http://localhost:7860',
                          credentials_provider=lambda: ('user', 'password'))
```

## Design notes

Extracted from IntraPaint's `src/api` and then decoupled in two ways:

- **No Qt.** `QImage` → `PIL.Image` (via the small helpers in `util/visual/image_utils.py`; decoded
  images are normalized to RGBA), `QSize` → the minimal `Size` value class in `util/geometry.py`, and
  the UI strings and `_tr()` translation helpers are removed.
  Dropping Qt also dropped `cv2` / `numpy`, which were only used by the old image helpers.
- **No config layer.** The original client read generation parameters out of JSON-backed `Cache` /
  `AppConfig` singletons. That whole system (and the `resources/config/*.json` definitions behind it)
  has been removed; parameters are now passed explicitly as the pydantic models described above.
  Moving to pydantic also replaced the hand-rolled request/response typing with validated models.

## Verifying against a live backend

The client's wire behavior is exercised by an integration test suite under `tests/` that runs against
real ComfyUI and A1111/Forge servers (see `tests/README.md`). If you rely on img2img / inpainting /
ControlNet, a real round-trip is worth more than trusting imports alone.

## Development

`pip install -r requirements-dev.txt` adds the test, lint and type-check tools. CI (`.github/workflows/ci.yml`) runs
the unit tests on Python 3.11-3.14, a pylint check and a mypy check on every pull request into `main`. Releases are
cut by [release-please](https://github.com/googleapis/release-please) from Conventional Commits pull request titles;
see `CHANGELOG.md` once the first release lands. Rules for coding agents are in `AGENTS.md`.

## License

Public domain, dedicated under [The Unlicense](https://unlicense.org) — see `LICENSE`. Use it for
anything, with no attribution or other obligations.

