# intrapaint_api

A standalone, self-contained copy of IntraPaint's Stable Diffusion backend client
(`src/api`). It talks to **ComfyUI**, **Forge/Automatic1111 WebUI**, and their
**ControlNet** extensions, and builds the request bodies / node graphs those backends expect.

This package was extracted from [IntraPaint](https://github.com/centuryglass/IntraPaint) so it can
be dropped into other projects without pulling in the rest of the editor. Everything lives under the
`intrapaint_api` package and imports only from within it (plus third-party libraries).

**No Qt / PySide6 dependency.** Images are plain [Pillow](https://python-pillow.org/) `PIL.Image`
objects and the package has no GUI requirements — it runs headless.

## Installation

```
pip install -r requirements.txt
```

Then put the `intrapaint_api/` directory on your import path (or `pip install` it once a
`pyproject.toml`/`setup.py` is added).

Requires **Python 3.11+** (uses `match` statements and PEP 604 / `X | Y` type aliases).
Runtime dependencies: `pillow`, `requests`, `platformdirs`, `websocket-client`.

## Layout

```
intrapaint_api/
  api/                 # the client code (was src/api)
    webservice.py            # base HTTP/session helper
    a1111_webservice.py      # Forge / A1111 WebUI client
    comfyui_webservice.py    # ComfyUI client
    comfyui/                 # ComfyUI node-graph + workflow builders
    webui/                   # WebUI request/response body formats
    controlnet/              # backend-agnostic ControlNet model
  config/              # trimmed JSON-backed config system (Cache / AppConfig)
  util/                # parameter model, shared constants, Size, PIL image helpers
  resources/config/    # the JSON definitions that back Cache / AppConfig
```

## Usage

Images passed in and returned are `PIL.Image` objects. No `QApplication` or event loop is needed.

```python
from PIL import Image
from intrapaint_api.config.cache import Cache
from intrapaint_api.api.comfyui_webservice import ComfyUiWebservice

Cache().set(Cache.PROMPT, 'a corgi astronaut, detailed')
service = ComfyUiWebservice('http://localhost:8188')
# ... call service methods to queue generation, upload images (PIL), download results (PIL), etc.
```

The client still reads generation parameters out of the JSON-backed config singletons
(`Cache` / `AppConfig`), exactly as it does inside IntraPaint. Their values persist to a per-user
data directory (via `platformdirs`, currently under the "IntraPaint"/"centuryglass" app dir — change
that in `intrapaint_api/util/shared_constants.py`, or pass an explicit path to
`Cache(path)` / `AppConfig(path)`, if you want a separate location). Note that with the default
path this shares IntraPaint's own config files.

### Authentication (A1111 / Forge)

IntraPaint prompted for credentials with a Qt login dialog. The standalone client instead accepts a
`credentials_provider` callback, invoked when the server requires auth; it returns a
`(username, password)` pair to try, or `None` to abort:

```python
from intrapaint_api.api.a1111_webservice import A1111Webservice

service = A1111Webservice('http://localhost:7860',
                          credentials_provider=lambda: ('user', 'password'))
```

## What was changed during extraction

Starting from a faithful copy of `src/api`, two rounds of decoupling were applied.

**1. Slim vendor** — dropped the parts of IntraPaint that only exist to render editing UI:

- All `src.` imports were rewritten to `intrapaint_api.`.
- The Qt **input-widget-building** code was removed (it pulled in IntraPaint's entire
  `ui/input_fields` widget library and isn't needed to build/send requests):
  `Parameter.get_input_widget()`, `Config.get_control_widget()`, and
  `ControlParameter.get_input_widget()` / `disconnect_input_widget()`. These classes still hold
  values, validate, and serialize/deserialize exactly as before.
- `LORA_KEY_PATH` (one constant previously imported from `ui/window/extra_network_window`) is now
  defined inline in `api/comfyui/diffusion_workflow_builder.py`.
- `PROJECT_DIR` in `util/shared_constants.py` resolves to this package root so the bundled
  `resources/config/*.json` definitions load correctly.

**2. Qt removal** — eliminated the PySide6 dependency entirely:

- **Images:** `QImage` → `PIL.Image`. `util/visual/image_utils.py` was replaced with a small
  PIL-based module (`image_to_base64` / `image_from_base64` / `image_to_png_bytes` /
  `image_from_bytes`). Downloaded/decoded images are normalized to RGBA. (This dropped `cv2` and
  `numpy`, which were only used by the old image helpers.)
- **`QSize`** → a minimal `Size` value class in `util/geometry.py` (same `.width()` / `.height()`
  API), used both for image dimensions and size-typed config values.
- **Translation:** the `QApplication.translate`-based `_tr()` helpers became no-op shims that return
  the source string (there is no bundled translation catalog).
- **Signals:** `ControlParameter`'s `QObject`/`Signal` became a tiny Qt-free callback object
  (`value_changed.connect(...)` / `.emit(...)` still work for external subscribers).
- **Save debounce:** the `QTimer` that batched config writes now writes synchronously (no event loop
  required).
- **Deleted UI-only support:** the widget-geometry cache in `Cache` (and `util/visual/display_size`),
  the theme/style setup in `AppConfig._adjust_defaults`, `config.get_color()` / `get_keycodes()`,
  and the `LoginModal` dialog (replaced by the `credentials_provider` callback above).

The request-building and networking logic itself was not modified.

## Verifying against a live backend

The extraction preserves the PNG-over-HTTP wire behavior, but image encoding now goes through Pillow
instead of Qt. If you rely on img2img / inpainting / ControlNet, it's worth a real round-trip against
a running ComfyUI or A1111/Forge instance rather than trusting imports alone.

## Possible next step: a fully data-driven library

The remaining coupling is to the JSON-backed config singletons: the client reads generation
parameters from `Cache` / `AppConfig` rather than from arguments. A future pass could have the API
take those parameters as plain dataclasses/dicts, removing the config layer entirely. That's a
larger, interface-changing refactor; the current package already works headless when copied out.
