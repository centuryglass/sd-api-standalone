# A1111 / Forge / ReForge integration tests

Live-API integration tests for `intrapaint_api.api.a1111_webservice.A1111Webservice`.
They talk to a **real, running** WebUI instance rather than mocking HTTP.

## Requirements

- A running A1111 / Forge / ReForge WebUI (`--api` enabled).
- `pip install -r requirements.txt` plus `pytest`.

## Environment

| Variable      | Default                  | Purpose                                        |
|---------------|--------------------------|------------------------------------------------|
| `SD_API_URL`  | `http://127.0.0.1:7860`  | Full base URL (overrides host+port).           |
| `SD_API_PORT` | `7860`                   | Port only, when `SD_API_URL` is unset.         |
| `SD_UNAME`    | —                        | Username, if the server uses `--api-auth`.     |
| `SD_PASS`     | —                        | Password, if the server uses `--api-auth`.     |
| `SD_TEST_OUTPUT_DIR` | `tests/output/`   | Where generation tests save PNGs for inspection.|

Config isolation: `conftest.py` points `Cache` / `AppConfig` at a throwaway temp dir
before anything else instantiates them, so tests never read or clobber the shared
IntraPaint per-user config.

## Running

```bash
# Fast, read-only metadata + auth tests (default):
pytest tests/

# Include the slow generation tests (real diffusion; needs a checkpoint loaded):
pytest tests/ --run-generation
# or
RUN_SD_GENERATION=1 pytest tests/
```

## What's covered

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

### Saved outputs

The generation tests write their results to the output dir (default `tests/output/`,
git-ignored) so you can eyeball fidelity — including the ControlNet input, the
`/controlnet/detect` preview, and the with/without-ControlNet baseline pair.

## Graceful degradation

The suite is designed to stay green across environments: it **skips** (never errors)
when the server is unreachable, when auth is disabled, when ControlNet is absent, or
when the generation flag is off.
