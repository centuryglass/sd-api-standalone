# Example scripts

Runnable command-line scripts built on the public `sd_backend_client` API. They work with ComfyUI and WebUI
(A1111 or Forge) unless noted. Run them from the repository root, e.g. `python examples/txt2img.py "a lighthouse"`,
with the package installed (`pip install -e .`). Pillow is the only dependency besides the package's own.

Every script that talks to a server takes `--url`, `--backend {auto,comfyui,webui}`, and for a WebUI started with
`--api-auth`, `--username` and `--password` (or `$SD_UNAME` and `$SD_PASS`). `--url` falls back to `$SD_API_URL`.
Run any script with `--help` for its flags.

| Script | What it does |
| --- | --- |
| `txt2img.py` | Text-to-image. Saves PNGs with A1111-style `parameters` metadata; progress goes to stderr. |
| `img2img.py` | Image-to-image, or inpainting when `--mask` is given. |
| `inpaint.py` | Inpainting with `--mask` required. |
| `rerun.py` | Regenerates an image from the `parameters` metadata of an earlier PNG, with overrides such as `--seed +1`. |
| `backend_info.py` | Lists a server's checkpoints, samplers, upscalers, ControlNet models and more. `--json` for machines. |
| `upscale.py` | Upscales an image by `--scale` or to `--size`, with an optional `--upscaler`. |
| `preprocess.py` | Runs any ControlNet preprocessor (`--list` shows them) and saves the control map. |
| `xy_grid.py` | Sweeps one or two parameters, or a prompt search-and-replace, into a labeled contact sheet. |
| `infinite_zoom.py` | Repeatedly shrinks the frame and inpaints the border into a zoom-out GIF or WebP. |
| `telephone.py` | Interrogate, then img2img, for N rounds. WebUI only. |

`_common.py` holds the helpers the scripts share.
