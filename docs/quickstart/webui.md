# Forge / A1111 WebUI

`A1111Webservice` drives the [Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and its
Forge forks. Start the server with `--api`, which enables the HTTP API this client uses.

## Connect

```python
from sd_backend_client import A1111Webservice

backend = A1111Webservice('http://127.0.0.1:7860')
```

`connect_to_backend(url)` returns the same client when the URL answers as a WebUI, so construct one directly only to
set options it doesn't pass through, such as `generation_timeout`.

### Authentication

A server started with `--api-auth` needs credentials. Pass a `credentials_provider` callback, which the client calls
when the server asks for authentication. It returns a `(username, password)` pair to try, or `None` to give up, which
raises `AuthError`.

```python
backend = A1111Webservice('http://127.0.0.1:7860', credentials_provider=lambda: ('user', 'password'))
```

`connect_to_backend` takes the same `credentials_provider` argument.

## Generate

`DiffusionRequestBody` adds the WebUI's own fields to `DiffusionParams`: batch repeats (`n_iter`), high-res fix
(`enable_hr`, `hr_scale`, ...), inpainting options, styles and scripts. A plain `DiffusionParams` works too, and the
WebUI fills in its own defaults for the rest.

```python
from sd_backend_client import DiffusionRequestBody

body = DiffusionRequestBody(prompt='a corgi astronaut, detailed', negative_prompt='blurry',
                            sampler='dpmpp_2m', scheduler='karras', steps=30, cfg_scale=7.0,
                            width=512, height=512, n_iter=2)
result = backend.submit_txt2img(body).wait()
for image, seed in zip(result.images, result.seeds):
    image.save(f'corgi-{seed}.png')
```

`sampler` and `scheduler` take the shared names that `list_samplers()` and `list_schedulers()` return, so the same
values work on ComfyUI. WebUI display names such as `'Euler a'` are accepted as well.

The WebUI runs one job at a time. The client queues submitted jobs on its own side and sends each when the previous one
finishes, so `submit_*` still returns at once.

## Inpaint

```python
from PIL import Image

from sd_backend_client import InpaintFillOption, ResizeMode

body = DiffusionRequestBody(prompt='a red scarf', init_images=[Image.open('corgi.png')],
                            mask=Image.open('scarf-mask.png'), denoising_strength=0.75,
                            inpainting_fill=InpaintFillOption.ORIGINAL, resize_mode=ResizeMode.CROP_AND_RESIZE)
result = backend.submit_inpaint(body).wait()
```

White or opaque mask pixels mark the region to change, on both backends.

## Progress and cancellation

`poll()` returns a `GenerationProgress` with the server's progress fraction, ETA and a preview frame when the server
sends one.

```python
handle = backend.submit_txt2img(body)
result = handle.wait(on_progress=lambda progress: print(progress.status.value, progress.progress))
```

`handle.cancel()` interrupts the job if it is running, or drops it from the client-side queue if it hasn't been sent
yet.

## Beyond the shared interface

`A1111Webservice` also has WebUI-only methods, such as `interrogate`, `get_styles`, `get_scripts`, `set_config` and the
blocking `txt2img` and `img2img`. Their return types are the WebUI's own formats, which are internal and may change.
See [Clients](../reference/clients.md#sd_backend_client.A1111Webservice).
