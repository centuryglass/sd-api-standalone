# Generation parameters

`DiffusionParams` holds the parameters both backends share, and works with either client. `DiffusionRequestBody`
(WebUI) and `ComfyUIDiffusionParams` (ComfyUI) subclass it to add backend-specific fields; the other backend ignores
those fields.

::: sd_backend_client.DiffusionParams

::: sd_backend_client.DiffusionRequestBody

::: sd_backend_client.ResizeMode

::: sd_backend_client.InpaintFillOption

::: sd_backend_client.ScriptRequestData

::: sd_backend_client.ComfyUIDiffusionParams

::: sd_backend_client.DiffusionUpscalingParams
