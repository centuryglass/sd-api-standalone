# ControlNet

ControlNet units are backend-agnostic. Add `ControlNetUnit`s to `DiffusionParams.controlnet_units`, and each client
serializes them into its own form. `Backend.list_controlnet_models` and `Backend.get_controlnet_preprocessors` list
the models and preprocessors a server offers.

::: sd_backend_client.ControlNetUnit

::: sd_backend_client.ControlNetModel

::: sd_backend_client.ControlNetPreprocessor

::: sd_backend_client.PreprocessorParams

::: sd_backend_client.ParameterDef

::: sd_backend_client.ControlTypeDef
