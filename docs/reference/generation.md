# Generation handles

Every `Backend.submit_*` method returns a `GenerationHandle` for the queued job. Call `wait()` to block until it
finishes, or `poll()` it for progress.

::: sd_backend_client.GenerationHandle

::: sd_backend_client.GenerationStatus

::: sd_backend_client.GenerationProgress

::: sd_backend_client.GenerationResult

::: sd_backend_client.ProgressCallback
