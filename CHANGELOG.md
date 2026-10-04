# Changelog

## [0.2.0](https://github.com/centuryglass/sd-api-standalone/compare/v0.1.0...v0.2.0) (2026-10-04)


### ⚠ BREAKING CHANGES

* **comfyui:** treat opaque/white as the region to change in inpainting masks ([#48](https://github.com/centuryglass/sd-api-standalone/issues/48))

### Bug Fixes

* **comfyui:** avoid invalid graphs for ControlNet units without image or model ([#45](https://github.com/centuryglass/sd-api-standalone/issues/45)) ([56be032](https://github.com/centuryglass/sd-api-standalone/commit/56be032301d0290bae656ee314de285f6a22e3d5))
* **comfyui:** make model config auto-detection robust to dot-less entries ([#44](https://github.com/centuryglass/sd-api-standalone/issues/44)) ([2b7e8de](https://github.com/centuryglass/sd-api-standalone/commit/2b7e8def9504c70ec8e86a1a4bfd3b9d37d36195))
* **comfyui:** resize upscale output to the requested size and apply the upscale model ([#49](https://github.com/centuryglass/sd-api-standalone/issues/49)) ([b8489a9](https://github.com/centuryglass/sd-api-standalone/commit/b8489a91c53b6432096736b91c3113bbb34e3605))
* **comfyui:** serialize FreeMemoryRequest in free_memory ([#43](https://github.com/centuryglass/sd-api-standalone/issues/43)) ([6548742](https://github.com/centuryglass/sd-api-standalone/commit/65487421a7d87364467054404461ac7e6ac4111f))
* **comfyui:** treat opaque/white as the region to change in inpainting masks ([#48](https://github.com/centuryglass/sd-api-standalone/issues/48)) ([f024a59](https://github.com/centuryglass/sd-api-standalone/commit/f024a591027e78acc0566564d545a2ec0a874f8e))
* stop creating directories at import time ([#42](https://github.com/centuryglass/sd-api-standalone/issues/42)) ([a9feb6a](https://github.com/centuryglass/sd-api-standalone/commit/a9feb6ae5e40c66a964e442b6807353d2d95f301))
* **webui:** bound the 401 retry loops and validate credentials against /sdapi ([#46](https://github.com/centuryglass/sd-api-standalone/issues/46)) ([02c9bd5](https://github.com/centuryglass/sd-api-standalone/commit/02c9bd50a82e6e4e1307c440164f3bb48a2acf42))
* **webui:** send inpainting masks as opaque grayscale ([#6](https://github.com/centuryglass/sd-api-standalone/issues/6)) ([a60ce40](https://github.com/centuryglass/sd-api-standalone/commit/a60ce4085eef357afa85cb701ab049bbbaec3b7d))
* **webui:** stop request methods mutating the caller's DiffusionRequestBody ([#47](https://github.com/centuryglass/sd-api-standalone/issues/47)) ([b91fc02](https://github.com/centuryglass/sd-api-standalone/commit/b91fc022299fac3e6134270a7cb23f4f2799f4d5))
