# Changelog

## [0.2.0](https://github.com/centuryglass/sd-backend-client/compare/v0.1.0...v0.2.0) (2026-10-06)


### ⚠ BREAKING CHANGES

* remove dead code and UI leftovers from the IntraPaint extraction ([#55](https://github.com/centuryglass/sd-backend-client/issues/55))
* rename the package to sd_backend_client ([#52](https://github.com/centuryglass/sd-backend-client/issues/52))
* **comfyui:** treat opaque/white as the region to change in inpainting masks ([#48](https://github.com/centuryglass/sd-backend-client/issues/48))

### Features

* ship a py.typed marker so type checkers read the package's annotations ([#53](https://github.com/centuryglass/sd-backend-client/issues/53)) ([f1f9533](https://github.com/centuryglass/sd-backend-client/commit/f1f95331216efe771adc399f7812fb80e913fa04)), closes [#5](https://github.com/centuryglass/sd-backend-client/issues/5)


### Bug Fixes

* **comfyui:** avoid invalid graphs for ControlNet units without image or model ([#45](https://github.com/centuryglass/sd-backend-client/issues/45)) ([56be032](https://github.com/centuryglass/sd-backend-client/commit/56be032301d0290bae656ee314de285f6a22e3d5))
* **comfyui:** make model config auto-detection robust to dot-less entries ([#44](https://github.com/centuryglass/sd-backend-client/issues/44)) ([2b7e8de](https://github.com/centuryglass/sd-backend-client/commit/2b7e8def9504c70ec8e86a1a4bfd3b9d37d36195))
* **comfyui:** resize upscale output to the requested size and apply the upscale model ([#49](https://github.com/centuryglass/sd-backend-client/issues/49)) ([b8489a9](https://github.com/centuryglass/sd-backend-client/commit/b8489a91c53b6432096736b91c3113bbb34e3605))
* **comfyui:** serialize FreeMemoryRequest in free_memory ([#43](https://github.com/centuryglass/sd-backend-client/issues/43)) ([6548742](https://github.com/centuryglass/sd-backend-client/commit/65487421a7d87364467054404461ac7e6ac4111f))
* **comfyui:** treat opaque/white as the region to change in inpainting masks ([#48](https://github.com/centuryglass/sd-backend-client/issues/48)) ([f024a59](https://github.com/centuryglass/sd-backend-client/commit/f024a591027e78acc0566564d545a2ec0a874f8e))
* stop creating directories at import time ([#42](https://github.com/centuryglass/sd-backend-client/issues/42)) ([a9feb6a](https://github.com/centuryglass/sd-backend-client/commit/a9feb6ae5e40c66a964e442b6807353d2d95f301))
* **webui:** bound the 401 retry loops and validate credentials against /sdapi ([#46](https://github.com/centuryglass/sd-backend-client/issues/46)) ([02c9bd5](https://github.com/centuryglass/sd-backend-client/commit/02c9bd50a82e6e4e1307c440164f3bb48a2acf42))
* **webui:** read ControlNet module details from the A1111 extension's module_detail key ([#56](https://github.com/centuryglass/sd-backend-client/issues/56)) ([65eae80](https://github.com/centuryglass/sd-backend-client/commit/65eae80d8613562d1be3de2ae5c787703f1cef2b)), closes [#50](https://github.com/centuryglass/sd-backend-client/issues/50)
* **webui:** send inpainting masks as opaque grayscale ([#6](https://github.com/centuryglass/sd-backend-client/issues/6)) ([a60ce40](https://github.com/centuryglass/sd-backend-client/commit/a60ce4085eef357afa85cb701ab049bbbaec3b7d))
* **webui:** stop request methods mutating the caller's DiffusionRequestBody ([#47](https://github.com/centuryglass/sd-backend-client/issues/47)) ([b91fc02](https://github.com/centuryglass/sd-backend-client/commit/b91fc022299fac3e6134270a7cb23f4f2799f4d5))


### Documentation

* add contributing guide, security policy, issue and PR templates ([#54](https://github.com/centuryglass/sd-backend-client/issues/54)) ([78df9af](https://github.com/centuryglass/sd-backend-client/commit/78df9afc06f679317675843d40a12ec0016ac409)), closes [#32](https://github.com/centuryglass/sd-backend-client/issues/32)


### Code Refactoring

* remove dead code and UI leftovers from the IntraPaint extraction ([#55](https://github.com/centuryglass/sd-backend-client/issues/55)) ([adf24ab](https://github.com/centuryglass/sd-backend-client/commit/adf24ab1ec7207acedfc3587be6399154247c966)), closes [#27](https://github.com/centuryglass/sd-backend-client/issues/27)
* rename the package to sd_backend_client ([#52](https://github.com/centuryglass/sd-backend-client/issues/52)) ([4e4ef24](https://github.com/centuryglass/sd-backend-client/commit/4e4ef242aaab84a4d725a3a24718f3ef7fd56ff5)), closes [#8](https://github.com/centuryglass/sd-backend-client/issues/8)
