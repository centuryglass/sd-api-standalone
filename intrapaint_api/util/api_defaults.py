"""
Defines a set of sensible defaults for Stable Diffusion API fields.
"""
from intrapaint_api.util.geometry import Size

GENERATION_SIZE_DEFAULT = 512

# TODO: do we actually want all of these, or should we just define mostdefaults in diffusion_request_body?
INPAINTING_MASK_BLUR_DEFAULT = 4
GUIDANCE_SCALE_DEFAULT = 9.0
BATCH_SIZE_DEFAULT = 1
BATCH_COUNT_DEFAULT = 1
EDIT_MODE_DEFAULT = "Text to Image"  # Also valid: "Image to Image", "Inpaint"
MASKED_CONTENT_DEFAULT = "original"  # Also valid: "fill", "latent noise", "latent nothing"
SAMPLING_STEPS_DEFAULT = 30
DENOISING_STRENGTH_DEFAULT = 0.5
SAMPLING_METHOD_DEFAULT = "Euler a"
SCHEDULER_DEFAULT = "default"
SEED_DEFAULT = -1
INPAINT_FULL_RES_DEFAULT = True
INPAINT_FULL_RES_PADDING_DEFAULT = 32
CLIP_SKIP_DEFAULT = 1

UPSCALING_DENOISING_STRENGTH_DEFAULT = 0.3
UPSCALING_STEP_COUNT_DEFAULT = 30
