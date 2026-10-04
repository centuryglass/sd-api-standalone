"""
Accesses ComfyUI through its REST API, providing access to image generation and editing through Stable Diffusion.
"""
import json
import logging
import os
import uuid
from contextlib import contextmanager
from copy import deepcopy
from enum import StrEnum, Enum
from typing import cast, Optional, Any, Generator, TYPE_CHECKING

import binascii
import websocket
from PIL import Image, ImageChops
from pydantic import BaseModel, ValidationError

from intrapaint_api.api.comfyui.basic_upscale_workflow_builder import build_basic_upscaling_workflow
from intrapaint_api.api.comfyui.comfyui_types import QueueAdditionRequest, QueueAdditionResponse, QueueDeletionRequest, \
    ImageFileReference, PromptExecOutputs, NodeInfoResponse, SystemStatResponse, ImageUploadParams, MaskUploadParams, \
    IMAGE_UPLOAD_FILE_NAME, ImageUploadResponse, QueueInfoResponse, ACTIVE_QUEUE_KEY, \
    PENDING_QUEUE_KEY, QueueHistoryResponse, PromptHistory, FreeMemoryRequest
from intrapaint_api.api.comfyui.controlnet_comfyui_utils import get_all_preprocessors
from intrapaint_api.api.comfyui.diffusion_workflow_builder import DiffusionWorkflowBuilder
from intrapaint_api.api.comfyui.latent_upscale_workflow_builder import LatentUpscaleWorkflowBuilder
from intrapaint_api.api.comfyui.nodes.ksampler_node import KSAMPLER_NAME
from intrapaint_api.api.comfyui.nodes.ultimate_upscale_node import ULTIMATE_UPSCALE_NODE_NAME
from intrapaint_api.api.comfyui.preprocessor_preview_workflow_builder import PreprocessorPreviewWorkflowBuilder
from intrapaint_api.api.shared_data.api_datatypes import DiffusionUpscalingParams
from intrapaint_api.api.shared_data.controlnet.controlnet_category_builder import ControlNetCategoryBuilder
from intrapaint_api.api.shared_data.controlnet.controlnet_constants import ControlTypeDef
from intrapaint_api.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor
from intrapaint_api.api.shared_data.controlnet.controlnet_unit import ControlNetUnit
from intrapaint_api.api.shared_data.diffusion_params import DiffusionParams
from intrapaint_api.api.webservice import WebService, MULTIPART_FORM_DATA_TYPE
from intrapaint_api.util.geometry import Size
from intrapaint_api.util.visual.image_utils import image_to_png_bytes, image_from_bytes, image_from_base64, ImageKey, \
    get_image_key, mask_to_grayscale

if TYPE_CHECKING:
    from intrapaint_api.api.comfyui.comfyui_generation_handle import ComfyGenerationHandle

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """Identifies login failures."""


UPSCALE_SCRIPT = 'ultimate sd upscale'
DEFAULT_TIMEOUT = 30
EXTENDED_TIMEOUT = 90
TYPE_PNG_IMAGE = 'image/png'
INTRAPAINT_UPLOAD_SUBFOLDER = 'IntraPaint'
LORA_EXTENSION = '.safetensors'

TILE_PREPROCESSOR_NODE_NAME = 'TilePreprocessor'
TILE_PREPROCESSOR_DOWNSAMPLING_PARAM_KEY = 'pyrUp_iters'
TILE_PREPROCESSOR_RESOLUTION_KEY = 'resolution'

# Keys used when extracting data from the KSampler node definition:
SAMPLER_OPTION_KEY = 'sampler_name'
SCHEDULER_OPTION_KEY = 'scheduler'


class ComfyEndpoints:
    """REST API endpoint constants."""
    # GET:
    WEBSOCKET = '/ws'
    EMBEDDINGS = '/embeddings'
    MODELS = '/models'
    EXTENSIONS = '/extensions'
    VIEW_IMAGE = '/view'
    SYSTEM_STATS = '/system_stats'
    OBJECT_INFO = '/object_info'
    # POST:
    IMG_UPLOAD = '/upload/image'
    MASK_UPLOAD = '/upload/mask'
    INTERRUPT = '/interrupt'
    FREE = '/free'
    # Both:
    PROMPT = '/prompt'
    QUEUE = '/queue'
    HISTORY = '/history'


class ComfyModelType(StrEnum):
    """Model type names, as returned by get_models.  Hardcoded here because it's very unlikely the core items on this
       list will change, and we're going to need to reference them to do things like load LoRA options."""
    CHECKPOINT = 'checkpoints'
    CONFIG = 'configs'
    LORA = 'loras'
    VAE = 'vae'
    CLIP = 'clip'
    EMBEDDING = 'embeddings'
    CONTROLNET = 'controlnet'
    CUSTOM_NODES = 'custom_nodes'
    HYPERNETWORKS = 'hypernetworks'
    UPSCALING = 'upscale_models'
    # Everything below this point is only included for completeness, and is unlikely to ever see support in IntraPaint
    # unless someone requests it.
    DIFFUSION_MODEL = 'diffusion_models'
    CLIP_VISION = 'clip_vision'
    STYLE_MODEL = 'style_models'
    DIFFUSER = 'diffusers'
    VAE_APPROX = 'vae_approx'
    GLIGEN = 'gligen'
    PHOTOMAKER = 'photomaker'
    CLASSIFIERS = 'classifiers'
    ANIMATE_DIFF = 'AnimateDiffEvolved_Models'
    ANIMATE_DIFF_LORA = 'AnimateDiffMotion_LoRA'
    VIDEO_FORMATS = 'video_formats'
    IP_ADAPTER = 'ipadapter'


class AsyncTaskStatus(Enum):
    """Represents a queued task's status."""
    PENDING = 0
    ACTIVE = 1
    FINISHED = 2
    FAILED = 3
    NOT_FOUND = 4


class AsyncTaskProgress(BaseModel):
    """The status of an async ComfyUI task, including queue index and generated image data when relevant."""
    status: AsyncTaskStatus
    index: Optional[int] = None  # Only used if statis is PENDING
    outputs: Optional[PromptExecOutputs] = None  # Only used if status is FINISHED


class ComfyUiWebservice(WebService):
    """
    ComfyUiWebservice provides access to Stable Diffusion through the ComfyUI REST API.
    """

    def __init__(self, url: str) -> None:
        super().__init__(url)
        self._preprocessor_cache: Optional[list[ControlNetPreprocessor]] = None
        self._ksampler_info: Optional[NodeInfoResponse] = None
        self._client_id = str(uuid.uuid4())
        self._uploaded_images: dict[ImageKey, ImageFileReference] = {}

    # Loading available options and settings:

    def _get_ksampler_info_caching(self) -> NodeInfoResponse:
        ksampler_info = self._ksampler_info
        if ksampler_info is None:
            info_endpoint = f'{ComfyEndpoints.OBJECT_INFO}/{KSAMPLER_NAME}'
            ksampler_info = NodeInfoResponse.model_validate(self.get(info_endpoint).json()[KSAMPLER_NAME])
            self._ksampler_info = ksampler_info
        return ksampler_info

    def get_sampler_names(self) -> list[str]:
        """Gets the list of sampling method names from KSampler node info."""
        ksampler_info = self._get_ksampler_info_caching()
        required_inputs = ksampler_info.input.required
        # After validation the option param is a tuple whose first element is the list of option names.
        assert SAMPLER_OPTION_KEY in required_inputs and isinstance(required_inputs[SAMPLER_OPTION_KEY], tuple)
        return cast(list[str], required_inputs[SAMPLER_OPTION_KEY][0])

    def get_scheduler_names(self) -> list[str]:
        """Gets the list of sampling scheduler names from KSampler node info."""
        ksampler_info = self._get_ksampler_info_caching()
        required_inputs = ksampler_info.input.required
        assert SCHEDULER_OPTION_KEY in required_inputs and isinstance(required_inputs[SCHEDULER_OPTION_KEY], tuple)
        return cast(list[str], required_inputs[SCHEDULER_OPTION_KEY][0])

    def is_node_available(self, node_name: str) -> bool:
        """Checks if a node with the given name is available."""
        info_endpoint = f'{ComfyEndpoints.OBJECT_INFO}/{node_name}'
        # Only the presence of the node key matters here, so keep the raw response as a plain dict.
        node_info: dict[str, Any] = self.get(info_endpoint).json()
        return node_name in node_info

    def get_embeddings(self) -> list[str]:
        """Returns the list of available embedding files."""
        return self.get(ComfyEndpoints.EMBEDDINGS, timeout=DEFAULT_TIMEOUT).json()

    def get_model_types(self) -> list[str]:
        """Returns the list of available model types."""
        return cast(list[str], self.get(ComfyEndpoints.MODELS, timeout=DEFAULT_TIMEOUT).json())

    def get_extensions(self) -> list[str]:
        """Returns the list of installed extension files."""
        return cast(list[str], self.get(ComfyEndpoints.EXTENSIONS, timeout=DEFAULT_TIMEOUT).json())

    def get_system_stats(self) -> SystemStatResponse:
        """Returns information about the system and device running Stable Diffusion."""
        return SystemStatResponse.model_validate(
                    self.get(ComfyEndpoints.SYSTEM_STATS, timeout=DEFAULT_TIMEOUT).json())

    def get_models(self, model_type: ComfyModelType) -> list[str]:
        """Returns the list of available models, given a particular model type."""
        endpoint = f'{ComfyEndpoints.MODELS}/{model_type.value}'
        return cast(list[str], self.get(endpoint, timeout=DEFAULT_TIMEOUT).json())

    def get_sd_checkpoints(self) -> list[str]:
        """Returns the list of available Stable Diffusion models."""
        return self.get_models(ComfyModelType.CHECKPOINT)

    def get_vae_models(self) -> list[str]:
        """Returns the list of available Stable Diffusion VAE models."""
        return self.get_models(ComfyModelType.VAE)

    def get_controlnet_models(self) -> list[str]:
        """Returns the list of available ControlNet models."""
        return self.get_models(ComfyModelType.CONTROLNET)

    def get_lora_models(self) -> list[str]:
        """Returns the list of available LoRA models."""
        return self.get_models(ComfyModelType.LORA)

    def get_hypernetwork_models(self) -> list[str]:
        """Returns the list of available Hypernetwork models."""
        return self.get_models(ComfyModelType.HYPERNETWORKS)

    def get_controlnet_preprocessors(self, update_cache=False) -> list[ControlNetPreprocessor]:
        """Scans all nodes for valid preprocessor nodes, and returns the list of parameterized options."""
        if update_cache or self._preprocessor_cache is None:
            raw_node_data = self.get(ComfyEndpoints.OBJECT_INFO, timeout=DEFAULT_TIMEOUT).json()
            node_data = {}
            for name, info in raw_node_data.items():
                try:
                    node_data[name] = NodeInfoResponse.model_validate(info)
                except ValidationError as err:
                    # ComfyUI installs an open-ended set of core + custom nodes; those that don't match the schema
                    # can't be usable ControlNet preprocessors anyway, so skip them instead of failing discovery.
                    logger.debug(f'Skipping node {name!r}, does not match NodeInfoResponse schema: {err}')
            self._preprocessor_cache = get_all_preprocessors(node_data)
        assert self._preprocessor_cache is not None
        return deepcopy(self._preprocessor_cache)

    def get_controlnet_type_categories(self, preprocessors: Optional[list[ControlNetPreprocessor]] = None
                                       ) -> dict[str, ControlTypeDef]:
        """Gets the set of valid ControlNet proeprocessor/model categories, taking into account available options and
           API category definitions if possible."""
        if preprocessors is None:
            preprocessors = self.get_controlnet_preprocessors()
        preprocessor_names = [module.name for module in preprocessors]
        preprocessor_categories: dict[str, str] = {}
        for module in preprocessors:
            preprocessor_categories[module.name] = module.category_name
        model_names = self.get_controlnet_models()
        control_type_builder = ControlNetCategoryBuilder(preprocessor_names, model_names, preprocessor_categories,
                                                         None)
        return control_type_builder.get_control_types()

    # File I/O:
    def _upload_image_file(self,
                           image: Image.Image | str,
                           endpoint:str, name: Optional[str] = None,
                           subfolder: Optional[str] = None,
                           temp=False, overwrite=True,
                           original_ref: Optional[ImageFileReference] = None) -> ImageFileReference:
        if isinstance(image, str):
            if os.path.isfile(image):
                with open(image, 'rb') as image_file:
                    image = image_from_bytes(image_file.read())
            else:
                try:
                    image = image_from_base64(image)
                except binascii.Error:
                    raise ValueError(f"invalid image string {image}: expected base64")
        assert isinstance(image, Image.Image)
        image_key = get_image_key(image)
        if image_key in self._uploaded_images:
            return self._uploaded_images[image_key]
        resolved_subfolder = subfolder if subfolder is not None else INTRAPAINT_UPLOAD_SUBFOLDER
        if original_ref is not None:
            # Mask uploads must reference the original image via `original_ref` (a JSON string), which ComfyUI's
            # /upload/mask endpoint json.loads(); omitting it makes the server fail on json.loads(None).
            body: ImageUploadParams = MaskUploadParams(type='temp' if temp else 'input',
                                                       subfolder=resolved_subfolder,
                                                       original_ref=original_ref.model_dump_json(exclude_none=True))
        else:
            body = ImageUploadParams(type='temp' if temp else 'input', subfolder=resolved_subfolder)
        if overwrite:
            body.overwrite = '1'
        image_data = image_to_png_bytes(image)
        if name is None:
            name = 'src_image.png'
        elif not name.endswith('.png'):
            name = f'{name}.png'
        files = {IMAGE_UPLOAD_FILE_NAME: (name, image_data, TYPE_PNG_IMAGE)}
        res_json = self.post(endpoint,
                             body=body.model_dump(exclude_none=True),
                             body_format=MULTIPART_FORM_DATA_TYPE,
                             files=files,
                             timeout=EXTENDED_TIMEOUT).json()
        res_body = ImageUploadResponse.model_validate(res_json)
        file_ref = ImageFileReference(filename = res_body.name,
                                      subfolder = res_body.subfolder or '',
                                      type = res_body.type)
        self._uploaded_images[image_key] = file_ref
        # Update overwritten references:
        old_keys: list[ImageKey] = []
        for key in self._uploaded_images:
            if key != image_key and self._uploaded_images[key] == file_ref:
                old_keys.append(key)
        for key in old_keys:
            del self._uploaded_images[key]
        return file_ref

    def upload_image(self, image: Image.Image | str,
                     name: Optional[str] = None, subfolder: Optional[str] = None,
                     temp=False, overwrite=True) -> ImageFileReference:
        """Uploads an image for img2img, inpainting, ControlNet, etc."""
        return self._upload_image_file(image, ComfyEndpoints.IMG_UPLOAD, name, subfolder, temp, overwrite)

    def upload_mask(self, mask: Image.Image | str, ref_image: ImageFileReference,
                    subfolder: Optional[str] = None,
                    overwrite=True) -> ImageFileReference:
        """Upload an inpainting mask for a particular image.

        The ref_image parameter should contain data returned by a previous upload_image request. Mask size must match
        original image size.

        `mask` follows the `DiffusionParams.mask` convention: opaque/white pixels mark the region to change. It is
        converted with `mask_to_grayscale` and uploaded as an RGBA PNG whose alpha is `255 - luminance`, because
        ComfyUI's `LoadImageMask` (channel `alpha`) treats transparent pixels as the region to change. Callers must
        not pre-invert the mask.
        """
        if isinstance(mask, str):
            if os.path.isfile(mask):
                with open(mask, 'rb') as mask_file:
                    mask = image_from_bytes(mask_file.read())
            else:
                try:
                    mask = image_from_base64(mask)
                except binascii.Error as err:
                    raise ValueError(f"invalid mask string {mask}: expected base64") from err
        comfy_mask = Image.new('RGBA', mask.size, (0, 0, 0, 255))
        comfy_mask.putalpha(ImageChops.invert(mask_to_grayscale(mask)))
        return self._upload_image_file(comfy_mask, ComfyEndpoints.MASK_UPLOAD, f'mask_{ref_image.filename}',
                                       subfolder, False, overwrite, original_ref=ref_image)


    def download_images(self, image_refs: list[ImageFileReference]) -> list[Image.Image]:
        """Download a list of images from ComfyUI as RGBA PIL images."""
        images: list[Image.Image] = []
        for image_ref in image_refs:
            try:
                image_res = self.get(ComfyEndpoints.VIEW_IMAGE, url_params=image_ref.model_dump(exclude_none=True),
                                     timeout=EXTENDED_TIMEOUT)
                images.append(image_from_bytes(image_res.content))
            except (IOError, ValueError, RuntimeError) as err:
                logger.error(f'Skipping image {image_ref}: {err}')
        return images

    # Running ComfyUI workflows:

    def _build_diffusion_body(self,
                              diffusion_params: DiffusionParams,
                              workflow_builder: Optional[DiffusionWorkflowBuilder] = None) -> DiffusionWorkflowBuilder:
        """Apply cached parameters to begin building a ComfyUI workflow."""
        if workflow_builder is None:
            workflow_builder = DiffusionWorkflowBuilder()
        workflow_builder.load_diffusion_parameters(diffusion_params)
        if diffusion_params.seed is not None:
            workflow_builder.seed = diffusion_params.seed
        config_names = self.get_models(ComfyModelType.CONFIG)
        if workflow_builder.model_config_path not in config_names:
            # Check available config, and if one matches the stable diffusion model name, use that one:
            workflow_builder.model_config_path = None
            model_name = workflow_builder.sd_model
            if '.' in model_name:
                ext_idx = model_name.rindex('.')
                model_name = model_name[:ext_idx]
            for config_file in config_names:
                config_ext_idx = config_file.index('.')
                if config_ext_idx != -1:
                    if config_file[:config_ext_idx] == model_name:
                        workflow_builder.model_config_path = config_file
        return workflow_builder

    def _prepare_controlnet_data(self, workflow_builder: DiffusionWorkflowBuilder,
                                 controlnet_units: list[ControlNetUnit]) -> None:
        """Loads ControlNet units from the cache into a workflow builder.

        Parameters:
        ----------
        workflow_builder: DiffusionWorkflowBuilder
            Workflow builder object where any active ControlNet units will be defined as ComfyUI nodes.
        controlnet_units: list[ControlNetUnit]: list of ControlNet units to add to the workflow.

        """
        for i in range(len(controlnet_units)):
            control_unit = controlnet_units[i]
            model_name = None if control_unit.model is None else control_unit.model.full_model_name
            preprocessor = control_unit.preprocessor
            if preprocessor is None and model_name is None:
                logger.info(f'Skipping unit {i}, no model or preprocessor set')
                continue
            elif model_name is None:
                # ControlNetApplyAdvanced requires a control_net input, so model-free units can't be applied.
                preprocessor_name = preprocessor.typedef.name if preprocessor is not None else None
                logger.warning(f'Skipping unit {i} with preprocessor {preprocessor_name}: ComfyUI requires a'
                               ' ControlNet model for every unit')
                continue

            control_image_ref: Optional[ImageFileReference] = None
            if control_unit.image is not None:
                control_image_ref = self.upload_image(control_unit.image, f"control_{i}")
            workflow_builder.add_controlnet_unit(model_name, preprocessor, control_image_ref,
                                                 float(control_unit.control_strength),
                                                 float(control_unit.control_start),
                                                 float(control_unit.control_end))

    def txt2img(self, diffusion_params: DiffusionParams) -> QueueAdditionResponse:
        """Queues an async text-to-image job with the ComfyUI server.

        Most parameters are read directly from the cache, where they should have been written from UI inputs. Calling
        this method will update the LAST_SEED value in the cache.

        Parameters:
        -----------
        diffusion_params: DiffusionParams
            Diffusion parameters to use for the txt2img operation.
        Returns
        -------
        comfy_type.QueueAdditionResponse
            Information needed to track the async task and download the resulting images once it finishes.
        """
        workflow_builder = self._build_diffusion_body(diffusion_params)
        if workflow_builder.denoising_strength != 1.0:
            workflow_builder.denoising_strength = 1.0
        if diffusion_params.controlnet_units is not None:
            self._prepare_controlnet_data(workflow_builder, diffusion_params.controlnet_units)
        prompt = workflow_builder.build_workflow().get_workflow_dict()
        body = QueueAdditionRequest(prompt=prompt, client_id=self._client_id)
        res = QueueAdditionResponse.model_validate(self.post(ComfyEndpoints.PROMPT, body=body.model_dump(exclude_none=True),
                                                    timeout=DEFAULT_TIMEOUT).json())
        res.seed = workflow_builder.seed
        return res

    def _generate(self, diffusion_params: DiffusionParams) -> QueueAdditionResponse:
        workflow_builder = self._build_diffusion_body(diffusion_params)
        if diffusion_params.init_images is not None and len(diffusion_params.init_images) > 0:
            image: Image.Image = diffusion_params.init_images[0]
            image_reference = self.upload_image(image)
            workflow_builder.set_source_image_from_reference(image_reference)
            if diffusion_params.mask is not None:
                mask_reference = self.upload_mask(diffusion_params.mask, image_reference)
                workflow_builder.set_mask_from_reference(mask_reference)
        if diffusion_params.controlnet_units is not None:
            self._prepare_controlnet_data(workflow_builder, diffusion_params.controlnet_units)
        prompt = workflow_builder.build_workflow().get_workflow_dict()
        body = QueueAdditionRequest(prompt=prompt, client_id=self._client_id)
        res = QueueAdditionResponse.model_validate(self.post(ComfyEndpoints.PROMPT, body=body.model_dump(exclude_none=True),
                                                   timeout=DEFAULT_TIMEOUT).json())
        res.seed = workflow_builder.seed
        return res

    def img2img(self, diffusion_params: DiffusionParams) -> QueueAdditionResponse:
        """Queues an async image-to-image job with the ComfyUI server.

        Parameters:
        -----------
        diffusion_params: DiffusionParams
            Diffusion parameters to use for the img2img operation.
        Returns
        -------
        comfy_type.QueueAdditionResponse
            Information needed to track the async task and download the resulting images once it finishes.
        """
        if diffusion_params.init_images is None or len(diffusion_params.init_images) == 0:
            raise ValueError("Must set an init image in diffusion_params for img2img")
        return self._generate(diffusion_params)

    def inpaint(self,  diffusion_params: DiffusionParams) -> QueueAdditionResponse:
        """Queues an async inpainting job with the ComfyUI server.

        Parameters:
        -----------
        diffusion_params: DiffusionParams
            Diffusion parameters to use for the inpainting operation.
        Returns
        -------
        comfy_type.QueueAdditionResponse
            Information needed to track the async task and download the resulting images once it finishes.
        """

        if diffusion_params.init_images is None or len(diffusion_params.init_images) == 0:
            raise ValueError("Must set an init image in diffusion_params for inpainting")
        if diffusion_params.mask is None:
            raise ValueError("Must set a mask in diffusion_params for inpainting.")
        return self._generate(diffusion_params)


    def controlnet_preprocessor_preview(self, image: Image.Image, mask: Image.Image,
                                        preprocessor: ControlNetPreprocessor) -> QueueAdditionResponse:
        """Runs a minimal workflow to load a ControlNet preprocessor preview."""
        image_reference = self.upload_image(image) if preprocessor.has_image_input else None
        if image_reference is not None and preprocessor.has_mask_input:
            mask_reference = self.upload_mask(mask, image_reference)
        else:
            mask_reference = None
        workflow_builder = PreprocessorPreviewWorkflowBuilder(preprocessor)
        workflow = workflow_builder.build_workflow(image_reference, mask_reference)
        prompt = workflow.get_workflow_dict()
        body = QueueAdditionRequest(prompt=prompt, client_id=self._client_id)
        return QueueAdditionResponse.model_validate(self.post(ComfyEndpoints.PROMPT, body=body.model_dump(exclude_none=True),
                                                    timeout=DEFAULT_TIMEOUT).json())

    def _validate_tile_controlnet(self, tile_control_unit: Optional[ControlNetUnit]) -> Optional[ControlNetUnit]:
        """Return the tile ControlNet unit only if it's fully specified and installed on the server, else None."""
        if tile_control_unit is None:
            return None
        if (tile_control_unit.model is None
                or tile_control_unit.preprocessor is None
                or float(tile_control_unit.control_strength) == 0.0
                or float(tile_control_unit.control_start) >= float(tile_control_unit.control_end)):
            return None
        models = self.get_controlnet_models()
        preprocessors = [preprocessor.name for preprocessor in self.get_controlnet_preprocessors()]
        if (tile_control_unit.model.full_model_name not in models
                or tile_control_unit.preprocessor.typedef.name not in preprocessors):
            return None
        return tile_control_unit

    def upscale(self, image: Image.Image, width: int, height: int,
                upscale_params: Optional[DiffusionUpscalingParams] = None) -> QueueAdditionResponse:
        """Upscale an image using an upscaling model and/or a latent updcaling workflow."""
        if upscale_params is None:
            upscale_params = DiffusionUpscalingParams()
        upscale_multiplier = max(width / image.width, height / image.height)
        if upscale_multiplier <= 1.0:
            raise ValueError(f'Requested size {width}x{height} must exceed the source size '
                             f'{image.width}x{image.height} in at least one dimension')

        image_reference = self.upload_image(image)

        # Check for valid upscaling model:
        upscale_model: Optional[str] = upscale_params.upscaling_mode
        upscale_model_options = self.get_models(ComfyModelType.UPSCALING)
        if upscale_model not in upscale_model_options:
            upscale_model = None


        if upscale_params.use_stable_diffusion_upscaling:
            # Check for the "Ultimate SD Upscale" node:
            ultimate_upscale_available = (upscale_params.use_ultimate_upscale_script
                                          and self.is_node_available(ULTIMATE_UPSCALE_NODE_NAME))

            # Validate the optional tile ControlNet unit against the models/preprocessors the server actually has:
            tile_control_unit = self._validate_tile_controlnet(upscale_params.tile_controlnet)

            workflow_builder = LatentUpscaleWorkflowBuilder(image_reference, upscale_multiplier, Size(width, height),
                                                            Size(upscale_params.tile_width, upscale_params.tile_height),
                                                            ultimate_upscale_available,
                                                            upscale_model,
                                                            tile_control_unit)
            # Populate the core diffusion pass (prompt, seed, cfg, sampler, checkpoint, ...), then override the
            # upscale-specific denoising strength / step count, which take precedence over diffusion_params:
            self._build_diffusion_body(upscale_params.diffusion_params, workflow_builder)
            workflow_builder.denoising_strength = upscale_params.denoising_strength
            workflow_builder.steps = upscale_params.step_count
            workflow_builder.mask_blur = upscale_params.mask_blur
            workflow_builder.tile_padding = upscale_params.tile_padding
            workflow_builder.redraw_mode = upscale_params.redraw_mode
            workflow_builder.force_uniform_tiles = upscale_params.force_uniform_tiles
            workflow_builder.tiled_decode = upscale_params.tiled_decode
            workflow_builder.seam_fix_mode = upscale_params.seam_fix_mode
            workflow_builder.seam_fix_denoise = upscale_params.seam_fix_denoise
            workflow_builder.seam_fix_width = upscale_params.seam_fix_width
            workflow_builder.seam_fix_mask_blur = upscale_params.seam_fix_mask_blur
            workflow_builder.seam_fix_padding = upscale_params.seam_fix_padding
            workflow_node_graph = workflow_builder.build_workflow()

        else:  # Basic upscaling workflow:
            if upscale_model is None:
                raise RuntimeError(f'No valid upscaling model, provided value was "{upscale_params.upscaling_mode}"')
            workflow_node_graph = build_basic_upscaling_workflow(image_reference, upscale_params.upscaling_mode,
                                                                Size(width, height))

        prompt = workflow_node_graph.get_workflow_dict()
        body = QueueAdditionRequest(prompt=prompt, client_id=self._client_id)
        res = QueueAdditionResponse.model_validate(self.post(ComfyEndpoints.PROMPT, body=body.model_dump(exclude_none=True),
                                                    timeout=DEFAULT_TIMEOUT).json())
        return res

    # Queued/in-progress workflow status and control:

    def get_queue_info(self) -> QueueInfoResponse:
        """Get info on the set of queued jobs."""
        res_body = self.get(ComfyEndpoints.QUEUE, timeout=DEFAULT_TIMEOUT).json()
        for queue_key in [ACTIVE_QUEUE_KEY, PENDING_QUEUE_KEY]:
            assert queue_key in res_body
            queue_list = res_body[queue_key]
            assert isinstance(queue_list, list)
            res_body[queue_key] = [tuple(queue_entry) for queue_entry in queue_list]
        return QueueInfoResponse.model_validate(res_body)

    def check_queue_entry(self, entry_uuid: str, task_number: int) -> AsyncTaskProgress:
        """Returns the status of a queued task, along with associated data when relevant."""
        endpoint = f'{ComfyEndpoints.HISTORY}/{entry_uuid}'
        history_response = self.get(endpoint, timeout=DEFAULT_TIMEOUT).json()
        if entry_uuid in history_response:
            entry_history = PromptHistory.model_validate(history_response[entry_uuid])
            if entry_history.status.status_str == 'error':
                return AsyncTaskProgress(status=AsyncTaskStatus.FAILED)
            if entry_history.status.completed:
                images = []
                for output_data in entry_history.outputs.values():
                    if output_data.images is not None:
                        for reference in output_data.images:
                            images.append(cast(ImageFileReference, reference))
                progress = AsyncTaskProgress(status=AsyncTaskStatus.FINISHED, outputs=PromptExecOutputs(images=images))
                return progress
        queue_info = self.get_queue_info()
        for running_task in queue_info.queue_running:
            if running_task[1] == entry_uuid:
                return AsyncTaskProgress(status=AsyncTaskStatus.ACTIVE)
        queue_index = 0
        task_found = False
        for pending_task in queue_info.queue_pending:
            if pending_task[0] < task_number:
                queue_index += 1
            elif pending_task[0] == task_number:
                task_found = True
        if task_found:
            return AsyncTaskProgress(status=AsyncTaskStatus.PENDING, index=queue_index)
        return AsyncTaskProgress(status=AsyncTaskStatus.NOT_FOUND)

    def interrupt(self, task_id: Optional[str] = None) -> None:
        """Stops the active workflow, and removes a task from the queue if task_id is not None."""
        if task_id is not None:
            queue_removal_body = QueueDeletionRequest(delete=[task_id])
            self.post(ComfyEndpoints.QUEUE, body=queue_removal_body.model_dump())
        self.post(ComfyEndpoints.INTERRUPT, body=None, timeout=DEFAULT_TIMEOUT)

    def remove_from_queue(self, task_id: str) -> None:
        """Drop a still-queued (pending) task *without* interrupting the running job.

        Unlike ``interrupt(task_id)``, this only posts the queue deletion and never hits ``/interrupt``,
        so cancelling a queued job leaves whatever is currently generating untouched.
        """
        queue_removal_body = QueueDeletionRequest(delete=[task_id])
        self.post(ComfyEndpoints.QUEUE, body=queue_removal_body.model_dump())

    def submit_txt2img(self, diffusion_params: DiffusionParams) -> 'ComfyGenerationHandle':
        """Queue a txt2img job and return a backend-agnostic handle to poll / wait / cancel it."""
        from intrapaint_api.api.comfyui.comfyui_generation_handle import ComfyGenerationHandle
        return ComfyGenerationHandle.from_queue_response(self, self.txt2img(diffusion_params))

    def submit_img2img(self, diffusion_params: DiffusionParams) -> 'ComfyGenerationHandle':
        """Queue an img2img job and return a backend-agnostic handle to poll / wait / cancel it."""
        from intrapaint_api.api.comfyui.comfyui_generation_handle import ComfyGenerationHandle
        return ComfyGenerationHandle.from_queue_response(self, self.img2img(diffusion_params))

    def submit_inpaint(self, diffusion_params: DiffusionParams) -> 'ComfyGenerationHandle':
        """Queue an inpaint job and return a backend-agnostic handle to poll / wait / cancel it."""
        from intrapaint_api.api.comfyui.comfyui_generation_handle import ComfyGenerationHandle
        return ComfyGenerationHandle.from_queue_response(self, self.inpaint(diffusion_params))

    @contextmanager
    def open_websocket(self) -> Generator[websocket.WebSocket, None, None]:
        """Yields an open ComfyUI websocket that automatically closes when the context exits."""
        ws = websocket.WebSocket()
        base_url = self.server_url  # http://localhost:8188
        assert '://' in base_url
        ws_url = f'ws://{base_url[base_url.index("://") + 3:]}/ws?clientId={self._client_id}'  # _client_id is uuid
        ws.connect(ws_url)
        try:
            yield ws
        finally:
            ws.close()

    @staticmethod
    def parse_percentage_from_websocket_message(websocket_text: str) -> Optional[float]:
        """Attempts to parse a percentage from a ComfyUI websocket message."""
        status: Optional[dict[str, Any]] = None
        if isinstance(websocket_text, str):
            try:
                status = json.loads(websocket_text)
            except json.decoder.JSONDecodeError:
                return None
        if status is not None and 'type' in status and 'data' in status:
            if status['type'] == 'progress':
                data = status['data']
                if 'value' in data and 'max' in data:
                    return round(data['value'] / data['max'] * 100, ndigits=4)
        return None

    # Misc. utility:
    def free_memory(self) -> None:
        """Clear cached data to free GPU memory."""
        body = FreeMemoryRequest(unload_models=True, free_memory=True)
        self.post(ComfyEndpoints.FREE, body, timeout=DEFAULT_TIMEOUT)
