"""Helpers shared by the scripts in `examples/`: connecting, common flags, progress output and PNG metadata.

Scripts import this module by name, so run them as `python examples/<script>.py` (Python puts the script's own
directory on `sys.path`). Only the public `sd_backend_client` API is used.
"""
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from sd_backend_client import A1111Webservice, Backend, ComfyUiWebservice, DiffusionParams, GenerationHandle, \
    GenerationProgress, GenerationResult, SDBackendError, connect_to_backend

DEFAULT_WEBUI_URL = 'http://127.0.0.1:7860'
DEFAULT_COMFYUI_URL = 'http://127.0.0.1:8188'
BACKEND_CHOICES = ('auto', 'comfyui', 'webui')
PARAMETERS_KEY = 'parameters'
"""PNG text chunk keyword that A1111-style tools read generation settings from."""
SIZE_MULTIPLE = 8


def add_connection_args(parser: argparse.ArgumentParser) -> None:
    """Add `--url`, `--backend`, `--username` and `--password`."""
    parser.add_argument('--url', default=None,
                        help='Server URL. Defaults to $SD_API_URL (or $COMFYUI_API_URL with --backend comfyui), '
                             f'then {DEFAULT_WEBUI_URL} ({DEFAULT_COMFYUI_URL} with --backend comfyui).')
    parser.add_argument('--backend', choices=BACKEND_CHOICES, default='auto',
                        help='Which kind of server to expect (default: auto-detect from the URL).')
    parser.add_argument('--username', default=os.environ.get('SD_UNAME'),
                        help='WebUI --api-auth user name (default: $SD_UNAME).')
    parser.add_argument('--password', default=os.environ.get('SD_PASS'),
                        help='WebUI --api-auth password (default: $SD_PASS).')


def add_generation_args(parser: argparse.ArgumentParser, *, prompt_required: bool = True) -> None:
    """Add the flags that map onto `DiffusionParams`, plus `--output-dir`."""
    parser.add_argument('prompt', nargs=None if prompt_required else '?', default='', help='Text prompt.')
    parser.add_argument('-n', '--negative-prompt', default=None, help='Negative prompt.')
    parser.add_argument('--width', type=int, default=None, help='Image width (default: 512, or the source image).')
    parser.add_argument('--height', type=int, default=None, help='Image height (default: 512, or the source image).')
    parser.add_argument('--steps', type=int, default=None, help='Sampling steps (default: 30).')
    parser.add_argument('--cfg', type=float, default=None, help='CFG scale (default: 7).')
    parser.add_argument('--seed', type=int, default=None, help='Seed (default: random).')
    parser.add_argument('--sampler', default=None, help='Sampler name, e.g. euler_ancestral or "DPM++ 2M".')
    parser.add_argument('--scheduler', default=None, help='Scheduler name, e.g. karras.')
    parser.add_argument('--checkpoint', default=None, help="Checkpoint name (default: the server's current one).")
    parser.add_argument('--batch-size', type=int, default=None, help='Images per job (default: 1).')
    parser.add_argument('-o', '--output-dir', type=Path, default=Path('outputs'),
                        help='Directory to save PNGs to (default: ./outputs).')


def params_from_args(args: argparse.Namespace, **extra: Any) -> DiffusionParams:
    """Build `DiffusionParams` from the flags `add_generation_args` defines. Unset flags keep the model defaults.

    `extra` fields are set too, and win over the flags.
    """
    fields: dict[str, Any] = {'prompt': args.prompt or ''}
    for flag, field in (('negative_prompt', 'negative_prompt'), ('width', 'width'), ('height', 'height'),
                        ('steps', 'steps'), ('cfg', 'cfg_scale'), ('seed', 'seed'), ('sampler', 'sampler'),
                        ('scheduler', 'scheduler'), ('checkpoint', 'sd_model_name'), ('batch_size', 'batch_size')):
        value = getattr(args, flag, None)
        if value is not None:
            fields[field] = value
    fields.update(extra)
    return build_params(fields)


def build_params(fields: dict[str, Any]) -> DiffusionParams:
    """`DiffusionParams(**fields)`, turning a validation failure into a one-line exit message."""
    try:
        return DiffusionParams(**fields)
    except ValueError as err:
        raise SystemExit(f'Invalid generation parameters: {err}') from err


def connect(args: argparse.Namespace) -> Backend:
    """Connect to the server named by `add_connection_args`' flags."""
    env_url = os.environ.get('COMFYUI_API_URL' if args.backend == 'comfyui' else 'SD_API_URL')
    url = args.url or env_url or (DEFAULT_COMFYUI_URL if args.backend == 'comfyui' else DEFAULT_WEBUI_URL)

    def credentials() -> Optional[tuple[str, str]]:
        if args.username and args.password:
            return args.username, args.password
        return None

    if args.backend == 'comfyui':
        return ComfyUiWebservice(url)
    if args.backend == 'webui':
        return A1111Webservice(url, credentials_provider=credentials)
    return connect_to_backend(url, credentials_provider=credentials)


def require_webui(backend: Backend, feature: str) -> A1111Webservice:
    """Return `backend` as a WebUI client, or exit with a message saying `feature` needs one."""
    if not isinstance(backend, A1111Webservice):
        raise SystemExit(f'{feature} needs a WebUI (A1111 or Forge) server.')
    return backend


class ProgressReporter:
    """A `wait(on_progress=...)` callback that writes progress to stderr.

    On a terminal it rewrites one line; otherwise it writes a line each time progress passes another 10%.
    """

    def __init__(self, label: str = 'generating') -> None:
        self._label = label
        self._last_bucket = -1
        self._tty = sys.stderr.isatty()

    def __call__(self, progress: GenerationProgress) -> None:
        percent = int((progress.progress or 0.0) * 100)
        text = f'{self._label}: {progress.status.value} {percent}%'
        if self._tty:
            end = '\n' if progress.done else ''
            print(f'\r{text}\033[K', end=end, file=sys.stderr, flush=True)
        elif percent // 10 != self._last_bucket or progress.done:
            self._last_bucket = percent // 10
            print(text, file=sys.stderr, flush=True)


def run_job(handle: GenerationHandle, label: str = 'generating') -> GenerationResult:
    """Wait for `handle`, reporting progress on stderr, and cancel the job on Ctrl-C."""
    try:
        return handle.wait(on_progress=ProgressReporter(label))
    except KeyboardInterrupt:
        handle.cancel()
        raise SystemExit('Cancelled.') from None


def load_image(path: str | Path) -> Image.Image:
    """Read an image file fully, so the file handle is closed."""
    try:
        with Image.open(path) as image:
            image.load()
            return image.copy()
    except OSError as err:
        raise SystemExit(f'Cannot read image {path}: {err}') from err


def round_size(value: int) -> int:
    """Round `value` down to a multiple of `SIZE_MULTIPLE`, but not below one multiple."""
    return max(SIZE_MULTIPLE, value - value % SIZE_MULTIPLE)


def option_labels(backend: Backend) -> dict[str, str]:
    """Map each shared sampler name to the WebUI-style label other tools expect, where the server lists one."""
    return {option.name: option.display_name or option.name for option in backend.list_samplers()}


def format_parameters(params: DiffusionParams, seed: int,
                      sampler_labels: Optional[dict[str, str]] = None) -> str:
    """Format generation settings as the A1111-style `parameters` text.

    Parameters
    ----------
    params
        The settings the job ran with.
    seed
        The seed of the saved image, which may differ from `params.seed` (a random seed or a later batch member).
    sampler_labels
        Shared sampler name to display label, from `option_labels`.
    """
    lines = [params.prompt]
    if params.negative_prompt:
        lines.append(f'Negative prompt: {params.negative_prompt}')
    sampler = (sampler_labels or {}).get(params.sampler, params.sampler)
    settings = [('Steps', params.steps), ('Sampler', sampler), ('Schedule type', params.scheduler),
                ('CFG scale', params.cfg_scale), ('Seed', seed), ('Size', f'{params.width}x{params.height}'),
                ('Batch size', params.batch_size)]
    if params.sd_model_name:
        settings.append(('Model', params.sd_model_name))
    if params.init_images:
        settings.append(('Denoising strength', params.denoising_strength))
    lines.append(', '.join(f'{key}: {_quote(value)}' for key, value in settings if value is not None))
    return '\n'.join(lines)


def _quote(value: Any) -> str:
    """Quote a settings value that holds a comma, as A1111 does."""
    text = str(value)
    return '"' + text.replace('"', '\\"') + '"' if ',' in text else text


SETTING_PATTERN = re.compile(r'\s*([\w ]+): ("(?:[^"\\]|\\.)*"|[^,]*)(?:,|$)')

# Settings keys the parser maps back to `DiffusionParams` fields, with the type of each value.
_PARSED_FIELDS: dict[str, tuple[str, Callable[[str], Any]]] = {
    'Steps': ('steps', int), 'Sampler': ('sampler', str), 'Schedule type': ('scheduler', str),
    'CFG scale': ('cfg_scale', float), 'Seed': ('seed', int), 'Batch size': ('batch_size', int),
    'Model': ('sd_model_name', str), 'Denoising strength': ('denoising_strength', float),
}


def parse_parameters(text: str) -> dict[str, Any]:
    """Parse A1111-style `parameters` text into `DiffusionParams` fields. Unknown settings are ignored.

    Raises
    ------
    ValueError
        If the text has no settings line.
    """
    lines = text.strip().split('\n')
    if not lines or not lines[-1].startswith('Steps:'):
        raise ValueError('no "Steps: ..." settings line found')
    fields: dict[str, Any] = {}
    for key, raw in SETTING_PATTERN.findall(lines[-1]):
        value = raw.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1].replace('\\"', '"')
        if key == 'Size':
            width, _, height = value.partition('x')
            fields['width'], fields['height'] = int(width), int(height)
        elif key in _PARSED_FIELDS:
            field, convert = _PARSED_FIELDS[key]
            fields[field] = convert(value)
    prompt_lines = lines[:-1]
    negative_start = next((i for i, line in enumerate(prompt_lines) if line.startswith('Negative prompt:')), None)
    if negative_start is not None:
        negative = prompt_lines[negative_start:]
        negative[0] = negative[0][len('Negative prompt:'):].lstrip()
        fields['negative_prompt'] = '\n'.join(negative)
        prompt_lines = prompt_lines[:negative_start]
    fields['prompt'] = '\n'.join(prompt_lines)
    return fields


def save_png(image: Image.Image, path: Path, parameters: Optional[str] = None) -> None:
    """Save `image` as a PNG, with `parameters` in the A1111 text chunk when given."""
    info = None
    if parameters is not None:
        info = PngInfo()
        info.add_text(PARAMETERS_KEY, parameters)
    image.save(path, pnginfo=info)


def save_results(result: GenerationResult, params: DiffusionParams, backend: Backend, output_dir: Path,
                 prefix: str = '') -> list[Path]:
    """Save each result image to `output_dir` as `<index>-<seed>.png` with its generation settings embedded.

    The index continues from the PNGs already in the directory, so runs never overwrite each other.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = option_labels(backend)
    index = len(list(output_dir.glob('*.png')))
    paths = []
    for position, image in enumerate(result.images):
        seed = result.seeds[position] if position < len(result.seeds) else (result.seed or params.seed)
        index += 1
        path = output_dir / f'{prefix}{index:05d}-{seed}.png'
        save_png(image, path, format_parameters(params, seed, sampler_labels=labels))
        paths.append(path)
    return paths


def cli_main(main: Callable[[], None]) -> None:
    """Run `main`, reporting a backend failure as a one-line error and a nonzero exit status."""
    try:
        main()
    except SDBackendError as err:
        raise SystemExit(f'error: {err}') from err
