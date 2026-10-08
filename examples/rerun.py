"""Regenerate an image from the settings saved in an earlier output's `parameters` PNG chunk.

Works on PNGs from `examples/txt2img.py` and on A1111-style PNGs from other tools. The run is a text-to-image
job, so a source image or mask used by the original is not reused. Flags override the saved settings.

Usage:
    python examples/rerun.py OUTPUT.png [--seed N | --seed +N] [--steps N] [--cfg X] [--prompt TEXT] \\
        [-n NEGATIVE] [--width W] [--height H] [--sampler NAME] [--scheduler NAME] [--checkpoint NAME] \\
        [--batch-size N] [-o DIR] [--backend {auto,comfyui,webui}] [--url SERVER_URL]

`--seed +N` adds N to the saved seed, and a plain `--seed N` sets it (use `--seed random` for a new random seed).
"""
import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from _common import PARAMETERS_KEY, add_connection_args, cli_main, connect, build_params, parse_parameters, run_job, \
    save_results


def read_parameters(path: Path) -> dict[str, Any]:
    """The generation settings embedded in the PNG at `path`, as `DiffusionParams` fields."""
    try:
        with Image.open(path) as image:
            text = image.info.get(PARAMETERS_KEY)
    except OSError as err:
        raise SystemExit(f'Cannot read {path}: {err}') from err
    if not isinstance(text, str):
        raise SystemExit(f'{path} has no "{PARAMETERS_KEY}" text chunk to rerun.')
    try:
        return parse_parameters(text)
    except ValueError as err:
        raise SystemExit(f'Cannot parse the settings in {path}: {err}') from err


def resolve_seed(saved: int, requested: str) -> int:
    """The seed for `--seed` text: `+N` offsets the `saved` seed, `random` is -1, anything else is the seed."""
    try:
        if requested == 'random':
            return -1
        if requested.startswith('+'):
            return saved + int(requested[1:])
        return int(requested)
    except ValueError as err:
        raise SystemExit(f'--seed must be a number, +N or random, not {requested!r}.') from err


def main() -> None:
    """Parse arguments, apply the overrides to the saved settings and run the job."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('image', type=Path, help='A PNG with a "parameters" text chunk.')
    parser.add_argument('--prompt')
    parser.add_argument('-n', '--negative-prompt')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--cfg', type=float)
    parser.add_argument('--seed', help='A seed, +N to offset the saved seed, or "random".')
    parser.add_argument('--sampler')
    parser.add_argument('--scheduler')
    parser.add_argument('--checkpoint')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('-o', '--output-dir', type=Path, default=Path('outputs'),
                        help='Directory to save PNGs to (default: ./outputs).')
    add_connection_args(parser)
    args = parser.parse_args()

    fields = read_parameters(args.image)
    for flag, field in (('prompt', 'prompt'), ('negative_prompt', 'negative_prompt'), ('width', 'width'),
                        ('height', 'height'), ('steps', 'steps'), ('cfg', 'cfg_scale'), ('sampler', 'sampler'),
                        ('scheduler', 'scheduler'), ('checkpoint', 'sd_model_name'),
                        ('batch_size', 'batch_size')):
        if getattr(args, flag) is not None:
            fields[field] = getattr(args, flag)
    if args.seed is not None:
        fields['seed'] = resolve_seed(fields.get('seed', -1), args.seed)
    params = build_params(fields)

    backend = connect(args)
    result = run_job(backend.submit_txt2img(params))
    for path in save_results(result, params, backend, args.output_dir):
        print(path)


if __name__ == '__main__':
    cli_main(main)
