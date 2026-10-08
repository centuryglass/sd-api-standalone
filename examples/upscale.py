"""Upscale an image to a scale factor or a target size with a server-side upscaling model.

Usage:
    python examples/upscale.py INPUT_IMAGE OUTPUT_IMAGE (--scale 2.0 | --size WIDTHxHEIGHT) [--upscaler NAME] \\
        [--backend {auto,comfyui,webui}] [--url SERVER_URL]
    python examples/upscale.py --list-upscalers [--url SERVER_URL]

Without `--upscaler` the server's first upscaling model is used.
"""
import argparse
from typing import Optional

from _common import add_connection_args, cli_main, connect, load_image, run_job
from sd_backend_client import DiffusionUpscalingParams


def target_size(source_size: tuple[int, int], scale: Optional[float], size: Optional[str]) -> tuple[int, int]:
    """The output size for a `--scale` factor or a `--size WIDTHxHEIGHT` string.

    Raises
    ------
    SystemExit
        If neither or both are given, or `size` is malformed.
    """
    if (scale is None) == (size is None):
        raise SystemExit('Give exactly one of --scale and --size.')
    if scale is not None:
        if scale <= 0:
            raise SystemExit('--scale must be positive.')
        return round(source_size[0] * scale), round(source_size[1] * scale)
    assert size is not None
    width, separator, height = size.lower().partition('x')
    if not separator or not width.isdigit() or not height.isdigit():
        raise SystemExit(f'--size must look like 1024x768, not {size!r}.')
    return int(width), int(height)


def main() -> None:
    """Parse arguments, run the upscale job and save its image."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_image', nargs='?', help='Path to the source image.')
    parser.add_argument('output_image', nargs='?', help='Path to save the upscaled image to.')
    parser.add_argument('--scale', type=float, help='Scale factor, e.g. 2.0.')
    parser.add_argument('--size', help='Target size as WIDTHxHEIGHT.')
    parser.add_argument('--upscaler', help='Upscaling model name (see --list-upscalers).')
    parser.add_argument('--list-upscalers', action='store_true', help='Print the server\'s upscaling models and exit.')
    add_connection_args(parser)
    args = parser.parse_args()

    backend = connect(args)
    if args.list_upscalers:
        for option in backend.list_upscalers():
            print(option.name)
        return
    if not args.input_image or not args.output_image:
        parser.error('input_image and output_image are required')

    source = load_image(args.input_image)
    width, height = target_size((source.width, source.height), args.scale, args.size)
    upscale_params = DiffusionUpscalingParams(upscaling_mode=args.upscaler) if args.upscaler else None
    try:
        handle = backend.submit_upscale(source, width, height, upscale_params)
    except ValueError as err:
        raise SystemExit(f'error: {err}') from err
    run_job(handle, 'upscaling').images[0].save(args.output_image)
    print(f'Saved {width}x{height} image to {args.output_image}')


if __name__ == '__main__':
    cli_main(main)
