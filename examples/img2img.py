"""Transform an image with a prompt (img2img), or change only a masked region of it (inpainting).

Without `--mask` the whole image is redrawn at the given strength. With `--mask`, white (or opaque) mask pixels mark
the region to change and the rest is kept. Output size defaults to the source image's size, rounded down to a
multiple of 8. See `examples/txt2img.py` for the shared flags and output format.

Usage:
    python examples/img2img.py "a snowy field" --image INPUT [--mask MASK] [--strength 0.75] [txt2img flags]
"""
import argparse

from _common import add_connection_args, add_generation_args, cli_main, connect, load_image, params_from_args, \
    round_size, run_job, save_results


def add_image_args(parser: argparse.ArgumentParser, *, mask_required: bool) -> None:
    """Add `--image`, `--mask` and `--strength`."""
    parser.add_argument('--image', required=True, help='Path to the source image.')
    parser.add_argument('--mask', required=mask_required,
                        help='Path to a mask image: white or opaque pixels are changed' +
                             ('.' if mask_required else '. Omit to redraw the whole image.'))
    parser.add_argument('--strength', type=float, default=None,
                        help='How far the result may stray from the source, 0.0-1.0 (default: 0.75).')


def run(args: argparse.Namespace) -> None:
    """Submit an img2img job, or an inpainting job when `args.mask` is set, and save the images."""
    backend = connect(args)
    source = load_image(args.image)
    extra = {'init_images': [source], 'denoising_strength': args.strength}
    if args.width is None:
        extra['width'] = round_size(source.width)
    if args.height is None:
        extra['height'] = round_size(source.height)
    if args.mask:
        extra['mask'] = load_image(args.mask)
    params = params_from_args(args, **extra)
    handle = backend.submit_inpaint(params) if args.mask else backend.submit_img2img(params)
    result = run_job(handle)
    for path in save_results(result, params, backend, args.output_dir):
        print(path)


def main() -> None:
    """Parse arguments and run the job."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_generation_args(parser)
    add_image_args(parser, mask_required=False)
    add_connection_args(parser)
    run(parser.parse_args())


if __name__ == '__main__':
    cli_main(main)
