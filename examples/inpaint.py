"""Change only the masked region of an image, guided by a prompt.

White (or opaque) mask pixels mark the region to change. This is `examples/img2img.py` with `--mask` required; see
it for the details and `examples/txt2img.py` for the shared flags.

Usage:
    python examples/inpaint.py "a red door" --image INPUT --mask MASK [--strength 0.75] [txt2img flags]
"""
import argparse

from _common import add_connection_args, add_generation_args, cli_main
from img2img import add_image_args, run


def main() -> None:
    """Parse arguments and run the inpainting job."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_generation_args(parser)
    add_image_args(parser, mask_required=True)
    add_connection_args(parser)
    run(parser.parse_args())


if __name__ == '__main__':
    cli_main(main)
