"""Run the WebUI canny ControlNet preprocessor on an image and save the result.

Usage:
    python examples/canny_preprocessor_preview.py INPUT_IMAGE OUTPUT_IMAGE \\
        [--low THRESHOLD_A] [--high THRESHOLD_B] [--url WEBUI_URL]
"""
import argparse

from PIL import Image

from intrapaint_api.api.a1111_webservice import A1111Webservice
from intrapaint_api.api.shared_data.controlnet.controlnet_preprocessor import PreprocessorParams
from intrapaint_api.api.webui.controlnet_webui_constants import (FIRST_GENERIC_PARAMETER_KEY,
                                                                  SECOND_GENERIC_PARAMETER_KEY)

CANNY_MODULE_NAME = 'canny'
DEFAULT_URL = 'http://127.0.0.1:7860'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_image', help='Path to the source image.')
    parser.add_argument('output_image', help='Path to save the preprocessor output to.')
    parser.add_argument('--low', type=int, default=100, help='Canny low threshold (default: 100).')
    parser.add_argument('--high', type=int, default=200, help='Canny high threshold (default: 200).')
    parser.add_argument('--url', default=DEFAULT_URL, help=f'WebUI server URL (default: {DEFAULT_URL}).')
    args = parser.parse_args()

    service = A1111Webservice(args.url)
    canny = next((p for p in service.get_controlnet_preprocessors() if p.name == CANNY_MODULE_NAME), None)
    if canny is None:
        raise RuntimeError(f'"{CANNY_MODULE_NAME}" preprocessor not found on server at {args.url}')

    params = PreprocessorParams(typedef=canny, parameter_values={
        FIRST_GENERIC_PARAMETER_KEY: args.low,
        SECOND_GENERIC_PARAMETER_KEY: args.high,
    })

    source = Image.open(args.input_image)
    preview = service.controlnet_preprocessor_preview(source, None, params)
    preview.save(args.output_image)
    print(f'Saved canny preview to {args.output_image}')


if __name__ == '__main__':
    main()
