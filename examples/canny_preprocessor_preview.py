"""Run the canny ControlNet preprocessor on an image and save the edge map it produces.

Works with ComfyUI (with comfyui_controlnet_aux) and with a WebUI that has ControlNet.

Usage:
    python examples/canny_preprocessor_preview.py INPUT_IMAGE OUTPUT_IMAGE \\
        [--low LOW_THRESHOLD] [--high HIGH_THRESHOLD] [--url SERVER_URL]
"""
import argparse

from PIL import Image

from sd_backend_client import ControlNetPreprocessor, PreprocessorParams, connect_to_backend

DEFAULT_URL = 'http://127.0.0.1:7860'
CANNY_CONTROL_TYPE = 'Canny'
# The canny thresholds' parameter keys: WebUI's generic slider keys, then comfyui_controlnet_aux's input names.
LOW_THRESHOLD_KEYS = ('threshold_a', 'low_threshold')
HIGH_THRESHOLD_KEYS = ('threshold_b', 'high_threshold')


def find_canny_preprocessor(preprocessors: list[ControlNetPreprocessor],
                            categories: dict) -> ControlNetPreprocessor:
    """Return the first preprocessor the server lists in its canny control type."""
    canny_names = set(categories.get(CANNY_CONTROL_TYPE, {}).get('module_list', []))
    for preprocessor in preprocessors:
        if preprocessor.name in canny_names and preprocessor.name.lower() != 'none':
            return preprocessor
    raise SystemExit('The server has no canny preprocessor. Is ControlNet installed?')


def threshold_values(preprocessor: ControlNetPreprocessor, low: int, high: int) -> dict[str, int]:
    """Map the low and high thresholds onto whichever parameter keys this preprocessor uses."""
    keys = {parameter.key for parameter in preprocessor.parameters}
    values = {}
    for candidates, value in ((LOW_THRESHOLD_KEYS, low), (HIGH_THRESHOLD_KEYS, high)):
        key = next((key for key in candidates if key in keys), None)
        if key is not None:
            values[key] = value
    return values


def main() -> None:
    """Parse arguments, run the preview job and save its image."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_image', help='Path to the source image.')
    parser.add_argument('output_image', help='Path to save the edge map to.')
    parser.add_argument('--low', type=int, default=100, help='Canny low threshold (default: 100).')
    parser.add_argument('--high', type=int, default=200, help='Canny high threshold (default: 200).')
    parser.add_argument('--url', default=DEFAULT_URL, help=f'ComfyUI or WebUI server URL (default: {DEFAULT_URL}).')
    args = parser.parse_args()

    backend = connect_to_backend(args.url)
    canny = find_canny_preprocessor(backend.get_controlnet_preprocessors(), backend.get_controlnet_type_categories())
    params = PreprocessorParams(typedef=canny, parameter_values=threshold_values(canny, args.low, args.high))

    with Image.open(args.input_image) as source:
        result = backend.submit_preprocessor_preview(source, params).wait()
    result.images[0].save(args.output_image)
    print(f'Saved {canny.name} output to {args.output_image}')


if __name__ == '__main__':
    main()
