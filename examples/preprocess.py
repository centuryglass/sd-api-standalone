"""Run any ControlNet preprocessor on an image and save the control map it produces.

Works with ComfyUI (with comfyui_controlnet_aux) and with a WebUI that has ControlNet.

Usage:
    python examples/preprocess.py INPUT_IMAGE OUTPUT_IMAGE --preprocessor NAME [--param KEY=VALUE ...] [--mask MASK] \\
        [--backend {auto,comfyui,webui}] [--url SERVER_URL]
    python examples/preprocess.py --list [--url SERVER_URL]
    python examples/preprocess.py --preprocessor NAME --show-params [--url SERVER_URL]

For example, `--preprocessor canny --param threshold_a=50 --param threshold_b=150`. `--param` keys are the ones
`--show-params` lists, and a preprocessor's unset parameters keep the server's defaults.
"""
import argparse
from typing import Any

from _common import add_connection_args, cli_main, connect, load_image, run_job
from sd_backend_client import Backend, ControlNetPreprocessor, ParameterDef, PreprocessorParams


def find_preprocessor(backend: Backend, name: str) -> ControlNetPreprocessor:
    """The server's preprocessor called `name` (case-insensitive), or exit with a message listing close matches."""
    preprocessors = backend.get_controlnet_preprocessors()
    for preprocessor in preprocessors:
        if preprocessor.name.lower() == name.lower():
            return preprocessor
    similar = [p.name for p in preprocessors if name.lower() in p.name.lower()]
    hint = f' Similar: {", ".join(similar)}.' if similar else ' Use --list to see them.'
    raise SystemExit(f'The server has no preprocessor named {name!r}.{hint}')


def coerce_value(parameter: ParameterDef, text: str) -> Any:
    """Convert `text` to the type of `parameter`'s default value."""
    default = parameter.default_value
    try:
        if isinstance(default, bool):
            if text.lower() not in ('true', 'false', '1', '0'):
                raise ValueError('expected true or false')
            return text.lower() in ('true', '1')
        if isinstance(default, int):
            return int(text)
        if isinstance(default, float):
            return float(text)
    except ValueError as err:
        raise SystemExit(f'Parameter {parameter.key} takes a {type(default).__name__}: {err}') from err
    return text


def parse_parameter_values(preprocessor: ControlNetPreprocessor, assignments: list[str]) -> dict[str, Any]:
    """Turn `KEY=VALUE` strings into typed values for `preprocessor`'s parameters."""
    by_key = {parameter.key: parameter for parameter in preprocessor.parameters}
    values: dict[str, Any] = {}
    for assignment in assignments:
        key, separator, text = assignment.partition('=')
        if not separator or key not in by_key:
            raise SystemExit(f'Bad --param {assignment!r}. {preprocessor.name} takes: ' +
                             (', '.join(by_key) or 'no parameters') + ' (as KEY=VALUE).')
        values[key] = coerce_value(by_key[key], text)
    return values


def describe(preprocessor: ControlNetPreprocessor) -> str:
    """One line per parameter of `preprocessor`, with its default and range."""
    lines = [f'{preprocessor.name} ({preprocessor.category_name or "uncategorized"})']
    for parameter in preprocessor.parameters:
        limits = ''
        if parameter.min_val is not None or parameter.max_val is not None:
            limits = f' [{parameter.min_val} to {parameter.max_val}]'
        lines.append(f'  {parameter.key}: default {parameter.default_value!r}{limits}')
    return '\n'.join(lines)


def main() -> None:
    """Parse arguments, run the preview job and save its image."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_image', nargs='?', help='Path to the source image.')
    parser.add_argument('output_image', nargs='?', help='Path to save the control map to.')
    parser.add_argument('--preprocessor', help='Preprocessor name, as --list prints it.')
    parser.add_argument('--param', action='append', default=[], metavar='KEY=VALUE',
                        help='Preprocessor parameter, repeatable.')
    parser.add_argument('--mask', help='Mask image for preprocessors that take one.')
    parser.add_argument('--list', action='store_true', help='Print the server\'s preprocessors and exit.')
    parser.add_argument('--show-params', action='store_true',
                        help='Print the chosen preprocessor\'s parameters and exit.')
    add_connection_args(parser)
    args = parser.parse_args()

    backend = connect(args)
    if args.list:
        for preprocessor in backend.get_controlnet_preprocessors():
            print(preprocessor.name)
        return
    if not args.preprocessor:
        parser.error('--preprocessor is required (use --list to see the choices)')
    preprocessor = find_preprocessor(backend, args.preprocessor)
    if args.show_params:
        print(describe(preprocessor))
        return
    if not args.input_image or not args.output_image:
        parser.error('input_image and output_image are required')

    params = PreprocessorParams(typedef=preprocessor,
                                parameter_values=parse_parameter_values(preprocessor, args.param))
    source = load_image(args.input_image)
    mask = load_image(args.mask) if args.mask else None
    result = run_job(backend.submit_preprocessor_preview(source, params, mask), 'preprocessing')
    result.images[0].save(args.output_image)
    print(f'Saved {preprocessor.name} output to {args.output_image}')


if __name__ == '__main__':
    cli_main(main)
