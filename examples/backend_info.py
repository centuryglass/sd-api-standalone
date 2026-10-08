"""List what a ComfyUI or WebUI server offers, as a connectivity and authentication check.

Prints checkpoints, VAEs, LoRAs, hypernetworks, samplers, schedulers, upscalers, ControlNet models and preprocessors,
and the server's optional features. Exits with status 1 and an error message if the server cannot be reached.

Usage:
    python examples/backend_info.py [--json] [--backend {auto,comfyui,webui}] [--url SERVER_URL]
"""
import argparse
import json
import sys
from typing import Any

from _common import add_connection_args, cli_main, connect
from sd_backend_client import Backend, BackendOption


def option_names(options: list[BackendOption]) -> list[str]:
    """The `name` of each option."""
    return [option.name for option in options]


def collect(backend: Backend) -> dict[str, Any]:
    """Query every discovery method on `backend` and return the results as plain JSON-serializable data."""
    return {
        'backend': type(backend).__name__,
        'capabilities': backend.get_capabilities().model_dump(),
        'checkpoints': option_names(backend.list_checkpoints()),
        'vaes': option_names(backend.list_vaes()),
        'loras': option_names(backend.list_loras()),
        'hypernetworks': option_names(backend.list_hypernetworks()),
        'samplers': option_names(backend.list_samplers()),
        'schedulers': option_names(backend.list_schedulers()),
        'upscalers': option_names(backend.list_upscalers()),
        'controlnet_models': [str(model) for model in backend.list_controlnet_models()],
        'controlnet_preprocessors': [preprocessor.name for preprocessor in backend.get_controlnet_preprocessors()],
    }


def print_report(info: dict[str, Any]) -> None:
    """Print `collect`'s result as readable text."""
    print(f"Connected to a {info['backend']} server.")
    print('Features: ' + ', '.join(f'{name}={"yes" if value else "no"}'
                                   for name, value in info['capabilities'].items()))
    for key, values in info.items():
        if isinstance(values, list):
            print(f"\n{key.replace('_', ' ')} ({len(values)}):")
            for value in values:
                print(f'  {value}')


def main() -> None:
    """Parse arguments, query the server and print what it offers."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--json', action='store_true', help='Print one JSON object instead of text.')
    add_connection_args(parser)
    args = parser.parse_args()

    info = collect(connect(args))
    if args.json:
        json.dump(info, sys.stdout, indent=2)
        print()
    else:
        print_report(info)


if __name__ == '__main__':
    cli_main(main)
