"""Generate images from a text prompt on a ComfyUI or WebUI server.

Each PNG is saved with its settings in the A1111-style `parameters` text chunk, which other Stable Diffusion tools
(and `examples/rerun.py`) can read. Progress goes to stderr and the saved paths to stdout.

Usage:
    python examples/txt2img.py "a lighthouse at dusk" [-n NEGATIVE] [--width W] [--height H] [--steps N] [--cfg X] \\
        [--seed N] [--sampler NAME] [--scheduler NAME] [--checkpoint NAME] [--batch-size N] [-o DIR] \\
        [--backend {auto,comfyui,webui}] [--url SERVER_URL]
"""
import argparse

from _common import add_connection_args, add_generation_args, cli_main, connect, params_from_args, run_job, \
    save_results


def main() -> None:
    """Parse arguments, run the job and save its images."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_generation_args(parser)
    add_connection_args(parser)
    args = parser.parse_args()

    backend = connect(args)
    params = params_from_args(args)
    result = run_job(backend.submit_txt2img(params))
    for path in save_results(result, params, backend, args.output_dir):
        print(path)


if __name__ == '__main__':
    cli_main(main)
