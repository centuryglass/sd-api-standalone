"""Sweep one or two generation parameters and stitch the results into a labeled contact sheet.

Each axis is `NAME=v1,v2,...` where NAME is one of cfg_scale (or cfg), steps, seed, sampler, scheduler,
denoising_strength, width, height, checkpoint, or `prompt_sr`. `prompt_sr=old,new1,new2` replaces `old` in the prompt
with each listed value in turn (the first value is the text to replace, and also the first cell). All jobs are
queued before any is awaited, so a ComfyUI server works through the whole grid without the script idling between cells.

Usage:
    python examples/xy_grid.py "a portrait of a fox" --x cfg=4,7,10 --y steps=10,20,30 [--grid-output grid.png] \\
        [txt2img flags]
    python examples/xy_grid.py "a red car" --x prompt_sr=red,blue,green --seed 5
"""
import argparse
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from _common import add_connection_args, add_generation_args, build_params, cli_main, connect, params_from_args, \
    save_png
from sd_backend_client import GenerationHandle, GenerationProgress

AXIS_ALIASES = {'cfg': 'cfg_scale', 'checkpoint': 'sd_model_name', 'model': 'sd_model_name'}
AXIS_FIELDS = {'cfg_scale', 'steps', 'seed', 'sampler', 'scheduler', 'denoising_strength', 'width', 'height',
               'sd_model_name', 'prompt_sr'}
LABEL_HEIGHT = 24
LABEL_WIDTH = 120
MAX_CELLS = 64


def parse_axis(text: str) -> tuple[str, list[str]]:
    """Split `NAME=v1,v2,...` into a canonical field name and its value strings."""
    name, separator, values = text.partition('=')
    name = AXIS_ALIASES.get(name.strip(), name.strip())
    items = [value.strip() for value in values.split(',')] if separator else []
    if name not in AXIS_FIELDS or not items or '' in items:
        raise SystemExit(f'Bad axis {text!r}. Use NAME=v1,v2,... with NAME one of: ' +
                         ', '.join(sorted(AXIS_FIELDS | set(AXIS_ALIASES))))
    return name, items


def cell_overrides(axis: tuple[str, list[str]], index: int, prompt: str) -> dict[str, Any]:
    """The `DiffusionParams` fields that set cell `index` of `axis`."""
    name, values = axis
    if name == 'prompt_sr':
        return {'prompt': prompt.replace(values[0], values[index])}
    return {name: values[index]}


def cell_label(axis: tuple[str, list[str]], index: int) -> str:
    """The text printed above or beside cell `index` of `axis`."""
    name, values = axis
    return values[index] if name == 'prompt_sr' else f'{name}={values[index]}'


def make_sheet(cells: list[list[Image.Image]], column_labels: list[str], row_labels: list[str]) -> Image.Image:
    """Tile `cells[row][column]` into one image with a label above each column and beside each row."""
    cell_width = max(image.width for row in cells for image in row)
    cell_height = max(image.height for row in cells for image in row)
    left = LABEL_WIDTH if any(row_labels) else 0
    sheet = Image.new('RGB', (left + cell_width * len(cells[0]), LABEL_HEIGHT + cell_height * len(cells)), 'white')
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(column_labels):
        draw.text((left + column * cell_width + 4, 6), label, fill='black')
    for row, images in enumerate(cells):
        top = LABEL_HEIGHT + row * cell_height
        if row_labels[row]:
            draw.text((4, top + 4), row_labels[row], fill='black')
        for column, image in enumerate(images):
            sheet.paste(image.convert('RGB'), (left + column * cell_width, top))
    return sheet


def wait_all(handles: list[GenerationHandle]) -> list[Image.Image]:
    """Wait for each handle in turn and return each one's first image."""
    images = []
    for number, handle in enumerate(handles, start=1):
        def report(progress: GenerationProgress, number: int = number) -> None:
            print(f'cell {number}/{len(handles)}: {progress.status.value}', file=sys.stderr, end='\r', flush=True)
        images.append(handle.wait(on_progress=report).images[0])
    print(file=sys.stderr)
    return images


def main() -> None:
    """Parse arguments, queue the grid's jobs, and save the contact sheet."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_generation_args(parser)
    parser.add_argument('--x', required=True, dest='x_axis', metavar='NAME=v1,v2', help='Column axis.')
    parser.add_argument('--y', dest='y_axis', metavar='NAME=v1,v2', help='Row axis (optional).')
    parser.add_argument('--grid-output', type=Path, default=Path('grid.png'), help='Contact sheet path.')
    add_connection_args(parser)
    args = parser.parse_args()

    x_axis = parse_axis(args.x_axis)
    y_axis = parse_axis(args.y_axis) if args.y_axis else ('', [''])
    if len(x_axis[1]) * len(y_axis[1]) > MAX_CELLS:
        raise SystemExit(f'The grid has more than {MAX_CELLS} cells.')
    base = params_from_args(args)
    if args.seed is None:
        # One shared seed makes the cells differ only in the swept parameters.
        base = base.model_copy(update={'seed': random.randrange(2 ** 32)})

    backend = connect(args)
    handles = []
    for row in range(len(y_axis[1])):
        for column in range(len(x_axis[1])):
            fields = base.model_dump(exclude={'controlnet_units'})
            fields.update(cell_overrides(x_axis, column, fields['prompt']))
            if y_axis[0]:
                fields.update(cell_overrides(y_axis, row, fields['prompt']))
            handles.append(backend.submit_txt2img(build_params(fields)))
    images = wait_all(handles)

    columns = len(x_axis[1])
    cells = [images[row * columns:(row + 1) * columns] for row in range(len(y_axis[1]))]
    sheet = make_sheet(cells, [cell_label(x_axis, i) for i in range(columns)],
                       [cell_label(y_axis, i) if y_axis[0] else '' for i in range(len(y_axis[1]))])
    save_png(sheet, args.grid_output)
    print(args.grid_output)


if __name__ == '__main__':
    cli_main(main)
