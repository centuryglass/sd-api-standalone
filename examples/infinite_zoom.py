"""Make an infinite-zoom animation by repeatedly shrinking the frame and inpainting the new border.

Starts from `--image`, or from a text-to-image generation of the prompt. Each round pastes the latest frame at
`--shrink` times its size in the middle of a new frame, inpaints the surrounding border with the prompt, and the
animation zooms out from each frame to the next. The result is written as an animated GIF or WebP (by file extension)
with Pillow alone.

Usage:
    python examples/infinite_zoom.py "a forest path, fantasy art" [--image START] [--rounds 6] [--shrink 0.6] \\
        [--frames-per-round 12] [--frame-ms 60] [--animation zoom.gif] [txt2img flags]
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from _common import add_connection_args, add_generation_args, cli_main, connect, load_image, params_from_args, \
    round_size, run_job, save_png
from sd_backend_client import Backend, DiffusionParams

MASK_OVERLAP = 16
"""Pixels of the shrunk frame's edge that the inpaint may repaint, to blend the seam."""
BACKGROUND_BLUR = 24


def zoom_out_canvas(frame: Image.Image, shrink: float) -> tuple[Image.Image, Image.Image]:
    """Return an inpainting init image and mask for the frame that surrounds `frame`.

    The init image holds `frame` scaled by `shrink` over a blurred, enlarged copy of itself. The mask is white
    (repaint) everywhere except the scaled frame, inset by `MASK_OVERLAP` (less for a small frame).
    """
    width, height = frame.size
    inner_size = (round(width * shrink), round(height * shrink))
    left, top = (width - inner_size[0]) // 2, (height - inner_size[1]) // 2
    canvas = frame.convert('RGB').filter(ImageFilter.GaussianBlur(BACKGROUND_BLUR))
    canvas.paste(frame.convert('RGB').resize(inner_size, Image.Resampling.LANCZOS), (left, top))
    overlap = min(MASK_OVERLAP, min(inner_size) // 4)
    mask = Image.new('L', frame.size, 255)
    ImageDraw.Draw(mask).rectangle((left + overlap, top + overlap, left + inner_size[0] - overlap - 1,
                                    top + inner_size[1] - overlap - 1), fill=0)
    return canvas, mask


def zoom_frames(outer: Image.Image, shrink: float, count: int) -> list[Image.Image]:
    """`count` views of `outer` that zoom out from its central `shrink`-sized region (the previous frame) to all of it."""
    width, height = outer.size
    views = []
    for step in range(count):
        side = shrink ** (1 - step / count)
        box = ((width - width * side) / 2, (height - height * side) / 2,
               (width + width * side) / 2, (height + height * side) / 2)
        views.append(outer.resize(outer.size, Image.Resampling.LANCZOS, box=box))
    return views


def build_animation(frames: list[Image.Image], shrink: float, frames_per_round: int) -> list[Image.Image]:
    """The zoom-out sequence through `frames`, which are ordered from the first (innermost) to the last."""
    sequence: list[Image.Image] = []
    for outer in frames[1:]:
        sequence += zoom_frames(outer, shrink, frames_per_round)
    sequence.append(frames[-1])
    return sequence


def first_frame(backend: Backend, args: argparse.Namespace, params: DiffusionParams) -> Image.Image:
    """The start image: `--image` resized to the generation size, or a text-to-image result."""
    if args.image:
        return load_image(args.image).convert('RGB').resize((params.width, params.height), Image.Resampling.LANCZOS)
    return run_job(backend.submit_txt2img(params), 'start image').images[0]


def main() -> None:
    """Parse arguments, generate the frames and write the animation."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_generation_args(parser)
    parser.add_argument('--image', help='Start from this image instead of generating one.')
    parser.add_argument('--rounds', type=int, default=6, help='Number of zoom-outs (default: 6).')
    parser.add_argument('--shrink', type=float, default=0.6, help='Size of each frame inside the next, 0.2-0.9.')
    parser.add_argument('--strength', type=float, default=0.95, help='Inpainting strength (default: 0.95).')
    parser.add_argument('--frames-per-round', type=int, default=12, help='Animation frames per zoom (default: 12).')
    parser.add_argument('--frame-ms', type=int, default=60, help='Milliseconds per animation frame (default: 60).')
    parser.add_argument('--animation', type=Path, default=Path('zoom.gif'), help='Output .gif or .webp path.')
    parser.add_argument('--save-keyframes', action='store_true', help='Also save each round\'s frame to --output-dir.')
    add_connection_args(parser)
    args = parser.parse_args()
    if not 0.2 <= args.shrink <= 0.9:
        parser.error('--shrink must be between 0.2 and 0.9')
    if args.rounds < 1 or args.frames_per_round < 1:
        parser.error('--rounds and --frames-per-round must be at least 1')
    if args.animation.suffix.lower() not in ('.gif', '.webp'):
        parser.error('--animation must end in .gif or .webp')

    backend = connect(args)
    size_args = {'width': round_size(args.width or 512), 'height': round_size(args.height or 512)}
    params = params_from_args(args, batch_size=1, **size_args)
    frames = [first_frame(backend, args, params)]
    for round_number in range(1, args.rounds + 1):
        canvas, mask = zoom_out_canvas(frames[-1], args.shrink)
        round_params = params.model_copy(update={'init_images': [canvas], 'mask': mask,
                                                 'denoising_strength': args.strength})
        frames.append(run_job(backend.submit_inpaint(round_params), f'round {round_number}/{args.rounds}').images[0])

    if args.save_keyframes:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for number, frame in enumerate(frames):
            save_png(frame, args.output_dir / f'zoom-{number:02d}.png')
    sequence = [frame.convert('RGB') for frame in build_animation(frames, args.shrink, args.frames_per_round)]
    sequence[0].save(args.animation, save_all=True, append_images=sequence[1:], duration=args.frame_ms, loop=0)
    print(f'Wrote {len(sequence)} frames to {args.animation}', file=sys.stderr)
    print(args.animation)


if __name__ == '__main__':
    cli_main(main)
