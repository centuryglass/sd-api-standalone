"""Play telephone with a Stable Diffusion server: describe an image, redraw the description, and repeat.

Each round interrogates the current image for a caption, then runs img2img with the caption as the prompt. The images
and captions drift further from the original each round. Needs a WebUI server, since only it can interrogate.

Usage:
    python examples/telephone.py START_IMAGE [--rounds 5] [--strength 0.8] [--interrogate-model NAME] [-o DIR] \\
        [txt2img flags other than the prompt]
"""
import argparse
from PIL import Image

from _common import add_connection_args, add_generation_args, cli_main, connect, load_image, params_from_args, \
    require_webui, round_size, run_job, save_png


def main() -> None:
    """Parse arguments, run the rounds, and save each round's image and caption."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('start_image', help='Path to the first image.')
    add_generation_args(parser, prompt_required=False)
    parser.add_argument('--rounds', type=int, default=5, help='Number of describe-and-redraw rounds (default: 5).')
    parser.add_argument('--strength', type=float, default=0.8, help='img2img strength per round (default: 0.8).')
    parser.add_argument('--interrogate-model', default=None, help="Interrogation model (default: the client's).")
    add_connection_args(parser)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be at least 1')

    webui = require_webui(connect(args), 'telephone')
    image: Image.Image = load_image(args.start_image).convert('RGB')
    size = {'width': round_size(args.width or image.width), 'height': round_size(args.height or image.height)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_png(image, args.output_dir / 'round-00.png')
    captions = []
    for round_number in range(1, args.rounds + 1):
        caption: str = webui.interrogate(image, args.interrogate_model)
        captions.append(caption)
        print(f'round {round_number}: {caption}')
        params = params_from_args(args, prompt=caption, init_images=[image], denoising_strength=args.strength, **size)
        image = run_job(webui.submit_img2img(params), f'round {round_number}/{args.rounds}').images[0]
        save_png(image, args.output_dir / f'round-{round_number:02d}.png')
    final_caption = webui.interrogate(image, args.interrogate_model)
    captions.append(final_caption)
    print(f'final: {final_caption}')
    (args.output_dir / 'captions.txt').write_text('\n'.join(captions) + '\n', encoding='utf-8')


if __name__ == '__main__':
    cli_main(main)
