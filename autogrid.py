#!venv/bin/python3
"""
Grid-based image outpainting using Stable Diffusion inpainting.

This script scrapes a remote grid-style image layout, reconstructs the
available tiles into a composite canvas, and uses Stable Diffusion
inpainting to synthesize missing regions. The generated result is then
iteratively refined to produce multiple variations, which are saved
both as full images and as individual grid tiles.

Core features
-------------
- Scrapes a 3×3 grid of images from a target website.
- Rebuilds the known tiles into a base image.
- Automatically constructs an inpainting mask for missing cells.
- Uses a cached Stable Diffusion inpainting model for fast iteration.
- Supports optional image embedding at user-defined coordinates.
- Generates multiple output variations in a single run.
- Saves per-iteration outputs in timestamped directories.

Intended use
------------
This tool is designed for experimental image exploration, procedural
outpainting, and grid-based visual expansion workflows. It is especially
useful when working with tiled or partially-known image layouts that
benefit from generative completion.

Requirements
------------
- CUDA-capable GPU
- PyTorch
- diffusers
- Pillow (PIL)
- BeautifulSoup4
- requests

Execution
---------
Run the script from the command line and provide prompts and parameters
via CLI arguments. See ``--help`` for details.

Note
----
NSFW filtering can be disabled via configuration. Use responsibly and
ensure compliance with model and content policies.
"""
from io import BytesIO
import os
import argparse
import datetime
import torch
import requests
from PIL import Image
from bs4 import BeautifulSoup
from diffusers import StableDiffusionInpaintPipeline

# =========================================================
# Defaults / Config
# =========================================================

N_OPTIONS = 200
TILE_SIZE = 150
SD_SIZE = 512

ALLOW_NSFW = True

URL = "https://www.sito.org/cgi-bin/gridcosm/gridcosm?level=top"
BASEURL = "https://www.sito.org"

MODEL_ID = "5w4n/deliberate-v2-inpainting"
MODEL_CACHE = "/home/rgg/hf_models"




SEED = 1337
DEVICE = "cuda"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

# =========================================================
# Helpers
# =========================================================

#def dummy(images, **kwargs):
#    return images, False

def dummy(images, **kwargs): #pylint: disable=unused-argument
    """
    Bypass safety checking by marking all images as safe.

    This function acts as a drop-in replacement for a Stable Diffusion
    safety checker. It returns the input images unchanged and marks each
    image as non-NSFW.

    Parameters
    ----------
    images : list
        A list of generated images.
    **kwargs
        Ignored keyword arguments included for API compatibility.

    Returns
    -------
    tuple
        A tuple ``(images, flags)`` where ``flags`` is a list of ``False``
        values indicating no NSFW content was detected.
    """
    return images, [False] * len(images)



def ensure_dir(path):
    """
    Ensure that a directory exists.

    This function creates the specified directory path if it does not
    already exist. Intermediate directories are created as needed.

    Parameters
    ----------
    path : str
        Directory path to create or verify.

    Notes
    -----
    - Directory creation is idempotent.
    - Uses ``exist_ok=True`` to avoid errors if the directory already exists.
    """
    os.makedirs(path, exist_ok=True)


# =========================================================
# Grid Scraping + Mask Construction
# =========================================================

def get_image_mask_rows(image): #pylint: disable=too-many-locals
    """
    Scrapes grid, reconstructs base image,
    and builds inpainting mask for missing tiles.
    """

    fn = lambda x: 255 if x > 254 else 0 #pylint: disable=unnecessary-lambda-assignment
    mask = image.convert("L").point(fn, mode="1")
    pmask = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))

    page = requests.get(URL, headers=HEADERS, timeout=15).text
    soup = BeautifulSoup(page, "html.parser")

    rows = soup.find_all(
        "table",
        attrs={
            "cellpadding": "0",
            "style": "width:100%; max-width: 450px;display:inline-block;vertical-align:top;"
        }
    )[0].find_all("tr")

    rowcount = 1

    for row in rows:
        data = row.find_all("td")

        for col in range(3):
            imgs = data[col].find_all("img")
            if len(imgs) != 1:
                continue

            img_src = imgs[0].get("src")
            full_url = BASEURL + img_src

            print(f"[FETCH] row={rowcount} col={col+1} → {full_url}")

            try:
                response = requests.get(
                    full_url,
                    headers=HEADERS,
                    timeout=15
                )
            except Exception as e: #pylint: disable=broad-exception-caught
                print(f"[ERROR] request failed: {e}")
                continue

            content_type = response.headers.get("Content-Type", "")

            if response.status_code != 200 or not content_type.startswith("image/"):
                print(
                    f"[SKIP] HTTP {response.status_code} "
                    f"{content_type} "
                    f"bytes={len(response.content)}"
                )
                with open("failed_urls.txt", "a") as f: #pylint: disable=unspecified-encoding
                    f.write(full_url + "\n")
                continue

            try:
                part = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e: #pylint: disable=broad-exception-caught
                print(f"[SKIP] PIL error: {e}")
                with open("failed_urls.txt", "a") as f: #pylint: disable=unspecified-encoding
                    f.write(full_url + "\n")
                continue

            image.paste(
                part,
                (col * TILE_SIZE, (rowcount - 1) * TILE_SIZE)
            )
            mask.paste(
                pmask,
                (col * TILE_SIZE, (rowcount - 1) * TILE_SIZE)
            )

        rowcount += 1

    mask = mask.resize((SD_SIZE, SD_SIZE), Image.Resampling.LANCZOS)
    image = image.resize((SD_SIZE, SD_SIZE), Image.Resampling.LANCZOS)

    mask.save("outm.png")
    image.save("outi.png")

    return image, mask, rows


# =========================================================
# Argument Parsing
# =========================================================

def read_arguments():
    """
    Parse command-line arguments for the grid outpainting workflow.

    This function defines and parses all CLI options required to control
    Stable Diffusion inpainting and grid-based outpainting. It returns an
    ``argparse.Namespace`` containing generation prompts, inference settings,
    and optional image embedding parameters.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with the following attributes:
        - text (str): Prompt describing the desired image.
        - negative (str): Negative prompt to suppress unwanted features.
        - steps (int): Number of inference steps.
        - seed (int): Random seed for reproducible generation.
        - embed (str | None): Optional path to an image to embed.
        - embedx (int): X-coordinate for embedded image placement.
        - embedy (int): Y-coordinate for embedded image placement.

    Notes
    -----
    - Required arguments must be explicitly provided on the command line.
    - Defaults are applied where appropriate (e.g., ``SEED``).
    - Intended to be used as the single source of CLI configuration.
    """
    parser = argparse.ArgumentParser(
        description="Grid outpainting with cached SD inpainting pipeline"
    )

    parser.add_argument("-t", "--text", required=True)
    parser.add_argument("-n", "--negative", required=True)
    parser.add_argument("-s", "--steps", type=int, required=True)

    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument("--embed")
    parser.add_argument("--embedx", type=int, default=0)
    parser.add_argument("--embedy", type=int, default=0)

    return parser.parse_args()


# =========================================================
# Main Generation Logic
# =========================================================

def gen_images(args): #pylint: disable=too-many-locals
    """
    Generate image variations using Stable Diffusion inpainting and tile extraction.

    This function builds a 3×3 canvas, optionally embeds a user-provided image,
    runs Stable Diffusion inpainting to generate multiple variations, and then
    slices the generated result into individual tiles based on a detected grid.
    Each iteration is saved to a timestamped output directory.

    Parameters
    ----------
    args : argparse.Namespace
        Runtime arguments controlling the generation process, including:
        - text: Prompt used for image generation.
        - negative: Negative prompt to steer generation away from undesired features.
        - steps: Number of inference steps.
        - seed: Random seed for deterministic output.
        - embed: Optional image path to embed into the canvas.
        - embedx / embedy: Coordinates where the embedded image is placed.

    Side Effects
    ------------
    - Creates a timestamped output directory and per-iteration subdirectories.
    - Writes generated images and cropped tile images to disk.
    - Allocates GPU memory and clears cache between iterations.

    Notes
    -----
    - Uses ``StableDiffusionInpaintPipeline`` with half-precision (FP16).
    - NSFW filtering can be disabled via ``ALLOW_NSFW``.
    - Tile extraction depends on ``get_image_mask_rows`` to detect valid regions.
    - Designed for batch exploration of multiple generation options.
    """
    timestamp = str(datetime.datetime.now().timestamp())
    ensure_dir(timestamp)

    image = Image.new(
        "RGB",
        (TILE_SIZE * 3, TILE_SIZE * 3),
        (255, 255, 255)
    )

    if args.embed:
        embed = Image.open(args.embed)
        image.paste(embed, (args.embedx, args.embedy))

    image, mask, rows = get_image_mask_rows(image)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE,
        torch_dtype=torch.float16,
        use_safetensors=False,
        local_files_only=False,
    ).to(DEVICE)

    if ALLOW_NSFW:
        pipe.safety_checker = dummy

    generator = torch.Generator(device=DEVICE).manual_seed(args.seed)

    for iteration in range(1, N_OPTIONS + 1):
        torch.cuda.empty_cache()

        result = pipe(
            prompt=args.text,
            negative_prompt=args.negative,
            image=image,
            mask_image=mask,
            num_inference_steps=args.steps,
            generator=generator
        ).images[0]

        iter_dir = os.path.join(timestamp, str(iteration))
        ensure_dir(iter_dir)

        result.save(os.path.join(timestamp, f"{iteration}.png"))

        resized = result.resize(
            (TILE_SIZE * 3, TILE_SIZE * 3),
            Image.Resampling.LANCZOS
        )

        rowcount = 1
        for row in rows:
            data = row.find_all("td")
            for col in range(3):
                if len(data[col].find_all("img")) != 1:
                    tile = resized.crop(
                        (
                            col * TILE_SIZE,
                            (rowcount - 1) * TILE_SIZE,
                            (col + 1) * TILE_SIZE,
                            rowcount * TILE_SIZE
                        )
                    )
                    tile.save(
                        os.path.join(
                            iter_dir,
                            f"{col + 1}-{rowcount}.png"
                        )
                    )
            rowcount += 1

        print(f"[✓] Option {iteration} done")

# =========================================================
# Entry
# =========================================================

def main():
    """
    Entry point for the grid-based Stable Diffusion outpainting pipeline.

    This function parses command-line arguments, initializes the runtime
    configuration, and launches the image generation workflow. The pipeline:

    - Scrapes a remote grid-based image layout from a target website.
    - Reconstructs the known tiles into a composite canvas.
    - Builds an inpainting mask for missing grid cells.
    - Loads a cached Stable Diffusion inpainting model.
    - Iteratively generates multiple outpainted variations.
    - Saves the full generated image and individual synthesized tiles
      for each iteration.

    The function acts as the orchestration layer, delegating scraping,
    masking, model inference, and output management to helper functions.

    Command-line arguments control prompts, inference steps, random seed,
    and optional embedded imagery.

    Returns:
        None
    """
    args = read_arguments()
    gen_images(args)

if __name__ == "__main__":
    main()
