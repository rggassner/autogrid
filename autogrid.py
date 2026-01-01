#!venv/bin/python3
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

def dummy(images, **kwargs):
    return images, [False] * len(images)



def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =========================================================
# Grid Scraping + Mask Construction
# =========================================================

def get_image_mask_rows(image):
    """
    Scrapes grid, reconstructs base image,
    and builds inpainting mask for missing tiles.
    """

    fn = lambda x: 255 if x > 254 else 0
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
            except Exception as e:
                print(f"[ERROR] request failed: {e}")
                continue

            content_type = response.headers.get("Content-Type", "")

            if response.status_code != 200 or not content_type.startswith("image/"):
                print(
                    f"[SKIP] HTTP {response.status_code} "
                    f"{content_type} "
                    f"bytes={len(response.content)}"
                )
                with open("failed_urls.txt", "a") as f:
                    f.write(full_url + "\n")
                continue

            try:
                part = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                print(f"[SKIP] PIL error: {e}")
                with open("failed_urls.txt", "a") as f:
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

def gen_images(args):
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

