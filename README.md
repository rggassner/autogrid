# Installation

 python3 -m venv .venv
 source .venv/bin/activate
 python3 -m pip install requirements.txt
 
# AutoGrid Outpainting

AutoGrid is a Python-based pipeline that scrapes a grid-based image layout from a remote website, reconstructs the known tiles, generates an inpainting mask for missing cells, and uses **Stable Diffusion inpainting** to outpaint the missing regions.  

The tool is designed for **iterative generation**, producing multiple plausible completions of the same grid and exporting both full images and individual synthesized tiles.

---

##  Features

- Scrapes grid-based image layouts from a remote HTML page
- Reconstructs known image tiles into a composite canvas
- Automatically builds an inpainting mask for missing grid cells
- Uses Stable Diffusion **inpainting** models
- Generates multiple output variations per run
- Exports:
  - Full generated images
  - Individual generated tiles only (no original tiles)
- Deterministic runs via manual random seed
- Robust scraping with:
  - Custom User-Agent
  - Content-type validation
  - Graceful handling of broken or missing images
  - Failed URL logging

---

##  How It Works (High Level)

1. Fetches the grid HTML page.
2. Parses the grid layout using BeautifulSoup.
3. Downloads existing image tiles.
4. Reconstructs the grid into a single canvas.
5. Builds a binary mask marking missing tiles.
6. Loads a Stable Diffusion inpainting pipeline.
7. Runs multiple inference passes.
8. Crops and saves only the newly generated tiles.

---

##  Requirements

- Python 3.10+ (tested on Python 3.12)
- CUDA-capable GPU
- PyTorch with CUDA support
- Hugging Face Diffusers
- Internet access (for scraping and model download)


