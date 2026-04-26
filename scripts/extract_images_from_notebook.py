#!/usr/bin/env python3
"""
Extract image outputs and attachments from a Jupyter notebook and save them to an assets folder.

Usage:
    python scripts/extract_images_from_notebook.py --notebook main_notebook.ipynb --outdir assets

This works in Colab or locally. It will decode image/png and image/jpeg outputs and save SVG attachments as .svg files.
"""
import argparse
import base64
import json
import os
from pathlib import Path

MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/svg+xml": ".svg",
}


def save_binary(data_b64, path: Path):
    data = base64.b64decode(data_b64)
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description="Extract images from a .ipynb file into an output directory")
    parser.add_argument("--notebook", "-n", required=True, help="Path to the notebook file (e.g., main_notebook.ipynb)")
    parser.add_argument("--outdir", "-o", default="assets", help="Output directory for extracted images")
    args = parser.parse_args()

    nb_path = Path(args.notebook)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    found = 0
    for i, cell in enumerate(nb.get("cells", []), start=1):
        # 1) attachments on markdown cells
        attachments = cell.get("attachments") or {}
        for a_name, a_data in attachments.items():
            for mime, content in a_data.items():
                ext = MIME_EXT.get(mime, None) or Path(a_name).suffix or ".bin"
                filename = f"cell{i}_attachment_{a_name}{ext if ext.startswith('.') else '.'+ext}"
                filepath = outdir / filename
                # content may be a list of lines or a string
                if isinstance(content, list):
                    content = "".join(content)
                if mime.startswith("image/"):
                    save_binary(content, filepath)
                    found += 1

        # 2) outputs on code cells
        for j, out in enumerate(cell.get("outputs", []), start=1):
            data = out.get("data") or {}
            for mime, content in data.items():
                if mime not in MIME_EXT:
                    continue
                ext = MIME_EXT[mime]
                filename = f"cell{i}_output{j}{ext}"
                filepath = outdir / filename
                # content may be list or single string
                if isinstance(content, list):
                    content = "".join(content)
                if isinstance(content, str):
                    save_binary(content, filepath)
                    found += 1

    print(f"Extracted {found} images to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
