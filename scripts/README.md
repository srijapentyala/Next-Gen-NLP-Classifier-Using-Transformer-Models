Helper scripts in `scripts/`

- `extract_images_from_notebook.py` — Extracts PNG/JPEG/SVG outputs and attachments from a notebook and saves them to an output folder (useful to capture figures produced in Colab and commit into `assets/`).
-  `rename_assets_by_notebook_context.py` — Renames images in assets folder).

Usage example:

```bash
mkdir -p assets
python scripts/extract_images_from_notebook.py --notebook main_notebook.ipynb --outdir assets
```

After running, review `assets/` and commit the images you want to include. Avoid committing large binary data or model files.

