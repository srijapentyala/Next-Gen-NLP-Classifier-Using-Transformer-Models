#!/usr/bin/env python3
"""
Rename assets extracted from a notebook from names like `cell37_output1.png`
to descriptive filenames using the nearest preceding markdown heading.

Usage:
    python scripts/rename_assets_by_notebook_context.py --notebook main_notebook.ipynb --assets assets

The script will print the performed renames and keep originals as backup by default
(unless --overwrite is passed).
"""
import argparse
import json
from pathlib import Path
import re

HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(.+)")


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s)
    s = s.strip("-")
    return s[:80]


def find_heading_for_cell(nb, idx):
    # idx is 1-based cell index
    # search backwards up to 8 cells for a markdown heading
    i = idx - 1
    for back in range(0, 9):
        j = i - back
        if j < 0:
            break
        cell = nb["cells"][j]
        if cell.get("cell_type") == "markdown":
            # join source lines
            src = "\n".join(cell.get("source", []))
            for line in src.splitlines()[0:5]:
                m = HEADING_RE.match(line)
                if m:
                    return m.group(1).strip()
            # fallback: return first non-empty line
            for line in src.splitlines():
                if line.strip():
                    return line.strip()[:80]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", "-n", required=True)
    parser.add_argument("--assets", "-a", default="assets")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    nb_path = Path(args.notebook)
    assets_dir = Path(args.assets)
    if not nb_path.exists():
        print("Notebook not found:", nb_path)
        return
    if not assets_dir.exists():
        print("Assets dir not found:", assets_dir)
        return

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    files = sorted(p for p in assets_dir.iterdir() if p.is_file())

    renamed = []
    for f in files:
        m = re.match(r"cell(\d+)_output(\d+)(\.[a-zA-Z0-9]+)$", f.name)
        if not m:
            continue
        cell_idx = int(m.group(1))
        out_idx = int(m.group(2))
        ext = m.group(3)
        heading = find_heading_for_cell(nb, cell_idx)
        if heading:
            slug = slugify(heading)
            new_name = f"{cell_idx:03d}-{slug}{ext}"
        else:
            new_name = f"cell{cell_idx}_output{out_idx}{ext}"

        dest = assets_dir / new_name
        # avoid overwriting unless requested
        if dest.exists() and not args.overwrite:
            # add numeric suffix
            k = 1
            while True:
                alt = assets_dir / f"{cell_idx:03d}-{slug}-{k}{ext}"
                if not alt.exists():
                    dest = alt
                    break
                k += 1
        f.rename(dest)
        renamed.append((f.name, dest.name))

    print("Renamed files:")
    for old, new in renamed:
        print(f"  {old} -> {new}")

if __name__ == '__main__':
    main()
