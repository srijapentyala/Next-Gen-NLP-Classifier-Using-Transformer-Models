"""
Clean notebook metadata by removing `widgets` entries from top-level metadata.
Usage: python scripts/clean_notebooks.py
"""
import json
import os

NOTEBOOKS = [
    "main_notebook.ipynb",
    os.path.join("checkpoints", "checkpoint_2.ipynb")
]

for nb_file in NOTEBOOKS:
    if not os.path.exists(nb_file):
        print(f"[WARN] Notebook not found: {nb_file}")
        continue
    try:
        with open(nb_file, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read {nb_file}: {e}")
        continue

    removed = False
    if isinstance(nb, dict) and 'metadata' in nb and 'widgets' in nb['metadata']:
        nb['metadata'].pop('widgets', None)
        removed = True

    # Optionally also remove 'widgets' from cell metadata if present
    for cell in nb.get('cells', []):
        if isinstance(cell, dict) and 'metadata' in cell and 'widgets' in cell['metadata']:
            cell['metadata'].pop('widgets', None)
            removed = True

    if removed:
        try:
            with open(nb_file, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f"[INFO] Cleaned widgets metadata from {nb_file}")
        except Exception as e:
            print(f"[ERROR] Failed to write {nb_file}: {e}")
    else:
        print(f"[INFO] No widgets metadata found in {nb_file}")
