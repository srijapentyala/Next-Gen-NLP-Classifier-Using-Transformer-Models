Data placement and usage

This directory documents how to provide DBpedia CSVs locally so `main_notebook.ipynb` can load them without re-downloading from the internet.

Recommended local layout

- `dbpedia_csv/train.csv`   # training CSV (large — do NOT commit)
- `dbpedia_csv/test.csv`    # test CSV (large — do NOT commit)
- `dbpedia_csv/classes.txt` # optional: one label per line

The notebook will look for files under `dbpedia_csv/` first. If not found, it falls back to the HuggingFace `datasets` loader.

Option A — Download provided Google Drive mirror (recommended if you want a direct local copy)

1. Install `gdown` (if not already installed):

```bash
pip install gdown
```

2. Download the mirror to the repo root and unpack if needed (macOS/Linux):

```bash
cd /path/to/Next-Gen-NLP-Classifier-Using-Transformer-Models
mkdir -p dbpedia_csv
# using the provided share ID
gdown --id 0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k -O dbpedia_csv/dbpedia_mirror.zip
unzip dbpedia_csv/dbpedia_mirror.zip -d dbpedia_csv/
# or if the link points directly to CSVs, move them into dbpedia_csv/
```

Notes: Google Drive links can be rate-limited; if you get permission/quota errors, download manually via a browser and then move files into `dbpedia_csv/`.

Option B — Programmatic download using HuggingFace `datasets` (no manual download required)

This is the easiest option inside Colab or a Python environment:

```python
from datasets import load_dataset
ds = load_dataset('dbpedia_14')
# ds['train'].to_csv('dbpedia_csv/train.csv', index=False)
# ds['test'].to_csv('dbpedia_csv/test.csv', index=False)
import os
os.makedirs('dbpedia_csv', exist_ok=True)
ds['train'].to_csv('dbpedia_csv/train.csv', index=False)
ds['test'].to_csv('dbpedia_csv/test.csv', index=False)
```

Option C — Manual download

1. Download via browser from the Drive mirror URL.
2. Move the extracted CSVs to `dbpedia_csv/`:

```bash
mkdir -p dbpedia_csv
mv ~/Downloads/train.csv dbpedia_csv/
mv ~/Downloads/test.csv dbpedia_csv/
```

Quick verification steps (after files are in `dbpedia_csv/`)

```bash
# show file sizes and head of CSVs
ls -lh dbpedia_csv
head -n 5 dbpedia_csv/train.csv
wc -l dbpedia_csv/train.csv dbpedia_csv/test.csv
```

Python checks you can run in Colab/notebook

```python
import pandas as pd
df = pd.read_csv('dbpedia_csv/train.csv')
print('rows,cols:', df.shape)
print(df.columns)
print(df.head(3).to_dict(orient='records'))
```

Important notes

- DO NOT commit large CSV files or model checkpoints to the Git repository — they will exceed GitHub limits. Use Git LFS or external hosting (Drive, S3) and add links here instead.
- If you host files externally, include direct download commands (gdown / curl) and a short checksum (md5/sha256) so reviewers can verify integrity.
- If you prefer we can add a small helper script (`scripts/download_data.sh`) to download and unpack the mirror; tell me and I will create it.

Questions or issues

If a download fails in Colab because of rate limits or network issues, prefer the HuggingFace programmatic route (Option B) — it's robust and fast inside Colab.

---

Direct dataset mirror (provided by the user):

https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k

---

End of data/README.md

