Data placement and usage

This folder is a placeholder for local data files you may wish to provide instead of downloading from HuggingFace inside the notebook.

Recommended layout:

- `dbpedia_csv/train.csv`  (OPTIONAL: large — do not commit to GitHub)
- `dbpedia_csv/test.csv`   (OPTIONAL)
- `dbpedia_csv/classes.txt` (small)

Notes:
- The notebook (`main_notebook.ipynb`) will automatically download DBpedia via the HuggingFace `datasets` library unless local CSVs are detected under `dbpedia_csv/`.
- Do NOT commit large files to the repository. If you must include large artifacts, use Git LFS or an external storage link and add instructions here.

Direct dataset mirror (provided by the user):

- Google Drive mirror (download URL):
  https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k

How to use the mirror locally:

1. Download the file to your local machine. If the file is a zip or tar archive, unzip it and place CSVs under `dbpedia_csv/`.

2. Suggested commands (macOS / Linux):

```bash
mkdir -p dbpedia_csv
cd dbpedia_csv
# replace <FILE> below with the downloaded filename
mv ~/Downloads/your_downloaded_file.zip ./
unzip your_downloaded_file.zip
# or if it's a CSV already:
# mv ~/Downloads/train.csv ./
```

3. After placing files under `dbpedia_csv/`, open `main_notebook.ipynb` and the notebook will detect the local files and use them instead of downloading.

If you'd prefer to keep large files outside the repo (recommended), upload them to a cloud storage bucket (Drive/Dropbox/S3) and add the download link(s) here for reviewers.

Example Drive link (already included above):
- https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k

