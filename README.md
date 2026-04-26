drive.mount('/content/drive')

# Does Context Actually Matter for Text Classification?

One-paragraph overview

This project compares classical bag-of-words pipelines (TF‑IDF + Logistic Regression) with contextual transformer fine‑tuning (DistilBERT) on the DBpedia 14 Wikipedia benchmark to answer when and why context helps. We run a three-stage study (TF‑IDF baseline, SVD compression test, DistilBERT sample-efficiency and full-data experiments), provide EDA, explainability (LIME), and a reproducible Colab-first workflow so reviewers can reproduce the results quickly.

Project video (watch first) 🎥

https://www.youtube.com/watch?v=tJMxr0M0WzE

---

1) Deliverable (exact file to open)

👉 The main deliverable is `main_notebook.ipynb` — open that file first and run top → bottom in Google Colab (GPU recommended for RQ3). Checkpoints live in `checkpoints/`.

---

2) Research questions

- RQ1 — How well does TF‑IDF + Logistic Regression perform on DBpedia 14?
- RQ2 — Does compressing TF‑IDF features with TruncatedSVD help or hurt predictive performance?
- RQ3 — Does a contextual model (DistilBERT) outperform TF‑IDF, and how many labeled examples are needed to match or exceed the baseline (8K vs full)?

---

3) Data — what, where, and preprocessing summary

- Dataset: DBpedia 14 (HuggingFace id: `dbpedia_14`), ~560k train / ~70k test. A Drive mirror is available if you prefer a local copy:
   - Drive mirror: `https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k`

Preprocessing (high level — fully implemented in `main_notebook.ipynb`):

- File discovery: notebook prefers local CSVs under `dbpedia_csv/` (or `data/dbpedia_csv/`), otherwise loads from HuggingFace and falls back to the Drive mirror via `gdown`.
- Robust CSV parsing: `safe_read_csv()` handles plain and gzipped CSVs, uses a tolerant CSV reader, and enforces columns `[label, title, text]` to avoid misaligned files.
- Text assembly: `title` and `text` are concatenated with a separator (`". "`) to form the model input.
- Label handling: labels normalized to 0–13 (0-based) with an assertion that the final label set is contiguous and size 14.
- Filtering: remove null/very-short texts (<5 chars) to avoid noisy examples.
- Splits: stratified 80/20 split; create a stratified 8K training subset for the sample-efficiency DistilBERT experiment.
- Tokenization: DistilBERT tokenizer with `max_length=128`, `padding='max_length'`, `truncation=True` (chosen after EDA showed minimal loss).
- Safety checks: assertions after each stage (class counts, nulls, stratification balance) to fail fast if something goes wrong.

---

4) How to reproduce (Colab-first, step-by-step)

1. Open `main_notebook.ipynb` in Google Colab.
2. (Optional) Mount Google Drive if you want to persist downloads or save models:

```python
from google.colab import drive
drive.mount('/content/drive')
# set DRIVE_ROOT = '/content/drive/MyDrive/dbpedia_csv' if you want persistent storage
```

3. Change runtime type → GPU (T4 recommended) for DistilBERT fine-tuning.
4. Install dependencies (or run the notebook helper cell which installs core libs):

```bash
!pip install -r requirements.txt
```

5. Recommended run order:
- Run EDA and RQ1 (TF‑IDF + Logistic Regression) first — fast on CPU.
- Run RQ2 (TF‑IDF + SVD) next if you want the compression baseline.
- Run RQ3 DistilBERT: first the 8K stratified fine-tune (short) then the full-data fine-tune (long). Use Colab GPU for both.

6. Export exact Colab environment for bit-for-bit reproducibility (run near the notebook end):

```python
!pip freeze > requirements.txt
from google.colab import files
files.download('requirements.txt')
```

Record the Python version as well:

```python
!python --version
```

7. (Optional) Extract notebook figures into `assets/` for easy review:

```bash
python scripts/extract_images_from_notebook.py --notebook main_notebook.ipynb --outdir assets
```

Notes: avoid committing large CSVs or model weights to GitHub; use Git LFS or external hosting and update `data/README.md` with links.

---

5) Key dependencies (high-level)

- Python 3.11 (recommended in Colab)
- torch (GPU build recommended; notebook tested with torch==2.10.0+cu128)
- transformers==5.0.0
- datasets (HuggingFace) ~4.x
- scikit-learn==1.6.1
- pandas==2.2.2
- numpy==2.0.2
- matplotlib, seaborn
- lime (explainability)

Full dependency freeze should be exported from Colab to `requirements.txt` and committed for exact reproducibility.

---

6) Repo structure (short tree)

```
Next-Gen-NLP-Classifier-Using-Transformer-Models/
├── main_notebook.ipynb            # Curated final notebook (start here)
├── requirements.txt               # Exported from Colab (session-specific freeze recommended)
├── README.md                      # This file (rubric-compliant)
├── checkpoints/                   # Checkpoint notebooks
│   ├── checkpoint_1.ipynb
│   └── checkpoint_2.ipynb
├── scripts/                       # Data helpers & image extraction tools
├── data/                          # Placeholder + download instructions (do not commit large files)
├── assets/                        # Extracted figures from the notebook
├── .gitignore
└── COMMIT_INSTRUCTIONS.md         # Local commit/push instructions
```

---

7) Results — short headline

Headline: TF‑IDF is an excellent, low-cost baseline on DBpedia (≈98.4%); DistilBERT is more sample-efficient (8K fine-tune matches or slightly outperforms TF‑IDF) and reaches a marginally higher ceiling when fine-tuned on the full dataset (~99.6%) at a much higher compute cost. See `main_notebook.ipynb` for full tables, confusion matrices, and per-class analysis.

Visual highlights (if images extracted to `assets/`):

- `assets/060-detailed-performance-breakdown-by-class.png` — per-class metrics
- `assets/090-phase-8-cross-model-comparison.png` — cross-model comparison

---

8) Future work (short bullets)

- k-fold cross-validation on the 8K experiments to estimate variance.
- Parameter-efficient fine-tuning (LoRA / adapters) to reduce GPU/time cost.
- Evaluate on noisier domains (social media, OCR) to measure real-world benefits of context.
- Add latency/throughput benchmarks for deployment scenarios (CPU/GPU).
- Provide a Dockerfile or conda YAML for fully local reproducibility.

---

9) Assets & notes

- Use `scripts/extract_images_from_notebook.py` to populate `assets/` with figures used in the notebook. `assets/README.md` contains captions and file mappings.
- Do not commit raw dataset CSVs or heavy model checkpoints; instead include download instructions in `data/README.md` or use Git LFS / external hosting.

---

**End of README**
