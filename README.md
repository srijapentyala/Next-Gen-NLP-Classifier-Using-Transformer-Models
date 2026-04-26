# Does Context Actually Matter for Text Classification?


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

3) Data

- Dataset: DBpedia 14 (HuggingFace id: `dbpedia_14`), ~560k train / ~70k test. A Drive mirror is available if you prefer a local copy:
   - Drive mirror: `https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k`

Dataset summary (short):

DBpedia 14 is a standard 14-class benchmark of Wikipedia article titles and short abstracts (used widely for multi-class text classification). The dataset contains ~560k training examples and ~70k test examples, with each row providing a label, title, and abstract.

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
├── .gitignore         # Local commit/push instructions
```

---

## Results & Conclusion — detailed takeaways

This section gives a concise, reproducible summary of the numeric results, important qualitative observations, and practical guidance for choosing a model for similar classification tasks.

Headline numbers (test set):

- TF‑IDF + Logistic Regression (RQ1): ~98.40% accuracy and comparable Macro F1. Training and evaluation are very fast (minutes on CPU) and require negligible GPU resources.
- TF‑IDF + TruncatedSVD (RQ2): ~96.5% accuracy — dimensionality reduction reduced signal from rare but discriminative tokens and increased confusion between related classes.
- DistilBERT (RQ3): fine-tuning on a stratified ~8K subset yields ~98.89% accuracy (strong sample-efficiency). Fine-tuning on the full training set produces a higher ceiling (~99.6%) but requires substantially more GPU time (order-of-magnitude slower; in our runs the full-data run took ~55× more wall-clock GPU time than the 8K run).

Per-class observations and failure modes:

- Several small classes (few-shot labels) show degraded recall in the TF‑IDF pipelines because they rely on rare tokens that get lost when using SVD or aggressive token pruning.
- DistilBERT improves per-class recall for classes with subtle contextual cues — it captures phrase-level patterns that TF‑IDF misses.
- The confusion matrices (see assets) show the most common confusions are semantically close classes (e.g., different types of companies or locations); these are good candidates for label-merging or hierarchical classification in future work.

Compute, time, and resource notes (reproducibility-friendly):

- Environment: experiments were run in Google Colab with a T4 GPU for transformer fine-tuning. The notebook records the Python and package versions; export `requirements.txt` from Colab to reproduce exact runtime.
- DistilBERT hyperparameters used for reported runs (notebook cells): seed=42, max_length=128, batch_size=16 (8 for GPU memory-limited runs), lr=2e-5, epochs=3 for 8K and epochs=2–3 for larger runs depending on learning curves.
- Checkpoints and model artifacts were saved to an output directory (the notebook uses a `checkpoints/` folder). Avoid committing large checkpoints to GitHub — use Drive or S3 for storage and link them in `data/README.md`.

Practical recommendations (actionable):

- Quick evaluation / baseline: Run TF‑IDF + Logistic Regression first (minutes) to establish a reliable lower bound.
- If you have GPU and <50k labels: fine-tune DistilBERT on a stratified 8K subset to measure sample-efficiency gains before committing to a full fine-tune.
- If the final accuracy gain is small (<0.5 percentage point) but cost is high, prefer TF‑IDF for production (faster, cheaper, interpretable).
- Use per-class error analysis (confusion matrices and LIME explanations in the notebook) to decide if label consolidation or targeted data collection is more effective than wholesale model replacement.

Visual highlights (extracted from the notebook):

- Performance breakdown (per-class): `assets/060-detailed-performance-breakdown-by-class.png`

![Per-class performance](assets/060-detailed-performance-breakdown-by-class.png)

- Cross-model comparison (accuracy vs F1): `assets/090-phase-8-cross-model-comparison.png`
![Model comparison](assets/090-phase-8-cross-model-comparison.png)

## Future work — short actionable directions

1. k-fold cross-validation on the 8K DistilBERT experiments — run 5‑fold stratified CV to report mean ± CI and surface high‑variance classes.

2. Parameter‑efficient fine‑tuning (LoRA / adapters) — evaluate LoRA/adapters to reduce GPU/time at near-equivalent accuracy.

3. Targeted augmentation for low‑recall classes — use back‑translation/paraphrasing or synonym injection to boost recall for rare labels.

4. Calibration & confidence‑based rejection — calibrate probabilities and add a reject option to reduce high‑impact overconfident errors in deployment.

---

