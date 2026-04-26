# 🤖 Does Context Actually Matter for Text Classification?
### A Three-Stage Data-Driven Investigation on 560,000 Wikipedia Articles

> *"In the age of transformers, everyone assumes you need a GPU. This project answers when that assumption is actually correct — with 560,000 data points, not conference paper hype."*

**Student:** Srija Pentyala  
**Dataset:** DBpedia 14-Class Ontology Classification — 560K Wikipedia article abstracts  
**Python Version:** 3.11 (Google Colab)

---

## 👉 Start Here: [`main_notebook.ipynb`](main_notebook.ipynb)

📹 **Project Video:** [Watch on YouTube](https://www.youtube.com/watch?v=tJMxr0M0WzE)

---

## 📌 Project Overview

Every NLP team eventually faces the same infrastructure question: *should we deploy a lightweight classical pipeline, or invest in expensive transformer models?*

This project uses Wikipedia's DBpedia 14-class classification task to answer a foundational question in modern NLP:

> **Does reading words *in context* actually improve text classification — or is knowing *which* words appear enough?**

We answer it through a principled three-stage progression across 560,000 Wikipedia article abstracts, comparing classical bag-of-words methods against state-of-the-art transformer-based fine-tuning — with rigorous EDA, explainability analysis (LIME), and a practical deployment framework.

---

## 🔬 Research Questions

| Stage | Model | Research Question |
|-------|-------|-------------------|
| **RQ1** | TF-IDF + Logistic Regression | How far can word frequencies alone take us? |
| **RQ2** | TF-IDF + Truncated SVD + LogReg | Can compressing features into semantic topics help? |
| **RQ3** | Fine-Tuned DistilBERT | Does reading words in context close the remaining gap? |

Each stage is a direct response to the limitation discovered in the previous one — the story builds progressively toward a clear, actionable conclusion.

---

# 🤖 Does Context Actually Matter for Text Classification?

A curated, reproducible study comparing classical bag-of-words pipelines with transformer fine-tuning on the DBpedia 14-class Wikipedia benchmark.

Student: Srija Pentyala

Project video: https://youtu.be/tJMxr0M0WzE
Repository: https://github.com/srijapentyala/Next-Gen-NLP-Classifier-Using-Transformer-Models

---

## Start here

- Final notebook (curated): `main_notebook.ipynb` — open this first (recommended in Google Colab).
- Checkpoint notebooks: `checkpoints/checkpoint_1.ipynb`, `checkpoints/checkpoint_2.ipynb`.
- Environment: `requirements.txt` (exported from Colab).

---

## One-paragraph overview

This project answers a single practical question: when and why does reading words in context (transformer models) materially improve text classification compared to classical bag-of-words approaches? The investigation uses the DBpedia 14-class dataset (560k train examples) and evaluates three stages: RQ1 (TF-IDF + Logistic Regression), RQ2 (TF-IDF + TruncatedSVD + LogReg), and RQ3 (fine-tuned DistilBERT on an 8K subset and on the full dataset). The notebook contains EDA, modeling code, evaluation, explainability (LIME), and deployment guidance.

---

## Research questions

1. RQ1 — How far can TF-IDF + Logistic Regression go on DBpedia 14?  
2. RQ2 — Does dimensionality reduction (SVD) help or hurt performance?  
3. RQ3 — Does a contextual model (DistilBERT) outperform TF-IDF, and what sample sizes are needed?

---

## Dataset

- DBpedia 14 (HuggingFace: `dbpedia_14`) — 14-class balanced Wikipedia abstracts.  
- Approx. 560,000 training examples and 70,000 test examples (40k / 5k per class in typical splits).  
- Preprocessing is implemented in the notebook: combine title+abstract, remove missing rows, verify labels, create stratified train/validation splits, and build an 8K stratified subset for the sample-efficiency experiment.

Link: https://huggingface.co/datasets/dbpedia_14

---

## Key results (short)

- TF-IDF + Logistic Regression: ~98.4% test accuracy (fast, CPU-friendly).  
- TF-IDF + TruncatedSVD: ~96.5% (compression hurts for this dataset).  
- DistilBERT (8K): ~98.4% (matches TF-IDF using 56× fewer labels).  
- DistilBERT (full): ~98.7% (marginal gain at much higher compute cost).

Full tables, confusion matrices, and per-class analysis are in `main_notebook.ipynb`.

---

## How to reproduce (recommended: Colab)

1. Open `main_notebook.ipynb` in Google Colab.
2. Runtime → Change runtime type → GPU (T4 recommended for fine-tuning).  
3. Run cells top → bottom. The notebook installs required packages when run in Colab, downloads the dataset via HuggingFace, and runs RQ1–RQ3 experiments.

To export the full environment from Colab (run in a cell at the end):

```python
!pip freeze > requirements.txt
from google.colab import files
files.download('requirements.txt')
```

Local reproduction (optional):

```bash
python --version  # record your Python version (e.g., 3.11)
pip install -r requirements.txt
jupyter lab  # or jupyter notebook
```

Notes:
- RQ1/RQ2 (TF-IDF) run quickly on CPU.  
- RQ3 (DistilBERT fine-tune) requires GPU; run time depends on instance type and number of epochs.

---

## Key dependencies

See `requirements.txt` for the full environment exported from Colab. Core libraries used in the notebook:

- Python 3.11 (recommended in Colab)  
- torch, transformers (DistilBERT)  
- datasets (HuggingFace), scikit-learn  
- pandas, numpy  
- matplotlib, seaborn  
- lime (explainability)

---

### Key versions (top dependencies)

These are the main packages used in the notebook (from `requirements.txt`):

- torch==2.10.0+cu128
- transformers==5.0.0
- datasets==4.0.0
- scikit-learn==1.6.1
- pandas==2.2.2
- numpy==2.0.2
- matplotlib==3.10.0
- seaborn==0.13.2

---

## Quickstart (run locally)

1. Clone the repo and change directory:

```bash
git clone https://github.com/srijapentyala/Next-Gen-NLP-Classifier-Using-Transformer-Models.git
cd Next-Gen-NLP-Classifier-Using-Transformer-Models
```

2. (Optional) Create a Python virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open `main_notebook.ipynb` (Jupyter or Colab recommended) and run cells top→bottom.

---

## Data download

The notebook downloads DBpedia automatically via HuggingFace. If you prefer to use a mirror, you can download the dataset archive here:

- Direct mirror: https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k

Place any local CSV files under `dbpedia_csv/` if you want the notebook to pick them up instead of re-downloading.

---

## Reproducibility checklist (for reviewers)

1. Open `main_notebook.ipynb` in Colab.  
2. Select Runtime → Change runtime type → GPU (T4 recommended).  
3. Run cells top→bottom.  
4. Export `requirements.txt` from Colab if you want a bit-for-bit freeze and commit it to the repo.  

Notes: include your Python version in the README after exporting the freeze.

---

## Repository structure

```
Next-Gen-NLP-Classifier-Using-Transformer-Models/
├── main_notebook.ipynb        # Curated final notebook (start here)
├── requirements.txt           # Environment exported from Colab
├── README.md                  # This file
├── checkpoints/
│   ├── checkpoint_1.ipynb
│   └── checkpoint_2.ipynb
├── dbpedia_csv/               # (optional) local data cache
├── scripts/                   # data download / preprocessing helpers
├── assets/                    # figures and visual assets used in the notebook
└── delete_checkpoint.sh       # helper script
```

---

## What to include for submission (rubric alignment)

This README and the repository include the rubric-required elements:

- Project title and one-paragraph overview: included above.  
- Pointer to `main_notebook.ipynb`: provided under "Start here".  
- Research questions: listed in the Research questions section.  
- Project video link: at the top (YouTube).  
- Data section and preprocessing summary: provided in Dataset.  
- Reproducibility instructions and `requirements.txt`: provided in How to reproduce and Key dependencies.  
- Repo structure and files included: provided in Repository structure.  
- Results summary: included in Key results (short).  

Follow these and reviewers should be able to reproduce and evaluate the project end-to-end.

---

## Contact

If you have questions, open an issue in the repository or contact the author via the GitHub profile: https://github.com/srijapentyala

---

**Before submission**: verify the repository is public and that `main_notebook.ipynb` opens in an incognito window without authentication.
