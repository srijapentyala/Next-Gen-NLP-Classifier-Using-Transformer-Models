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

## 🏆 Key Results

| Model | Test Accuracy | Test Macro F1 | Training Samples | Key Finding |
|-------|:---:|:---:|:---:|---|
| Majority-class baseline | 10.0% | 10.0% | — | Absolute floor |
| **RQ1: TF-IDF + LR** | **98.4%** | **98.4%** | ~448,000 | Strong, fast — context-blind |
| RQ2: TF-IDF + SVD | 96.5% | 96.5% | ~448,000 | Compression makes things *worse* |
| **RQ3: DistilBERT (8K)** | **98.4%** | **98.4%** | **8,000** | Matches Stage 1 with **56× less data** |
| RQ3: DistilBERT (Full) | **98.7%** | **98.7%** | ~448,000 | Transfer learning slight edge at scale |

**Central Finding:** On clean, balanced Wikipedia text, classical TF-IDF is *competitive* with modern transformers in raw accuracy. But DistilBERT achieves that same accuracy with **56× less labeled data** — revealing where transfer learning truly earns its GPU cost.

---

## 📁 Repository Structure

```
Next-Gen-NLP-Classifier-Using-Transformer-Models/
│
├── main_notebook.ipynb          # 👈 Final deliverable — start here
├── requirements.txt             # Full environment from Colab
├── README.md                    # You are here
│
├── checkpoints/
│   ├── checkpoint_1.ipynb       # Checkpoint 1: Initial exploration
│   └── checkpoint_2.ipynb       # Checkpoint 2: Expanded experiments
│
└── data/
    └── README_data.md           # Instructions for downloading DBpedia 14
```

---

## 📦 Data

**Dataset:** [DBpedia 14](https://huggingface.co/datasets/dbpedia_14) — 14-class Wikipedia ontology classification benchmark

- **Train:** 560,000 articles | **Test:** 70,000 articles
- **Classes (14):** Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, WrittenWork, Film
- **Perfectly balanced** — 40,000 train / 5,000 test samples per class
# Does Context Actually Matter for Text Classification?
A focused, reproducible comparison of classical bag-of-words pipelines versus transformer-based fine-tuning on the DBpedia 14-class benchmark.

Student: Srija Pentyala

Project video: https://www.youtube.com/watch?v=tJMxr0M0WzE&t=1s

Dataset mirror (direct download): https://drive.usercontent.google.com/download?id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k&export=download&authuser=1

---

## Start here

- Final notebook: `main_notebook.ipynb` — open this first (Colab recommended)
- Checkpoint notebooks: `checkpoints/checkpoint_1.ipynb`, `checkpoints/checkpoint_2.ipynb`

## Project summary

This project asks a single practical question: when and why does context (as provided by modern transformer models) matter for large-scale text classification? I evaluate three stages:

- RQ1 — TF-IDF + Logistic Regression (classical baseline)
- RQ2 — TF-IDF + Truncated SVD + Logistic Regression (compressed features)
- RQ3 — Fine-tuned DistilBERT (contextual model), both on a small 8k subset and on the full dataset

The deliverable is a curated, well-documented notebook (`main_notebook.ipynb`) that reproduces the experiments and contains the results, visualizations, and a short explainability section using LIME.

## Research questions

1. How far can TF-IDF + Logistic Regression go on DBpedia 14?  
2. Does dimensionality reduction (SVD) help or hurt classical models?  
3. Does a contextual model (DistilBERT) outperform TF-IDF, and if so, at what sample sizes?

## Key results (short)

- TF-IDF + Logistic Regression achieves strong accuracy on DBpedia (near 98% test accuracy).  
- Truncated SVD (topic compression) reduces performance slightly.  
- DistilBERT matches classical accuracy using a small labeled set (~8k samples), demonstrating sample-efficiency benefits of transfer learning.  

See `main_notebook.ipynb` for detailed tables and per-class breakdowns.

## Files and repository layout

```
Next-Gen-NLP-Classifier-Using-Transformer-Models/
├── main_notebook.ipynb        # Curated final notebook (start here)
├── requirements.txt           # Exported environment (from Colab)
├── README.md                  # This file
├── checkpoints/
│   ├── checkpoint_1.ipynb
│   └── checkpoint_2.ipynb
├── dbpedia_csv/               # (optional) local data store
├── scripts/                   # helper scripts (data download / preprocessing)
├── assets/                    # figures and visual assets used in the notebook
└── delete_checkpoint.sh       # helper script created during repo organization
```

## Data

- Primary dataset: DBpedia 14-class ontology (560k train / 70k test).  
- Official HuggingFace dataset: https://huggingface.co/datasets/dbpedia_14  
- Direct mirror (provided): https://drive.usercontent.google.com/download?id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k&export=download&authuser=1

Preprocessing highlights (implemented in the notebooks):

1. Combine title + content into a single `text` field.  
2. Remove empty/missing entries.  
3. Map and verify 14 class labels.  
4. Produce stratified train/validation splits and an 8k stratified subset used for the sample-efficiency experiment.

## How to reproduce (Colab recommended)

1. Open `main_notebook.ipynb` in Google Colab.  
2. Runtime → Change runtime type → GPU (T4 recommended for DistilBERT).  
3. Run all cells (top → bottom). The notebook will install required packages, download datasets, and run experiments.  

To export the environment from Colab (recommended step in the notebook):

```python
!pip freeze > requirements.txt
from google.colab import files
files.download('requirements.txt')
```

Local install (optional):

```bash
python --version  # note the version (e.g., Python 3.11)
pip install -r requirements.txt
```

Notes:
- Stage 1 & 2 (TF-IDF) run quickly on CPU.  
- Stage 3 (DistilBERT fine-tuning) requires a GPU and takes tens of minutes depending on instance type.

## Key dependencies

Listed in `requirements.txt`. Example core libs used in the notebooks:

- Python 3.11  
- torch, transformers  
- datasets, scikit-learn  
- pandas, numpy  
- matplotlib, seaborn  
- lime (explainability)

## Results and figures

The notebook contains:  
- A results table comparing accuracy / macro-F1 across methods.  
- Per-class confusion matrices and F1 breakdowns.  
- A sample-efficiency analysis (8k vs full dataset) with time/accuracy trade-offs.  

Highlight: DistilBERT reached comparable accuracy to TF-IDF using only a small fraction of labeled data, emphasizing transfer learning's sample-efficiency.

## Video presentation

Watch a short walk-through of the project: https://www.youtube.com/watch?v=tJMxr0M0WzE&t=1s

## Reproducibility & notes

- The notebooks are curated and documented: please follow the annotated markdown cells inside `main_notebook.ipynb`.  
- If you run locally, export and use the same `requirements.txt` generated from Colab.  
- If you have a GPU and want to re-run DistilBERT experiments faster, use a Colab Pro / GCP / AWS instance with a T4 / A100.

## Credits & references

- DistilBERT: Sanh et al., 2019  
- BERT: Devlin et al., 2019  
- LIME: Ribeiro et al., 2016  

## Contact

If you spot issues or want to discuss the project, open an issue on the repo or reach out to the author via the GitHub profile: https://github.com/srijapentyala

---

**Before submission**: Verify the repository is public and loads in an incognito browser window. Point reviewers to `main_notebook.ipynb` as the starting point.
