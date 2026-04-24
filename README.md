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
- **Format:** CSV with columns `[label, title, content]`

**Preprocessing steps:**
1. Combine `title` + `content` into a single `text` field
2. Drop rows with missing or empty text
3. Map numeric labels (1–14) to human-readable class names
4. 80/20 stratified train/validation split for classical models
5. Separate 8K stratified subsample for DistilBERT fine-tuning

The notebook auto-downloads the dataset from HuggingFace on first run — no manual download needed.

---

## 🚀 How to Reproduce

This project was built and tested entirely in **Google Colab** with a T4 GPU.

**Steps:**

1. Open [`main_notebook.ipynb`](main_notebook.ipynb) in [Google Colab](https://colab.research.google.com/)
2. Go to **Runtime → Change runtime type → T4 GPU** (required for Stage 3 / DistilBERT)
3. Run **all cells from top to bottom** — the notebook will:
   - Auto-install dependencies
   - Auto-download the DBpedia 14 dataset from HuggingFace
   - Execute all three stages sequentially
4. For Stages 1 & 2 (TF-IDF), CPU is sufficient. Stage 3 requires GPU.

**Install dependencies locally (optional):**
```bash
pip install -r requirements.txt
```

> ⚠️ Stages 1 & 2 take ~2–3 minutes on CPU. Stage 3 (DistilBERT) takes ~30 minutes on a T4 GPU.

---

## 🔑 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.11 | Runtime |
| torch | 2.3.0+cu121 | DistilBERT training |
| transformers | 4.41.2 | DistilBERT model & tokenizer |
| datasets | 2.19.1 | DBpedia 14 dataset loading |
| scikit-learn | 1.4.2 | TF-IDF, LogReg, SVD, metrics |
| pandas | 2.0.3 | Data manipulation |
| numpy | 1.25.2 | Numerical operations |
| lime | 0.2.0.1 | Model explainability |
| matplotlib | 3.7.1 | Visualization |
| seaborn | 0.13.2 | Statistical plots |

Full environment: [`requirements.txt`](requirements.txt)

---

## 📓 Checkpoint Notebooks

The `checkpoints/` folder preserves the progression of the project throughout the semester:

- **`checkpoint_1.ipynb`** — Initial dataset exploration across DBpedia, AG News, and Amazon Reviews; early EDA; baseline model sketches
- **`checkpoint_2.ipynb`** — Expanded research questions; preliminary TF-IDF and SVD experiments; dataset narrowing to DBpedia 14

---

## 📚 References

1. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *arXiv:1810.04805*
2. Sanh et al. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108*
3. Ribeiro et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD 2016*
4. Auer et al. (2007). DBpedia: A Nucleus for a Web of Open Data. *ISWC 2007*
5. Zhang, Zhao & LeCun (2015). Character-level Convolutional Networks for Text Classification. *NeurIPS 28*
