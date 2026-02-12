# Next-Gen-NLP-Classifier-Using-Transformer-Models

This project aims to build a **scalable and robust NLP text classification system** that leverages **state-of-the-art transformer models** (e.g., BERT, RoBERTa) to automatically categorize text into multiple classes with high accuracy and minimal manual feature engineering. Transformers model long-range dependencies and semantic context far better than traditional approaches like bag-of-words or TF-IDF.  

The current workflow performs **data loading, preprocessing, and exploratory data analysis (EDA)**, setting the stage for transformer-based fine-tuning in the next steps.

---

## 📌 Project Objectives

- **Preprocess text data** by combining titles with full text and cleaning out missing or empty entries.  
- **Explore the dataset** to understand class balance, text length distributions, and sample content per label.  
- **Visualize key characteristics** such as class distributions and word count histograms.  
- **Prepare the dataset** (train/validation split) for later use with transformer models.  
- **Set up a scalable NLP pipeline** that can later be extended with transformer models for classification.

---
---

## 📂 Dataset Details

**Dataset:** DBpedia Ontology Classification  
**Source:** https://www.google.com/url?q=https%3A%2F%2Fdrive.google.com%2Fuc%3Fexport%3Ddownload%26id%3D0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k
**Task:** Multi-class text classification  
**Classes:** 14 categories  
**Size:** ~560,000 training samples, ~70,000 test samples  
**Language:** English  

The DBpedia Ontology Classification dataset is constructed from structured Wikipedia entries. 
Each sample contains a title and a short abstract describing an entity, and the task is to classify the text into one of 14 ontology-based categories such as Company, Artist, Athlete, Film, or Natural Place.

The dataset represents real-world encyclopedic knowledge and is commonly used as a benchmark for large-scale multi-class text classification tasks.

### Key Characteristics
- Numerically balanced class distribution  
- Moderate text length suitable for transformer models  
- Public research dataset  


## 🛠️ Get Started

### Installation

```bash
git clone https://github.com/srijapentyala/Next-Gen-NLP-Classifier-Using-Transformer-Models
cd Next-Gen-NLP-Classifier-Using-Transformer-Models

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
``` 

## 🚀 Future Scope & Next Steps

- **Transformer Fine-Tuning:** Train transformer models (BERT, RoBERTa, DistilBERT) on this dataset for superior performance compared to traditional models.  
- **Hyperparameter Tuning:** Optimize learning rate, batch size, and number of epochs for best results.  
- **Long Text Strategies:** Implement truncation, sliding window, or hierarchical approaches for texts longer than model token limits.  
- **Evaluation Metrics:** Analyze model performance using accuracy, precision, recall, F1-score, and confusion matrices.  
- **Deployment:** Package the trained model as a REST API or web application for real-time inference.  
- **Benchmarking:** Compare performance against baseline TF-IDF + SVM models to quantify improvements.

