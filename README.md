# LLM-Powered Fraud Detection with FAISS

🚀 **Fraud Detection Using FAISS & Transformer-Based Embeddings**  

---

## 📖 Project Overview
This project implements a **fraud detection system** using **LLM-based embeddings, FAISS for similarity search, and anomaly detection techniques**.  

✔ **Generates a synthetic transaction dataset** (100M+ records)  
✔ **Preprocesses & tokenizes descriptions using SentencePiece**  
✔ **Generates embeddings with Word2Vec, FastText & AutoEncoders**  
✔ **Indexes embeddings into FAISS** for **fast similarity search**  
✔ **Detects anomalies based on similarity scores to flag fraud**  

---

## 📌 Steps Implemented in Colab Notebook

### 1️⃣ Generate a Synthetic Transaction Dataset
Simulates **100M+ transactions** with:  
- `user_id`, `age`, `credit_score`, `amount`, `merchant`, `location`, `IP`, `device_fingerprint`, `fraud_flag`  
- **Fraud cases (2%) added to simulate real-world fraud patterns**  

```bash
python generate_data.py
```
**Output:** `synthetic_transactions.csv`

---

### 2️⃣ Preprocess & Tokenize Transaction Descriptions
✔ **Lowercases text, removes special characters & stopwords**  
✔ **Tokenizes using SentencePiece**  

```bash
python preprocess_data.py
```
**Output:** `cleaned_transactions.csv`

---

### 3️⃣ Generate Text-Based Embeddings (Word2Vec & FastText)
✔ **Trains Word2Vec & FastText on tokenized descriptions**  
✔ **Generates vector embeddings for each transaction**  

```bash
python train_embeddings.py
```
**Outputs:**  
- `word2vec.model` (Word2Vec embeddings)  
- `fasttext.model` (FastText embeddings)  
- `word2vec_transactions.csv` (Dataset with embeddings)

---

### 4️⃣ Generate Structured Embeddings Using AutoEncoders & PCA
✔ **Compresses high-dimensional transaction features**  
✔ **Reduces dimensions using PCA for anomaly detection**  

```bash
python train_autoencoder.py
```
**Outputs:**  
- `structured_embeddings.npy`  
- `pca_embeddings.npy`

---

### 5️⃣ FAISS Indexing & Anomaly Detection
✔ **Indexes embeddings in FAISS for fast similarity search**  
✔ **Detects anomalies using cosine similarity**  

```bash
python faiss_index.py
```
**Output:** `faiss_fraud_index.idx`

```bash
python detect_anomalies.py
```
**Output:** `flagged_anomalies.csv`

---

## 🔍 Next Steps
✅ **Fine-tune fraud detection thresholds**  
✅ **Visualize fraud patterns using t-SNE**  
✅ **Deploy real-time fraud detection**  

---

## 📌 Running the Entire Project in Colab
1️⃣ Open **Google Colab**  
2️⃣ Upload the notebook (`fraud_detection.ipynb`)  
3️⃣ Run all cells **step by step**

---

## 📌 Install Requirements
To install all required dependencies, run:  
```bash
pip install -r requirements.txt
```
