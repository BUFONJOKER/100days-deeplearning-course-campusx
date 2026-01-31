
---


# 📝 Text Classification with Transformers (Keras)

This repository contains a Jupyter Notebook demonstrating **text classification using a Transformer encoder built from scratch with Keras**.  
The project focuses on understanding how Transformers work internally rather than using prebuilt models.

📌 **Notebook:** `36-text-classification.ipynb`

---

## 🚀 Project Overview

In this project, we:
- Implement the **Transformer Encoder** architecture from scratch
- Build **custom embedding layers with positional encoding**
- Train a Transformer model for **sentiment analysis**
- Use the **IMDB movie reviews dataset**
- Perform **binary text classification**

This notebook is ideal for learners who want to deeply understand **how Transformers work under the hood** using Keras.

---

## 🧠 Key Concepts Covered

- Text preprocessing & tokenization  
- Integer encoding and padding  
- Positional encoding  
- Multi-Head Self-Attention  
- Transformer encoder blocks  
- Feed Forward Neural Networks (FFN)  
- Binary classification with Transformers  

---

## 🏗️ Model Architecture

- **Embedding Layer**
- **Positional Encoding**
- **Transformer Encoder Block**
  - Multi-Head Attention
  - Add & Norm
  - Feed Forward Network
- **Global Pooling**
- **Dense Output Layer**

> 🔍 *Only the encoder part of the Transformer is implemented (no decoder).*

---

## 📊 Dataset

- **IMDB Movie Reviews Dataset**
- 50,000 reviews
- Binary labels:
  - `1` → Positive  
  - `0` → Negative  

---

## 🛠️ Tech Stack

- Python 🐍  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Jupyter Notebook  

---

## 📈 Training & Evaluation

- Binary Crossentropy loss
- Adam optimizer
- Accuracy used as evaluation metric
- Validation performed during training

---

## 🎯 Learning Outcome

After completing this notebook, you will:
- Understand how Transformers process text
- Be able to implement a Transformer encoder from scratch
- Know how positional encoding works mathematically
- Gain confidence building custom deep learning layers in Keras

---



## 📌 Notes

* This project is **educational**, not optimized for production
* No pretrained Transformer models are used
* Focus is on **clarity and understanding**, not SOTA performance

---

## 🤝 Contributing

Feel free to fork the repo, improve the model, or add:

* Visualization of attention weights
* Model performance comparison
* Multi-class classification

---

## ⭐ Acknowledgment

Inspired by the original Transformer paper:
**“Attention Is All You Need” – Vaswani et al.**

---



