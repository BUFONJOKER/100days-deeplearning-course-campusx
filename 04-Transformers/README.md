
---


# 📝 Text Classification with Transformers (Keras)

This repository contains a Jupyter Notebook demonstrating **text classification using a Transformer encoder built from scratch with Keras**.  
The project focuses on understanding how Transformers work internally rather than using prebuilt models.

## 📓 Detailed Notebook Overview

### **36-text-classification.ipynb**

**Topic:** Text Classification using Transformer Encoder

**Overview:**
This comprehensive notebook demonstrates implementing a Transformer encoder from scratch for text classification tasks, specifically binary sentiment classification on the IMDB movie reviews dataset.

**Dataset:**
- **Source:** IMDB Movie Reviews Dataset
- **Size:** 50,000 reviews
- **Labels:** Binary (0 = Negative, 1 = Positive)
- **Challenge:** Classify sentiment from raw review text

**Key Concepts Implemented:**

1. **Text Preprocessing**
   - Tokenization and vocabulary building
   - Integer encoding of sequences
   - Padding to uniform length

2. **Positional Encoding**
   - Mathematical formulation of positional encodings
   - Sine and cosine encoding patterns
   - Adding position information to embeddings

3. **Multi-Head Self-Attention**
   - Query, Key, Value projections
   - Scaled dot-product attention
   - Multiple attention heads for diverse feature capture
   - Attention weight visualization

4. **Transformer Encoder Block**
   - Layer normalization
   - Residual connections (Add & Norm)
   - Feed-forward neural network (FFN)
   - Position-wise fully connected layers

5. **Complete Architecture**
   - Embedding layer with positional encoding
   - Multiple stacked Transformer encoder blocks
   - Global average pooling
   - Dense output layer for binary classification

**Implementation Details:**
- **Framework:** TensorFlow/Keras
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy
- **Architecture:** Encoder-only (no decoder)

**Learning Outcomes:**
- Understand Transformer architecture mechanics
- Implement custom Transformer blocks in Keras
- Build production-ready text classification models
- Gain intuition about self-attention mechanisms
- Learn positional encoding mathematics

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



