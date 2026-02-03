
---

# 📘 CampusX – 100 Days of Deep Learning

## Repository Description

This repository is a comprehensive collection of **36 hands-on Jupyter notebooks** covering the complete curriculum of the **CampusX 100 Days of Deep Learning** course. It serves as both a learning resource and a practical implementation guide for understanding deep learning concepts from fundamentals to advanced architectures.

**Key Highlights:**
- 📚 **36 Progressive Notebooks** covering Artificial Neural Networks (ANN), Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and Transformers
- 🎯 **Structured Learning Path** starting from basic perceptrons to modern transformer architectures
- ☁️ **Colab-Ready** - All notebooks are optimized for execution in Google Colab without local GPU requirements
- 🔬 **Theory + Practice** - Each notebook combines mathematical concepts with practical TensorFlow/Keras implementations
- 📊 **Real-World Projects** - Includes hands-on applications like customer churn prediction, image classification, and sentiment analysis
- 📖 **Comprehensive Documentation** - Every notebook includes detailed explanations, visualizations, and learning outcomes

**Use Cases:**
- Students transitioning from machine learning to deep learning
- Practitioners seeking hands-on deep learning experience
- Anyone looking for a structured, project-based learning path in modern AI
- Researchers exploring different neural network architectures

---

## About This Repository

This repository contains **hands-on practice and implementations** from the **CampusX 100 Days of Deep Learning** course.
Every topic/chapter has its own **Jupyter Notebook (`.ipynb`)** which can be run directly in **Google Colab**.

---

## 🚀 Goal of This Repository

* 💻 Implement concepts taught in the **100 Days of Deep Learning** course
* 🧠 Strengthen understanding of **ML & DL architectures**
* 📊 Practice neural networks, CNNs, RNNs, GANs, and Transformers
* ☁️ Make all notebooks **Colab-ready** for easy execution
* 📚 Document all learnings in a structured, organized way


---

## 🧠 Topics Covered (Example)

### 🔹 **Perceptron & Perceptron Trick**

* Binary classification
* Weight updates
* TensorFlow implementation

### 🔹 **Activation Functions**

* Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax
* Usage in hidden & output layers

### 🔹 **Loss Functions**

* Regression: MSE, MAE, Huber
* Classification: Binary & Categorical Crossentropy, Hinge Loss
* GAN & VAE losses

### 🔹 **Neural Network Architectures**

* Feedforward / MLP
* CNNs for images
* RNN, LSTM, GRU for sequences
* Autoencoders
* GANs
* Transformers (BERT, GPT concepts)

### 🔹 **Model Training & Optimization**

* Forward & Backward propagation
* Optimizers: SGD, Adam, RMSprop
* Regularization: Dropout, BatchNorm, L2
* Early stopping & learning rate schedules

### 🔹 **Metrics & Evaluation**

* Accuracy, Precision, Recall, F1-score
* Confusion Matrix
* ROC-AUC

---

## 📁 Repository Structure

### **Root Directory**
```
100days-deeplearning-course-campusx/
├── main.ipynb                    # Main entry point notebook
├── README.md                     # Project documentation
├── 01-ANN/                       # Artificial Neural Networks
├── 02-CNN/                       # Convolutional Neural Networks
├── 03-RNN/                       # Recurrent Neural Networks
└── 04-Transformers/             # Transformer Models
```

### **01-ANN/ - Artificial Neural Networks (19 notebooks)**

Comprehensive coverage of foundational neural network concepts and techniques:

| Notebook | Description |
|----------|-------------|
| `01-perceptron-trick.ipynb` | Introduction to perceptron and the perceptron trick algorithm |
| `02-loss-function-perceptron.ipynb` | Understanding loss functions in perceptron learning |
| `03-problem-with-perceptron.ipynb` | Limitations of perceptron and why we need multi-layer networks |
| `04-customer-churn-prediction.ipynb` | Real-world ANN application for customer churn prediction |
| `05-handwrittendigits-classification.ipynb` | MNIST dataset classification using ANN |
| `06-graduate-admission-regression.ipynb` | Regression task using neural networks |
| `07-batch-vs-stochastic-gradient-descent.ipynb` | Comparison of batch and stochastic gradient descent |
| `08-vanishing-gradient-problem.ipynb` | Understanding vanishing gradient problem and solutions |
| `09-early-stopping.ipynb` | Preventing overfitting using early stopping |
| `10-feature-scaling.ipynb` | Importance of feature normalization and scaling |
| `11-dropout-layer-classification.ipynb` | Dropout regularization for classification tasks |
| `12-dropout-layer-regression.ipynb` | Dropout regularization for regression tasks |
| `13-regularization.ipynb` | L1 and L2 regularization techniques |
| `14-zero-weight-initialization.ipynb` | Issues with zero weight initialization |
| `15-zero-point5--weight-initialization.ipynb` | Small constant weight initialization |
| `16-xavier-normal-weight-initialization.ipynb` | Xavier/Glorot uniform weight initialization |
| `17-he-normal-he-uniform-glorot-uniform-normal-weight-initialization.ipynb` | He and Glorot initialization methods |
| `18-batch-normaliztion.ipynb` | Batch normalization for stable training |
| `19-exponentially-weighted-moving-average.ipynb` | EWMA and adaptive learning rate algorithms |

**Key Topics:** Perceptrons, MLPs, Loss Functions, Gradient Descent, Regularization, Weight Initialization, Batch Normalization, Optimization

---

### **02-CNN/ - Convolutional Neural Networks (12 notebooks)**

Deep dive into convolutional networks for image processing and computer vision:

| Notebook | Description |
|----------|-------------|
| `20-padding.ipynb` | Padding strategies in convolutions (same, valid) |
| `21-pooling.ipynb` | Max pooling and average pooling operations |
| `22-lenet-architecture.ipynb` | Classic LeNet architecture implementation |
| `23-dog-vs-cat-cnn-classification.ipynb` | CNN for binary image classification |
| `24-data-augmentation.ipynb` | Data augmentation techniques for improved generalization |
| `25-pretrained-models.ipynb` | Using pre-trained models (VGG, ResNet, etc.) |
| `26-transfer-learning-feature-extraction.ipynb` | Transfer learning using feature extraction |
| `27-transfer-learning-feature-extraction-data-augmentation.ipynb` | Combining transfer learning with data augmentation |
| `28-transfer-learning-fine-tunning.ipynb` | Fine-tuning pre-trained models |
| `29-keras-functional-api-single-input-multiple-output.ipynb` | Multi-output model architecture |
| `30-keras-functional-api-multiple-input-single-output.ipynb` | Multi-input model architecture |
| `31-keras-functional-model-transfer-learning.ipynb` | Transfer learning with functional API |

**Key Topics:** Convolutions, Pooling, LeNet, Padding, Data Augmentation, Transfer Learning, Fine-tuning, Functional API, Pre-trained Models

---

### **03-RNN/ - Recurrent Neural Networks (4 notebooks)**

Understanding sequential data processing with recurrent architectures:

| Notebook | Description |
|----------|-------------|
| `32-rnn-architecture.ipynb` | RNN fundamentals and architecture |
| `33-integer-encoding-simple-rnn.ipynb` | Integer encoding and basic RNN implementation |
| `34-sentimental-analysis-embedding.ipynb` | Sentiment analysis with word embeddings |
| `35-deep-rnn.ipynb` | Deep RNN architectures with multiple layers |

**Key Topics:** RNN Architecture, Sequence Processing, Word Embeddings, Sentiment Analysis, Deep RNNs, LSTM, GRU

---

### **04-Transformers/ - Transformer Models (1 notebook)**

Introduction to modern transformer architectures:

| Notebook | Description |
|----------|-------------|
| `36-text-classification.ipynb` | Text classification using transformer models |

**Key Topics:** Transformers, Attention Mechanism, BERT, GPT, Text Classification

---

## 🛠 Tech Stack

* **Python**
* **TensorFlow / Keras**
* **NumPy, Pandas**
* **Matplotlib, Seaborn**
* **Jupyter Notebook / Google Colab**

---


## 🙌 Acknowledgements

Thanks to **CampusX** for the **100 Days of Deep Learning** course!
This repository is meant for **practice, learning, and documenting** every topic from the course.

---

