#  Spam Classification Using CNN

##  Overview
This project focuses on detecting **spam messages in Email data** using **Deep Learning**.  
The goal is to automatically classify messages as *spam* or *ham* (non-spam) based on their textual content.

After experimenting with traditional preprocessing and tokenization methods, I applied a **1D Convolutional Neural Network (Conv1D CNN)** to learn meaningful patterns in the text data.  
The model successfully captures contextual word relationships and achieves **high accuracy in spam detection**.

---

##  How It Works

###  1. Data Preprocessing
- Unnecessary columns were dropped, and labels were mapped to:
  - `0` → ham (non-spam)
  - `1` → spam  
- **Text cleaning**:
  - Tokenization with **NLTK**
  - Removal of English **stopwords**
  - Conversion of text into numeric **sequences** using Keras `Tokenizer`
  - Padding sequences to equal lengths with `pad_sequences`

###  2. Handling Class Imbalance
- The dataset was imbalanced (more *ham* than *spam* messages).
- Used **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic spam samples, balancing the dataset before training.

###  3. CNN Model Architecture
The model is built using **Keras Sequential API**:
1. **Embedding Layer** – converts word indices into dense vector representations.  
2. **Conv1D Layers** – slide multiple filters (kernels) across sequences to extract n-gram features and local patterns.  
3. **MaxPooling & GlobalMaxPooling** – reduce dimensionality and focus on the most important features.  
4. **Batch Normalization** – stabilizes learning and accelerates convergence.  
5. **Dropout** – prevents overfitting by randomly dropping neurons during training.  
6. **Dense Layers** – learn higher-level representations.  
7. **Sigmoid Output Layer** – outputs probability between 0 (ham) and 1 (spam).

The **ReLU activation function** was used throughout hidden layers for efficiency and non-linearity.  
The **Adam optimizer** combined the benefits of momentum and adaptive learning rate to speed up convergence.

---

##  Technologies Used
| Library / Tool | Purpose |
|----------------|----------|
| **Pandas** | Handle and clean tabular data |
| **NumPy** | Matrix and numerical operations |
| **Matplotlib** | Data visualization (accuracy/loss graphs) |
| **NLTK** | Text tokenization and stopword removal |
| **SMOTE (imblearn)** | Handle class imbalance |
| **TensorFlow / Keras** | Deep Learning framework for building CNN |
| **Scikit-learn** | Train-test split and utility functions |

---

##  Results Visualization
![Loss and Accuracy per Epoch](https://github.com/minhhai1208/spamEmailDetection/blob/main/Loss%20and%20Accuracy%20per%20epoch.png)
Training and validation performance were visualized using Matplotlib:

- **Left:** Training vs. Validation Accuracy  
- **Right:** Training vs. Validation Loss  

The plots clearly show stable learning and no major overfitting issues.

---

##  Key Takeaways
- Deep Learning (CNN) can handle **text classification** tasks effectively — even though CNNs are more common in image tasks.  
- Combining **tokenization, embeddings, and convolution** layers allows the model to extract **contextual patterns** from sequences.  
- **SMOTE** plays an important role in ensuring balanced learning.  
- **Dropout** and **Batch Normalization** greatly improve generalization.

---
