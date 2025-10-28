# 🔡 Seq2Seq — Hindi Transliteration Model

A deep learning project implementing a **Sequence-to-Sequence (Seq2Seq)** model with **attention mechanism** for transliterating **romanized Hindi text to Devanagari script**.  
Built using **PyTorch** and trained on the **Aksharantar dataset**.

---

## 🎯 Overview

This project tackles the problem of **character-level transliteration** from romanized Hindi (Latin script) to Devanagari script using a neural sequence-to-sequence model.  
Unlike traditional rule-based systems, this model **learns the mapping directly from data** using:

- 🧩 **Encoder–Decoder Architecture** with LSTM cells  
- 🎯 **Bahdanau Attention Mechanism** for better alignment  
- 🔄 **Bidirectional Encoder** for capturing context from both directions  
- 🧮 **Beam Search Decoding** for improved inference quality  

---

## ✨ Features

- 🧠 3-Layer Deep LSTM Encoder and Decoder  
- 🎯 Bahdanau Attention Mechanism for better character alignment  
- 🔄 Bidirectional Encoder for enhanced context understanding  

---

## 🚀 Quick Start (Google Colab Recommended)

### Step 1: Open Google Colab
- Go to [colab.research.google.com](https://colab.research.google.com)  
- Create a **new notebook**

### Step 2: Enable GPU
- Click **Runtime → Change runtime type**  
- Select **GPU** under *Hardware accelerator*  
- Click **Save**

### Step 3: Download Dataset Files
From this repository, download:
- `hin_train.csv`  
- `hin_valid.csv`  
- `hin_test.csv` *(optional)*  

### Step 4: Upload Files to Colab
- Click the 📁 **Files** icon in the left sidebar  
- Click 📤 **Upload**  
- Upload all three CSV files

### Step 5: Copy the Code
- Open `transliteration_model.py` from this repo  
- Copy the entire code  
- Paste it into a new code cell in Colab

### Step 6: Run the Code
That’s it! 🎉  
No installation, no setup — just upload files and run.

---

## 📊 Dataset

This project uses the **Aksharantar dataset** (Hindi transliteration subset) from [AI4Bharat](https://ai4bharat.org/).

| File | Samples | Description |
|------|----------|-------------|
| `hin_train.csv` | ~90,000 | Training data |
| `hin_valid.csv` | ~10,000 | Validation data |
| `hin_test.csv`  | ~10,000 | Test data |

**Data Format:**  
ghar,घर
dost,दोस्त
namaste,नमस्ते
kitaab,किताब



> ⚠️ Ensure all three CSV files are in the **same directory** as your code/notebook.

---

## 📈 Model Performance

**Training Results**
- 🧩 Word Accuracy: **55%** ✨  
- 🔤 Character Accuracy: **70%+** 🔥  
- 📉 Training Loss: ~0.8 – 1.2  
- 📉 Validation Loss: ~1.0 – 1.5  
- 🧠 Test Loss: ~1.2 – 1.8  
- ⚡ Training Time: ~60 – 70 minutes on Colab T4 GPU (40 epochs)

**Performance Highlights**
- 💬 55% **Word-Level Accuracy** — more than half the words are perfectly transliterated!  
- 🔠 70%+ **Character-Level Accuracy** — most characters are correct even when the full word isn’t.  

This is **strong performance** for a character-level Seq2Seq task with a complex script like Devanagari.

---

## 🧪 Sample Predictions

<img width="589" height="313" alt="Seq2Seq transliteration sample" src="https://github.com/user-attachments/assets/b3859fc9-7e3f-4f47-8221-0a2b27ae4dc2" />

---

💡 *Optimized for Colab — run instantly, no setup required!*

Each CSV contains pairs of romanized Hindi and Devanagari text:  

