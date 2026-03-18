# 🧠 Document AI System (Offline-Capable)

## 📌 Overview

This project is an **offline-capable AI system** built using FastAPI, Gradio, and Docker. It provides multiple AI services including OCR, document-based question answering (RAG), information extraction , text classification, and general LLM interaction.

The system is designed to meet the requirements of the AI Academy Final Assignment, focusing on:

* Offline AI inference
* Modular architecture
* Practical deployment (Docker + Cloud demo)

---

## 🚀 Features

### 1. OCR (Optical Character Recognition)

* Extracts text from images and PDFs
* Supports multiple languages

### 2. RAG (Retrieval-Augmented Generation)

* Upload PDF documents
* Ask questions based on document content
* Uses vector database for retrieval

### 3. Information Extraction

* Extract structured fields from documents
* Customizable fields (e.g., name, date, amount)

### 4. ML Prediction

* Spam/Ham text classification
* Uses a trained ML model

### 5. Local LLM (TinyLlama)

* General chat capability
* Fully offline using Ollama (primary)
* Transformers fallback for cloud demo

---

## 🧱 Architecture

Gradio UI → FastAPI → Service Layer → Models

* OCR → Tesseract
* RAG → FAISS / embeddings
* LLM → Ollama (offline) / Transformers (fallback)
* ML → Scikit-learn model

---

## ⚙️ Installation (Local)

### 1. Clone repo

```bash
git clone https://github.com/laithdabbas/LaithDabbas_API
cd LaithDabbas_API
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Ollama (required for offline LLM)

```bash
ollama pull tinyllama
ollama run tinyllama
or
ollama run tinyllama 
```

### 4. Run the app

```bash
python start_server.py
```

---

## 🐳 Docker Usage

```bash
docker compose up --build -d
```

---

## 🌐 Cloud Demo (Hugging Face Space)

Due to platform limitations (no background services), Ollama cannot run in Hugging Face Spaces.

👉 The system automatically switches to a Transformers-based LLM for demo purposes.

---


## 📦 Deliverables

See `/deliverables` folder for:

* Setup guide
* Packaged executable
* Documentation

---

## 🧠 Engineering Decisions

* Ollama used for **offline LLM compliance**
* Transformers fallback for **cloud deployment**
* Modular service layer for maintainability
* RAG implemented for document understanding
* Docker used for reproducibility

---

## 📌 Notes

* Fully offline system supported locally
* Designed with scalability and maintainability in mind

---

## 👤 Author

Laith Dabbas
