# RAG System for Scientific NLP (Alzheimer’s Disease)

## Project Overview

This is an end-to-end Retrieval-Augmented Generation (RAG) system for answering scientific NLP questions using biomedical literature (PubMed).
It showcases hands-on work with LLMs, NLP pipelines, embeddings, vector search, and evaluation-ready ML architecture.

Designed as a research-oriented ML prototype, it mirrors real-world scientific NLP workflows.

## Architecture

### 1. Retrieval

* Uses `sentence-transformers/all-MiniLM-L6-v2`
* Embedding computation with PyTorch backend
* FAISS vector index (inner product and normalization)
* Retrieves relevant scientific text chunks with metadata (PMID, title)

### 2. Generation

* LLM accessed via OpenRouter API
* Prompt engineering for scientific-style answers
* Explicit restriction to retrieved context (no hallucinations)
* Source citation support

### 3. Pipeline

* Modular pipeline combining retrieval and generation
* Easily extensible for benchmarking, evaluation, or model replacement

### 4. UI & Visualization

* Streamlit app for interactive querying
* Displays answers and cited sources
* Useful for qualitative analysis and demo purposes

## Technologies Used

* **PyTorch, HuggingFace (sentence-transformers)**
* **FAISS for vector search**
* **Requests for API calls**
* **Streamlit**
* **LLM APIs (OpenRouter)**

## How to run 

pip install -r requirements.txt

In app/main.py:
OPENROUTER_API_KEY = "your_openrouter_api_key"

streamlit run app/main.py
