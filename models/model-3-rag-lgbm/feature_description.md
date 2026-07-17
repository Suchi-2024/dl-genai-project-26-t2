# Feature Description - Model 3

## Problem Formulation

The MCQ task is formulated as a ranking problem.

For every prompt, external supporting context is retrieved before ranking the five answer options.

---

## Retrieval Features

Knowledge Base

Generated from the training dataset using prompt–answer pairs.

Retriever

SentenceTransformer (all-MiniLM-L6-v2)

Vector Search

FAISS IndexFlatIP

Re-ranking

CrossEncoder (ms-marco-MiniLM-L6-v2)

---

## Text Features

- Word TF-IDF
- Character TF-IDF

---

## Similarity Features

- Cosine Similarity
- Word Overlap
- Overlap Ratio

---

## Length Features

- Prompt Length
- Option Length

---

## Context Features

- Retrieved Context
- Context Length
- Context Similarity

---

## Final Feature Matrix

The final model combines:

- TF-IDF Features
- Character TF-IDF
- Cosine Similarity
- Engineered Numerical Features
- Retrieved RAG Context Features

---

## Model

LightGBM Ranker