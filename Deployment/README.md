---
title: Smart MCQ Solver
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# Smart MCQ Solver — Model 3 (Enhanced RAG + LightGBM Ranker)

Retrieval-augmented ranking pipeline for 5-option multiple-choice questions.
Built for the `smart-mcq-solver-challenge` Kaggle competition (BSDA2001P).

**Pipeline:** MiniLM bi-encoder retrieval (top-20) → MS-MARCO cross-encoder
rerank → TF-IDF (word + char) + numeric features → LightGBM ranker.

Public leaderboard MAP@3: **0.76059** (best of 4 models evaluated; outperformed
a handcrafted-TF-IDF-only baseline and a fine-tuned DeBERTa-v3 transformer —
see the full project report for the comparative analysis).
