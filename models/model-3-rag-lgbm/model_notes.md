# Model 3

## Objective

Improve the baseline ranking model by augmenting each question with retrieved external context using a Retrieval-Augmented Generation (RAG) pipeline.

---

## Model Architecture

Hybrid Retrieval + Learning-to-Rank

- SentenceTransformer Retriever
- FAISS Vector Database
- CrossEncoder Re-ranker
- LightGBM Ranker

---

## Retrieval Pipeline

Knowledge Base

- Constructed from the training dataset by pairing prompts with their correct answers.

Retriever

- SentenceTransformer (all-MiniLM-L6-v2)

Vector Database

- FAISS IndexFlatIP

Re-ranking

- CrossEncoder (cross-encoder/ms-marco-MiniLM-L6-v2)

---

## Feature Engineering

### Text Features

- Word TF-IDF
- Character TF-IDF

### Similarity Features

- Cosine Similarity

### Engineered Features

- Prompt Length
- Option Length
- Word Overlap
- Overlap Ratio

### RAG Features

- Retrieved Context
- Context Length
- Context Similarity

---

## Final Model

LightGBM Ranker

---

## Validation Performance

Cross Validation MAP@3

| Fold | MAP@3 |
|------|-------|
| Fold 1 | 0.8965 |
| Fold 2 | 0.8975 |
| Fold 3 | 0.8965 |
| Fold 4 | 0.8975 |
| Fold 5 | 0.8975 |

Average MAP@3 ≈ 0.8971

---

## Observations

- Retrieved context provided additional semantic information for each question.
- The hybrid RAG + LightGBM approach significantly outperformed the classical TF-IDF baseline.
- Combining retrieval with engineered features proved more effective than using retrieval alone.

---

## Future Improvements

- Fine-tune the retrieval and ranking components jointly.
- Ensemble the hybrid RAG model with the DeBERTa transformer model.