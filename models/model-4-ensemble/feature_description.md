# Feature Description – Model 4 (Weighted Ensemble)

Unlike previous models, the ensemble does not create new textual features.

Instead, it combines the prediction probabilities produced by earlier models.

## Inputs

### Model 1
- TF-IDF features
- Character TF-IDF
- Cosine similarity
- Engineered numerical features

### Model 2
- DeBERTa contextual embeddings
- Transformer logits

### Model 3
- Retrieved contextual information
- TF-IDF + RAG features
- Learning-to-Rank scores

## Ensemble Features

The ensemble uses:

- Probability distributions from Model 1
- Probability distributions from Model 2
- Probability distributions from Model 3

These are combined using weighted averaging:

- Model 1 → 35%
- Model 2 → 20%
- Model 3 → 45%

The resulting probability vector is sorted to generate the Top-3 predictions for each question.