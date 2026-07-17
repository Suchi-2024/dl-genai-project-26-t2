# Model 1

## Objective
Build a ranking model for the Smart MCQ Solver Challenge.

## Features
- Word TF-IDF
- Character TF-IDF
- Cosine Similarity
- Additional Engineered Features

## Model
LightGBM Ranker

## Performance
MAP@3 ≈ 0.755

## Experiments Performed
1. TF-IDF baseline
2. Added character TF-IDF
3. Added cosine similarity features
4. Added engineered features
5. Switched to LightGBM Ranker

## Key Learnings
- Character features improved robustness.
- Cosine similarity provided useful semantic signals.
- Learning-to-rank formulation worked better than simple classification.