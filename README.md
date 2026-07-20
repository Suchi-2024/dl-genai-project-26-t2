# Milestone 5 – Model Ensembling & Prediction Ranking

## Overview

This milestone focuses on improving multiple-choice question answering through model ensembling. The objective is to combine predictions from multiple transformer models, analyze confidence scores, compare individual and ensemble predictions, and evaluate the final performance using the MAP@3 metric.

## Objectives

- Load multiple fine-tuned transformer models.
- Perform inference using DeBERTa and RoBERTa.
- Apply Softmax to obtain class probabilities.
- Combine predictions using weighted probability averaging.
- Rank answer options based on ensemble probabilities.
- Compare individual model predictions with ensemble predictions.
- Evaluate the final ensemble using MAP@3.

## Topics Covered

- Transformer Model Inference
- Softmax Probability Calculation
- Weighted Probability Averaging
- Model Ensembling
- Top-1 Prediction Comparison
- Top-3 Prediction Ranking
- Confidence Score Analysis
- Prediction Agreement Analysis
- MAP@3 Evaluation

## Repository Contents

| File | Description |
|------|-------------|
| `Milestone-5-Solutions.ipynb` | Complete notebook containing all milestone questions, implementations, outputs, and observations. |

## Learning Outcomes

After completing this milestone, I gained practical experience in:

- Performing inference using multiple transformer models.
- Combining predictions through weighted ensembling.
- Ranking answer choices based on probability distributions.
- Comparing individual model predictions with ensemble predictions.
- Analyzing prediction confidence and agreement between models.
- Evaluating ensemble performance using the MAP@3 metric.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- NumPy
- Pandas
- Scikit-learn

## Dataset

Smart MCQ Solver Challenge Dataset (Kaggle)

## Status

**Completed**