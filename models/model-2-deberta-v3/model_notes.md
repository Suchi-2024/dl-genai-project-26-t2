# Model 2

## Objective

Develop a transformer-based Multiple Choice Question Answering model using DeBERTa-v3 to improve contextual understanding over the classical TF-IDF baseline.

---

## Model Architecture

- Model: microsoft/deberta-v3-base
- Task: Multiple Choice Classification
- Framework: Hugging Face Transformers

---

## Data Preparation

- Cleaned prompt and option texts
- Encoded answer labels (A-E → 0-4)
- Train-validation split (80:20)
- Converted dataset to Hugging Face Dataset format

---

## Tokenization

Tokenizer:
- DeBERTa-v3 Tokenizer

Input Format:

Prompt

paired with

Option A
Option B
Option C
Option D
Option E

Maximum sequence length:
- 256

---

## Training Configuration

- Learning Rate: 2e-5
- Epochs: 4
- Batch Size: 4
- Gradient Accumulation: 2
- Weight Decay: 0.01

Optimizer:
- AdamW (Trainer default)

---

## Validation Performance

| Epoch | Validation Accuracy |
|--------|---------------------|
| 1 | 0.6075 |
| 2 | 0.8150 |
| 3 | 0.8900 |
| 4 | 0.8975 |

---

## Saved Artifacts

- Fine-tuned DeBERTa model
- Tokenizer
- Prediction logits
- Kaggle submission file

---

## Observations

- Transformer models capture contextual relationships better than sparse TF-IDF features.
- Validation accuracy improved steadily across epochs.
- This serves as the first deep learning baseline for the project.

---

## Future Improvements

- Fine-tune the model using LoRA for better task adaptation.
- Combine transformer and ranking models through ensembling to improve MAP@3.