# Feature Description - Model 2

## Problem Formulation

The task is formulated as a Multiple Choice Question Answering problem.

Each training example consists of:

- One prompt
- Five candidate options
- One correct answer

The model predicts the probability of each option being correct.

---

## Text Features

Unlike Model 1, no handcrafted TF-IDF features are used.

The transformer automatically learns contextual representations.

---

## Input Representation

Each sample is encoded as:

Prompt + Option A

Prompt + Option B

Prompt + Option C

Prompt + Option D

Prompt + Option E

The tokenizer converts each pair into:

- Input IDs
- Attention Masks

---

## Feature Extraction

Contextual embeddings are learned internally by DeBERTa through self-attention.

No manual feature engineering is required.

---

## Labels

A → 0

B → 1

C → 2

D → 3

E → 4

---

## Model

microsoft/deberta-v3-base

Task:

Multiple Choice Classification