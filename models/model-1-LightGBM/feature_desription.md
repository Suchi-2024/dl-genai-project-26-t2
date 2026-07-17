# Feature Description - Model 1

## Problem Formulation

The multiple-choice question answering task is converted into a ranking problem.

Each question is expanded into five (question, option) pairs corresponding to options A, B, C, D, and E.

The model predicts a relevance score for each option, and the top three options are selected for submission.

---

## Text Features

### 1. Word TF-IDF Features

TF-IDF vectors are generated using:

Prompt + Option

Example:

```text
Prompt [SEP] Option
```

Purpose:

* Capture important keywords
* Identify lexical overlap between prompt and option
* Provide sparse text representation

---

### 2. Character TF-IDF Features

Character n-gram TF-IDF representations are generated.

Purpose:

* Capture partial word matches
* Handle spelling variations
* Improve robustness for unseen words

---

## Similarity Features

### 3. Cosine Similarity

Cosine similarity is computed between:

* Prompt TF-IDF vector
* Option TF-IDF vector

Purpose:

* Measure semantic closeness
* Provide an explicit relevance score

---

## Engineered Features

### 4. Text Length Features

Features extracted:

* Prompt length
* Option length
* Word count
* Character count

Purpose:

* Capture structural information

---

### 5. Length Difference Features

Features extracted:

* Difference between prompt and option lengths

Purpose:

* Identify unusually short or long candidate answers

---

## Final Feature Set

The final feature matrix consists of:

1. Word TF-IDF Features
2. Character TF-IDF Features
3. Cosine Similarity Features
4. Length-Based Features
5. Additional Engineered Features

---

## Model

LightGBM Ranker

---

## Performance

MAP@3 ≈ 0.755
