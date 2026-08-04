"""
Smart MCQ Solver — Model 3 (Enhanced RAG + LightGBM Ranker)
Streamlit app for Streamlit Cloud.

Pipeline (mirrors the notebook exactly):
  1. Clean the prompt
  2. Retrieve top-20 candidate passages (BGE-base bi-encoder + FAISS)
  3. Rerank with a cross-encoder, keep the single best passage
  4. Build context+prompt+option text for each of the 5 options
  5. TF-IDF (word + char) + 2 numeric features -> LightGBM ranker
  6. Rank the 5 options, show the top-3
"""

import json
import pickle

import faiss
import numpy as np
import streamlit as st
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
TOP_K = 20
# ARTIFACT_DIR = "artifacts"
BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
RETRIEVER_NAME = "BAAI/bge-base-en-v1.5"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

REMOVE_PHRASES = [
    "Pick the best possible answer:",
    "Determine the correct option:",
    "Choose the correct answer:",
    "Select the most accurate option:",
    "Identify the correct statement:",
]


def clean_prompt(text: str) -> str:
    for phrase in REMOVE_PHRASES:
        text = text.replace(phrase, "")
    return text.strip()


def overlap(a: str, b: str) -> int:
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    return len(a_words & b_words)


# ----------------------------------------------------------------
# Load everything once, cache across requests
# ----------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models and knowledge base (first request only)...")
def load_pipeline():
    with open(ARTIFACT_DIR/"knowledge.json") as f:
        knowledge = json.load(f)

    index = faiss.read_index(f"{ARTIFACT_DIR}/kb_index.faiss")

    with open(ARTIFACT_DIR/"word_vectorizer.pkl", "rb") as f:
        word_vec = pickle.load(f)
    with open(ARTIFACT_DIR/"char_vectorizer.pkl", "rb") as f:
        char_vec = pickle.load(f)
    with open(ARTIFACT_DIR/"lgbm_ranker.pkl", "rb") as f:
        model = pickle.load(f)

    retriever = SentenceTransformer(RETRIEVER_NAME)
    reranker = CrossEncoder(CROSS_ENCODER_NAME)

    return {
        "knowledge": knowledge,
        "index": index,
        "word_vec": word_vec,
        "char_vec": char_vec,
        "model": model,
        "retriever": retriever,
        "reranker": reranker,
    }


def retrieve_docs(question: str, pipeline: dict, k: int = TOP_K):
    # emb = pipeline["retriever"].encode([clean_prompt(question)], normalize_embeddings=True)
    # _, ids = pipeline["index"].search(emb, k)
    emb = pipeline["retriever"].encode(
        [clean_prompt(question)],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    if emb.shape[1] != pipeline["index"].d:
        raise ValueError(
            f"Embedding dimension {emb.shape[1]} "
            f"does not match FAISS index dimension {pipeline['index'].d}"
        )

    _, ids = pipeline["index"].search(emb, k)
    return [pipeline["knowledge"][i] for i in ids[0]]


def get_best_context(question: str, pipeline: dict) -> str:
    docs = retrieve_docs(question, pipeline)
    pairs = [[clean_prompt(question), doc] for doc in docs]
    scores = pipeline["reranker"].predict(pairs)
    return docs[int(np.argmax(scores))]


def predict(prompt: str, options: dict, pipeline: dict):
    """
    options: dict like {"A": "...", "B": "...", ...}
    """

    # Retrieve best context
    context = get_best_context(prompt, pipeline)

    labels = list(options.keys())

    # Build text exactly as during training
    texts = [
        f"{context} [SEP] {prompt} [SEP] {options[label]}"
        for label in labels
    ]

    # TF-IDF features
    X_word = pipeline["word_vec"].transform(texts)
    X_char = pipeline["char_vec"].transform(texts)

    # ---------------------------------------------------------
    # Numeric Features (must match notebook exactly)
    # ---------------------------------------------------------

    # Encode prompt ONCE
    prompt_emb = pipeline["retriever"].encode(
        [clean_prompt(prompt)],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)[0]

    num_feats = []

    for label in labels:

        option_text = options[label]

        # Encode option
        option_emb = pipeline["retriever"].encode(
            [option_text],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)[0]

        # Same as notebook:
        # (prompt_emb * option_emb).sum(axis=1)
        semantic_similarity = float(np.dot(prompt_emb, option_emb))

        num_feats.append([
            overlap(context, option_text),   # context_overlap
            len(option_text),                # option_len
            semantic_similarity              # semantic_similarity
        ])

    X_num = csr_matrix(np.array(num_feats, dtype=float))

    # ---------------------------------------------------------
    # Final feature matrix
    # ---------------------------------------------------------

    X = hstack([X_word, X_char, X_num]).tocsr()

    # Uncomment for debugging
    # st.write("Input features:", X.shape[1])
    # st.write("Model expects:", pipeline["model"].n_features_)

    scores = pipeline["model"].predict(X)

    ranked = sorted(
        zip(labels, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked, context

# ----------------------------------------------------------------
# UI
# ----------------------------------------------------------------
st.set_page_config(page_title="Smart MCQ Solver", page_icon="🧠", layout="centered")
st.title("🧠 Smart MCQ Solver")
st.caption("Model 3 (Enhanced) — Retrieval-Augmented Ranking. Public leaderboard MAP@3 = 0.76142")

pipeline = load_pipeline()

with st.form("mcq_form"):
    prompt = st.text_area("Question", placeholder="e.g. What is the primary source of energy in stars?", height=80)
    col1, col2 = st.columns(2)
    with col1:
        opt_a = st.text_input("Option A")
        opt_b = st.text_input("Option B")
        opt_c = st.text_input("Option C")
    with col2:
        opt_d = st.text_input("Option D")
        opt_e = st.text_input("Option E")
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    options = {"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d, "E": opt_e}
    if not prompt.strip() or any(not v.strip() for v in options.values()):
        st.error("Please fill in the question and all 5 options.")
    else:
        with st.spinner("Retrieving context, reranking, and scoring..."):
            ranked, context = predict(prompt, options, pipeline)

        st.subheader("Top-3 Predicted Answers")
        medals = ["🥇", "🥈", "🥉"]
        for i, (label, score) in enumerate(ranked[:3]):
            st.markdown(f"{medals[i]} **Option {label}**: {options[label]}  \n*Ranker score: {score:.3f}*")

        with st.expander("Retrieved supporting context (used for this prediction)"):
            st.write(context)

        with st.expander("All 5 options ranked"):
            for label, score in ranked:
                st.write(f"{label}: {score:.3f} — {options[label]}")


st.divider()
st.caption(
    "Pipeline: BGE-base bi-encoder retrieval (top-20) \u2192 MS-MARCO cross-encoder rerank "
    "\u2192 TF-IDF + numeric features \u2192 LightGBM ranker. "
    "See the [project report] for full architecture details."
)
