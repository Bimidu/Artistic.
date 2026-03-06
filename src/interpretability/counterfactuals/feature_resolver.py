# src/interpretability/feature_resolver.py

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
from openai import OpenAI
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

client = OpenAI()

# ---------------------------------------------------
# LOAD FEATURE DATABASE
# ---------------------------------------------------

FEATURE_CSV = Path("assets/feature_explanations/feature_explanations_literal.csv")

feature_df = pd.read_csv(FEATURE_CSV)

#extracts feature names from file to a list
FEATURE_NAMES = feature_df["feature_name"].astype(str).tolist()
FEATURE_DESCRIPTIONS = (
    feature_df["description"]
    .fillna("")
    .astype(str)
    .tolist()
)

# ---------------------------------------------------
# EMBEDDING CACHE
# ---------------------------------------------------

EMBED_CACHE_PATH = Path("assets/feature_embeddings.npy")


def _embed(text: str):
    """Create embedding using OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def _build_embedding_cache():
    logger.info("Building feature embedding cache...")

    embeddings = []

    for name, desc in zip(FEATURE_NAMES, FEATURE_DESCRIPTIONS):
        text = f"{name}. {desc}"
        embeddings.append(_embed(text))

    embeddings = np.vstack(embeddings)
    np.save(EMBED_CACHE_PATH, embeddings)

    logger.info("Feature embeddings cached.")
    return embeddings


# Load or create embeddings
if EMBED_CACHE_PATH.exists():
    FEATURE_EMBEDDINGS = np.load(EMBED_CACHE_PATH)
else:
    FEATURE_EMBEDDINGS = _build_embedding_cache()


# ---------------------------------------------------
# FUZZY MATCH (TYPO HANDLING)
# ---------------------------------------------------

def fuzzy_match_feature(query: str, threshold: int = 75):

    match = process.extractOne(
        query,
        FEATURE_NAMES,
        scorer=fuzz.WRatio
    )

    if match and match[1] >= threshold:
        return match[0]

    return None

# ---------------------------------------------------
# SEMANTIC MATCH (CLINICAL LANGUAGE)
# ---------------------------------------------------

def semantic_match_feature(query: str):

    q_emb = _embed(query)

    similarities = FEATURE_EMBEDDINGS @ q_emb

    best_idx = int(np.argmax(similarities))
    best_feature = FEATURE_NAMES[best_idx]
    score = float(similarities[best_idx])

    return best_feature, score

# ---------------------------------------------------
# MAIN RESOLVER
# ---------------------------------------------------

def resolve_feature(user_phrase: str):

    phrase = user_phrase.strip().lower()

    # Exact match
    if phrase in FEATURE_NAMES:
        logger.info(f"Exact feature match: {phrase}")
        return phrase

    # Fuzzy match (typos)
    fuzzy = fuzzy_match_feature(phrase)
    if fuzzy:
        logger.info(f"Fuzzy matched '{phrase}' → '{fuzzy}'")
        return fuzzy

    # Semantic match (concept understanding)
    semantic_feature, score = semantic_match_feature(phrase)

    logger.info(
        f"Semantic matched '{phrase}' → '{semantic_feature}' (score={score:.3f})"
    )

    return semantic_feature