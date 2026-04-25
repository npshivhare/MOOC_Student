import re
import string
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
import streamlit as st

import nltk
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

PREDEFINED_TOPICS = {
    "Technology": ["software", "tech", "code", "computer", "ai", "data", "network", "system", "digital", "cloud"],
    "Science": ["research", "study", "experiment", "biology", "physics", "chemistry", "lab", "science", "theory"],
    "Politics": ["government", "policy", "election", "political", "vote", "president", "law", "senate", "congress"],
    "Sports": ["game", "team", "player", "score", "championship", "match", "tournament", "league", "win", "sport"],
    "Business": ["company", "market", "finance", "investment", "revenue", "profit", "corporate", "industry", "trade"],
    "Health": ["medical", "health", "disease", "hospital", "patient", "doctor", "treatment", "medicine", "care"],
    "Entertainment": ["movie", "music", "film", "actor", "show", "celebrity", "art", "culture", "media", "entertainment"],
    "Education": ["school", "university", "student", "learning", "education", "teacher", "course", "knowledge"],
}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(_PUNCT_TABLE)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def extract_top_keywords(texts: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
    # For large communities, sample texts to stay fast
    if len(texts) > 500:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(texts), 500, replace=False)
        texts = [texts[i] for i in idx]

    cleaned = [clean_text(t) for t in texts]
    non_empty = [t for t in cleaned if t.strip()]
    if not non_empty:
        return []

    if len(non_empty) == 1:
        words = non_empty[0].split()
        freq = Counter(words)
        total = sum(freq.values()) or 1
        return sorted([(w, round(c / total, 4)) for w, c in freq.items()], key=lambda x: -x[1])[:top_n]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Limit vocabulary for speed
        vec = TfidfVectorizer(max_features=300, ngram_range=(1, 1), min_df=1)
        tfidf_matrix = vec.fit_transform(non_empty)
        scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        feature_names = vec.get_feature_names_out()
        top_idx = scores.argsort()[-top_n:][::-1]
        return [(feature_names[i], round(float(scores[i]), 4)) for i in top_idx]
    except Exception:
        words = " ".join(non_empty).split()
        freq = Counter(words)
        total = sum(freq.values()) or 1
        return sorted([(w, round(c / total, 4)) for w, c in freq.items()], key=lambda x: -x[1])[:top_n]


def classify_topic_simple(keywords: List[Tuple[str, float]]) -> str:
    kw_set = {kw.lower() for kw, _ in keywords}
    best_topic, best_score = "General", 0
    for topic, topic_kws in PREDEFINED_TOPICS.items():
        score = len(kw_set & set(topic_kws))
        if score > best_score:
            best_score, best_topic = score, topic
    return best_topic


def analyze_communities(
    communities: List[List],
    node_texts: Optional[Dict[str, List[str]]],
    top_n: int = 10,
) -> List[Dict]:
    results = []
    for i, comm in enumerate(communities):
        entry: Dict = {
            "community_id": i,
            "size": len(comm),
            "nodes": comm[:20],
            "keywords": [],
            "label": f"Community {i}",
            "topic": "N/A",
        }

        if node_texts:
            texts = []
            for node in comm:
                texts.extend(node_texts.get(str(node), []))

            if texts:
                keywords = extract_top_keywords(texts, top_n)
                entry["keywords"] = keywords
                top3 = [kw for kw, _ in keywords[:3]]
                entry["label"] = " · ".join(top3) if top3 else f"Community {i}"
                entry["topic"] = classify_topic_simple(keywords)

        results.append(entry)
    return results