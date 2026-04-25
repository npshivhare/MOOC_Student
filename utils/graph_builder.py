import networkx as nx
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import hashlib
import streamlit as st


def _df_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def detect_format(df: pd.DataFrame) -> str:
    cols = [c.lower().strip() for c in df.columns]
    if "userid" in cols and "total_actions" in cols and "cluster" in cols:
        return "behavioral"
    return "edge_list"


def build_behavioral_graph(df: pd.DataFrame) -> Tuple[nx.Graph, Optional[dict], Optional[pd.DataFrame]]:
    G = nx.Graph()
    
    user_data = {}
    for _, row in df.iterrows():
        user_id = str(row["userid"]).strip()
        G.add_node(user_id, 
                   total_actions=int(row["total_actions"]),
                   avg_gap=float(row["avg_gap"]),
                   duration=float(row["duration"]),
                   unique_targets=int(row["unique_targets"]),
                   cluster=int(row["cluster"]),
                   learner_type=str(row["learner_type"]).strip())
        user_data[user_id] = {
            "total_actions": int(row["total_actions"]),
            "avg_gap": float(row["avg_gap"]),
            "duration": float(row["duration"]),
            "unique_targets": int(row["unique_targets"]),
            "cluster": int(row["cluster"]),
            "learner_type": str(row["learner_type"]).strip()
        }
    
    for _, row in df.iterrows():
        src = str(row["userid"]).strip()
        tgt = str(row["unique_targets"])
        if src != tgt:
            G.add_edge(src, tgt)
    
    return G, None, pd.DataFrame.from_dict(user_data, orient="index")


@st.cache_data(show_spinner=False, max_entries=5)
def _build_graph_cached(df_hash: str, df: pd.DataFrame, format_type: str) -> Tuple[object, Optional[dict], Optional[pd.DataFrame]]:
    cols = [c.lower().strip() for c in df.columns]
    df = df.copy()
    df.columns = cols

    if format_type == "behavioral":
        return build_behavioral_graph(df)

    if "source" not in cols or "target" not in cols:
        raise ValueError("Dataset must have 'source' and 'target' columns.")

    has_weight = "weight" in cols
    has_text = "text" in cols

    sources = df["source"].astype(str).str.strip().tolist()
    targets = df["target"].astype(str).str.strip().tolist()

    G = nx.Graph()

    if has_weight:
        weights = df["weight"].astype(float).tolist()
        G.add_weighted_edges_from(zip(sources, targets, weights))
    else:
        G.add_edges_from(zip(sources, targets))

    node_texts = None
    if has_text:
        node_texts = {}
        for src, tv in zip(sources, df["text"].astype(str).str.strip()):
            node_texts.setdefault(src, []).append(tv)

    return G, node_texts, None


def build_graph_from_df(df: pd.DataFrame) -> Tuple[nx.Graph, Optional[dict]]:
    h = _df_hash(df)
    fmt = detect_format(df)
    result = _build_graph_cached(h, df, fmt)
    return result[0], result[1]


def get_df_hash(df: pd.DataFrame) -> str:
    return _df_hash(df)


def get_behavioral_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    fmt = detect_format(df)
    if fmt == "behavioral":
        cols = [c.lower().strip() for c in df.columns]
        df_copy = df.copy()
        df_copy.columns = cols
        return df_copy
    return None


@st.cache_data(show_spinner=False, max_entries=10)
def graph_stats_cached(df_hash: str, num_nodes: int, num_edges: int) -> dict:
    return {}


def get_graph_stats(G: nx.Graph, df_hash: str) -> dict:
    cache_key = f"_gstats_{df_hash}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    n = G.number_of_nodes()
    density = nx.density(G)

    if n <= 2000:
        avg_clustering = nx.average_clustering(G)
    else:
        sample = list(G.nodes())[:1500]
        avg_clustering = nx.average_clustering(G, nodes=sample)

    degrees = [d for _, d in G.degree()]
    avg_degree = float(np.mean(degrees)) if degrees else 0.0
    num_components = nx.number_connected_components(G)

    result = {
        "num_nodes": n,
        "num_edges": G.number_of_edges(),
        "density": round(density, 6),
        "avg_clustering": round(avg_clustering, 6),
        "avg_degree": round(avg_degree, 4),
        "is_connected": num_components == 1,
        "num_components": num_components,
    }
    st.session_state[cache_key] = result
    return result