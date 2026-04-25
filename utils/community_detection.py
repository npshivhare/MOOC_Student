import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
import community as community_louvain
import igraph as ig
import leidenalg
import streamlit as st

GN_NODE_LIMIT = 500


def _partition_to_labels(partition: dict, nodes: list) -> List[int]:
    return [partition[node] for node in nodes]


def _communities_to_labels(communities: List[List], nodes: list) -> List[int]:
    label_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            label_map[node] = i
    return [label_map.get(node, 0) for node in nodes]


def run_louvain(G: nx.Graph) -> Tuple[List[int], float, List[List]]:
    partition = community_louvain.best_partition(G)
    labels = _partition_to_labels(partition, list(G.nodes()))
    mod = community_louvain.modularity(partition, G)
    label_set = set(labels)
    node_list = list(G.nodes())
    communities = [
        [n for n, l in zip(node_list, labels) if l == i]
        for i in label_set
    ]
    return labels, mod, communities


def run_leiden(G: nx.Graph) -> Tuple[List[int], float, List[List]]:
    node_list = list(G.nodes())
    node_index = {n: i for i, n in enumerate(node_list)}
    edges = [(node_index[u], node_index[v]) for u, v in G.edges()]

    g_ig = ig.Graph(n=len(node_list), edges=edges, directed=False)
    leiden_part = leidenalg.find_partition(
        g_ig, leidenalg.ModularityVertexPartition,
        n_iterations=2,
        seed=42,
    )

    idx_to_node = {i: n for n, i in node_index.items()}
    communities = [[idx_to_node[i] for i in c] for c in leiden_part]
    labels = _communities_to_labels(communities, node_list)
    mod = leiden_part.modularity
    return labels, mod, communities


def run_label_propagation(G: nx.Graph) -> Tuple[List[int], float, List[List]]:
    lp_comms = list(nx.algorithms.community.label_propagation_communities(G))
    communities = [list(c) for c in lp_comms]
    labels = _communities_to_labels(communities, list(G.nodes()))
    mod = nx.algorithms.community.quality.modularity(G, lp_comms)
    return labels, mod, communities


def run_girvan_newman(G: nx.Graph) -> Tuple[List[int], float, List[List]]:
    n = G.number_of_nodes()
    if n > GN_NODE_LIMIT:
        lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        if lcc.number_of_nodes() > GN_NODE_LIMIT:
            top_nodes = sorted(lcc.degree(), key=lambda x: x[1], reverse=True)[:GN_NODE_LIMIT]
            lcc = lcc.subgraph([n for n, _ in top_nodes]).copy()
        G_run = lcc
        _warn_gn(n)
    else:
        G_run = G

    gn_gen = nx.algorithms.community.girvan_newman(G_run)
    comms = next(gn_gen)
    communities = [list(c) for c in comms]

    node_list = list(G.nodes())
    labels = _communities_to_labels(communities, list(G_run.nodes()))

    if G_run is not G:
        sub_nodes = set(G_run.nodes())
        label_map = {n: l for n, l in zip(G_run.nodes(), labels)}
        full_labels = [label_map.get(node, 0) for node in node_list]
        full_communities = [
            [n for n in node_list if full_labels[i] == idx]
            for idx, i in enumerate(set(full_labels))
        ]
        lset = sorted(set(full_labels))
        full_communities = [
            [n for n, l in zip(node_list, full_labels) if l == idx]
            for idx in lset
        ]
        mod = nx.algorithms.community.quality.modularity(G, [set(c) for c in full_communities])
        return full_labels, mod, full_communities

    mod = nx.algorithms.community.quality.modularity(G, [set(c) for c in communities])
    return labels, mod, communities


def _warn_gn(original_n: int):
    st.warning(
        f"Girvan-Newman is O(n) — your graph has {original_n:,} nodes. "
        f"Running on a representative subgraph of {GN_NODE_LIMIT} nodes for speed. "
        "Use Louvain or Leiden for full accuracy on large graphs."
    )


ALGORITHMS = {
    "Louvain": run_louvain,
    "Leiden": run_leiden,
    "Label Propagation": run_label_propagation,
    "Girvan-Newman": run_girvan_newman,
}


def detect_communities(G: nx.Graph, algorithm: str, cache_key: str = "") -> Tuple[List[int], float, List[List]]:
    skey = f"_detect_{cache_key}_{algorithm}"
    if skey in st.session_state and st.session_state[skey] is not None:
        return st.session_state[skey]

    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    result = ALGORITHMS[algorithm](G)
    st.session_state[skey] = result
    return result