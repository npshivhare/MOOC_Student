import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Dict


def compute_nmi(labels1: List[int], labels2: List[int]) -> float:
    return float(normalized_mutual_info_score(labels1, labels2))


def compute_jaccard_fast(labels1: List[int], labels2: List[int]) -> float:
    """
    Fast Jaccard using set-based contingency instead of O(n²) pairs.
    Equivalent result, O(n) time.
    """
    n = len(labels1)
    if n == 0:
        return 0.0

    a1 = np.array(labels1)
    a2 = np.array(labels2)

    tp = fp = fn = 0
    # Group indices by label in labels1
    from collections import defaultdict
    groups1 = defaultdict(set)
    groups2 = defaultdict(set)
    for i, (l1, l2) in enumerate(zip(a1, a2)):
        groups1[l1].add(i)
        groups2[l2].add(i)

    for g1 in groups1.values():
        for g2 in groups2.values():
            inter = len(g1 & g2)
            if inter == 0:
                continue
            tp += inter * (inter - 1) // 2
            fp += inter * (len(g2) - inter)
            fn += inter * (len(g1) - inter)

    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def compute_cosine(labels1: List[int], labels2: List[int]) -> float:
    v1 = np.array(labels1, dtype=float).reshape(1, -1)
    v2 = np.array(labels2, dtype=float).reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0])


def pairwise_comparison(algorithms: Dict[str, List[int]]) -> List[Dict]:
    names = list(algorithms.keys())
    results = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            l1, l2 = algorithms[n1], algorithms[n2]
            results.append({
                "pair": f"{n1} vs {n2}",
                "NMI": round(compute_nmi(l1, l2), 4),
                "Jaccard": round(compute_jaccard_fast(l1, l2), 4),
                "Cosine": round(compute_cosine(l1, l2), 4),
            })
    return results


def community_summary(G: nx.Graph, communities: List[List], labels: List[int], modularity: float) -> Dict:
    sizes = [len(c) for c in communities]
    return {
        "num_communities": len(communities),
        "modularity": round(modularity, 6),
        "largest_community": max(sizes) if sizes else 0,
        "smallest_community": min(sizes) if sizes else 0,
        "avg_community_size": round(float(np.mean(sizes)), 2) if sizes else 0,
    }