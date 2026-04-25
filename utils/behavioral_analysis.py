import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def calculate_learner_statistics(behavioral_df: pd.DataFrame) -> Dict:
    stats = {
        "total_actions": {
            "mean": float(behavioral_df["total_actions"].mean()),
            "std": float(behavioral_df["total_actions"].std()),
            "min": float(behavioral_df["total_actions"].min()),
            "max": float(behavioral_df["total_actions"].max()),
        },
        "avg_gap": {
            "mean": float(behavioral_df["avg_gap"].mean()),
            "std": float(behavioral_df["avg_gap"].std()),
            "min": float(behavioral_df["avg_gap"].min()),
            "max": float(behavioral_df["avg_gap"].max()),
        },
        "duration": {
            "mean": float(behavioral_df["duration"].mean()),
            "std": float(behavioral_df["duration"].std()),
            "min": float(behavioral_df["duration"].min()),
            "max": float(behavioral_df["duration"].max()),
        },
        "unique_targets": {
            "mean": float(behavioral_df["unique_targets"].mean()),
            "std": float(behavioral_df["unique_targets"].std()),
            "min": float(behavioral_df["unique_targets"].min()),
            "max": float(behavioral_df["unique_targets"].max()),
        },
    }
    return stats


def classify_learner(row: pd.Series, stats: Dict) -> str:
    ta = row["total_actions"]
    ag = row["avg_gap"]
    dur = row["duration"]
    ut = row["unique_targets"]

    ta_mean, ta_std = stats["total_actions"]["mean"], stats["total_actions"]["std"]
    ag_mean, ag_std = stats["avg_gap"]["mean"], stats["avg_gap"]["std"]
    dur_mean, dur_std = stats["duration"]["mean"], stats["duration"]["std"]
    ut_mean, ut_std = stats["unique_targets"]["mean"], stats["unique_targets"]["std"]

    std_ta = ta_std if ta_std > 0 else 1
    std_ag = ag_std if ag_std > 0 else 1
    std_dur = dur_std if dur_std > 0 else 1
    std_ut = ut_std if ut_std > 0 else 1

    active_score = 0
    inactive_score = 0

    if ta > ta_mean + 0.3 * std_ta:
        active_score += 1
    elif ta < ta_mean - 0.3 * std_ta:
        inactive_score += 1

    if ag < ag_mean - 0.3 * std_ag:
        active_score += 1
    elif ag > ag_mean + 0.3 * std_ag:
        inactive_score += 1

    if dur > dur_mean + 0.3 * std_dur:
        active_score += 1
    elif dur < dur_mean - 0.3 * std_dur:
        inactive_score += 1

    if ut > ut_mean + 0.3 * std_ut:
        active_score += 1
    elif ut < ut_mean - 0.3 * std_ut:
        inactive_score += 1

    if active_score >= 3:
        return "Active"
    elif inactive_score >= 3:
        return "Inactive"
    else:
        return "Average"


def analyze_cluster_behavior(behavioral_df: pd.DataFrame, community_labels: List[int], nodes: List[str]) -> List[Dict]:
    node_to_label = dict(zip(nodes, community_labels))

    results = []
    for cluster_id in sorted(behavioral_df["cluster"].unique()):
        cluster_data = behavioral_df[behavioral_df["cluster"] == cluster_id]

        ta_mean = cluster_data["total_actions"].mean()
        ag_mean = cluster_data["avg_gap"].mean()
        dur_mean = cluster_data["duration"].mean()
        ut_mean = cluster_data["unique_targets"].mean()

        learner_type_counts = cluster_data["learner_type"].value_counts().to_dict()

        determined_type = classify_learner_by_stats(ta_mean, ag_mean, dur_mean, ut_mean, behavioral_df)

        results.append({
            "cluster_id": int(cluster_id),
            "size": len(cluster_data),
            "avg_total_actions": round(float(ta_mean), 2),
            "avg_avg_gap": round(float(ag_mean), 2),
            "avg_duration": round(float(dur_mean), 2),
            "avg_unique_targets": round(float(ut_mean), 2),
            "learner_distribution": learner_type_counts,
            "determined_learner_type": determined_type,
        })

    return results


def classify_learner_by_stats(ta_mean: float, ag_mean: float, dur_mean: float, ut_mean: float, df: pd.DataFrame) -> str:
    global_stats = calculate_learner_statistics(df)

    ta_std = global_stats["total_actions"]["std"] if global_stats["total_actions"]["std"] > 0 else 1
    ag_std = global_stats["avg_gap"]["std"] if global_stats["avg_gap"]["std"] > 0 else 1
    dur_std = global_stats["duration"]["std"] if global_stats["duration"]["std"] > 0 else 1
    ut_std = global_stats["unique_targets"]["std"] if global_stats["unique_targets"]["std"] > 0 else 1

    active_conditions = 0
    inactive_conditions = 0

    if ta_mean > global_stats["total_actions"]["mean"] + 0.3 * ta_std:
        active_conditions += 1
    elif ta_mean < global_stats["total_actions"]["mean"] - 0.3 * ta_std:
        inactive_conditions += 1

    if ag_mean < global_stats["avg_gap"]["mean"] - 0.3 * ag_std:
        active_conditions += 1
    elif ag_mean > global_stats["avg_gap"]["mean"] + 0.3 * ag_std:
        inactive_conditions += 1

    if dur_mean > global_stats["duration"]["mean"] + 0.3 * dur_std:
        active_conditions += 1
    elif dur_mean < global_stats["duration"]["mean"] - 0.3 * dur_std:
        inactive_conditions += 1

    if ut_mean > global_stats["unique_targets"]["mean"] + 0.3 * ut_std:
        active_conditions += 1
    elif ut_mean < global_stats["unique_targets"]["mean"] - 0.3 * ut_std:
        inactive_conditions += 1

    if active_conditions >= 3:
        return "Active Learners"
    elif inactive_conditions >= 3:
        return "Inactive / Dead Learners"
    else:
        return "Average Learners"


def analyze_communities_behavioral(behavioral_df: pd.DataFrame, communities: List[List], labels: List[int], nodes: List[str]) -> List[Dict]:
    results = []
    node_to_data = {str(row["userid"]): row for _, row in behavioral_df.iterrows()}

    global_stats = calculate_learner_statistics(behavioral_df)

    for i, comm_nodes in enumerate(communities):
        comm_data = []
        for node in comm_nodes:
            if node in node_to_data:
                comm_data.append(node_to_data[node])

        if not comm_data:
            results.append({
                "community_id": i,
                "size": len(comm_nodes),
                "avg_total_actions": 0,
                "avg_avg_gap": 0,
                "avg_duration": 0,
                "avg_unique_targets": 0,
                "learner_type": "Unknown",
                "description": "No behavioral data available",
            })
            continue

        comm_df = pd.DataFrame(comm_data)

        ta_mean = comm_df["total_actions"].mean()
        ag_mean = comm_df["avg_gap"].mean()
        dur_mean = comm_df["duration"].mean()
        ut_mean = comm_df["unique_targets"].mean()

        learner_type = classify_learner_by_stats(ta_mean, ag_mean, dur_mean, ut_mean, behavioral_df)

        description = get_learner_description(ta_mean, ag_mean, dur_mean, ut_mean, global_stats)

        results.append({
            "community_id": i,
            "size": len(comm_nodes),
            "avg_total_actions": round(float(ta_mean), 2),
            "avg_avg_gap": round(float(ag_mean), 2),
            "avg_duration": round(float(dur_mean), 2),
            "avg_unique_targets": round(float(ut_mean), 2),
            "learner_type": learner_type,
            "description": description,
        })

    return results


def get_learner_description(ta: float, ag: float, dur: float, ut: float, stats: Dict) -> str:
    descriptions = []

    ta_ratio = ta / stats["total_actions"]["mean"] if stats["total_actions"]["mean"] > 0 else 1
    ag_ratio = ag / stats["avg_gap"]["mean"] if stats["avg_gap"]["mean"] > 0 else 1
    dur_ratio = dur / stats["duration"]["mean"] if stats["duration"]["mean"] > 0 else 1
    ut_ratio = ut / stats["unique_targets"]["mean"] if stats["unique_targets"]["mean"] > 0 else 1

    if ta_ratio > 1.2:
        descriptions.append("High activity")
    elif ta_ratio < 0.8:
        descriptions.append("Low activity")

    if ag_ratio < 0.8:
        descriptions.append("Consistent engagement")
    elif ag_ratio > 1.2:
        descriptions.append("Irregular engagement")

    if dur_ratio > 1.2:
        descriptions.append("Long sessions")
    elif dur_ratio < 0.8:
        descriptions.append("Short sessions")

    if ut_ratio > 1.2:
        descriptions.append("Diverse interactions")
    elif ut_ratio < 0.8:
        descriptions.append("Focused interactions")

    return ", ".join(descriptions) if descriptions else "Moderate behavior"


def assign_learner_types(behavioral_df: pd.DataFrame) -> pd.DataFrame:
    stats = calculate_learner_statistics(behavioral_df)
    df = behavioral_df.copy()
    df["learner_type_assigned"] = df.apply(lambda row: classify_learner(row, stats), axis=1)
    return df


def recalculate_all_learner_types(behavioral_df: pd.DataFrame) -> pd.DataFrame:
    stats = calculate_learner_statistics(behavioral_df)
    df = behavioral_df.copy()
    
    if "learner_type" in df.columns:
        df["learner_type"] = df.apply(lambda row: classify_learner(row, stats), axis=1)
    else:
        df["learner_type"] = df.apply(lambda row: classify_learner(row, stats), axis=1)
    
    return df