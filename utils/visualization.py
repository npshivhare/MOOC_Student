import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple
import io
import streamlit as st

GRAPH_RENDER_LIMIT = 1500
EDGE_RENDER_LIMIT = 5000


def _sample_graph(G: nx.Graph, labels: List[int], max_nodes: int) -> Tuple[nx.Graph, List[int]]:
    if G.number_of_nodes() <= max_nodes:
        return G, labels
    top = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
    keep = {n for n, _ in top}
    sub = G.subgraph(keep).copy()
    node_list = list(G.nodes())
    label_map = {n: l for n, l in zip(node_list, labels)}
    sub_labels = [label_map[n] for n in sub.nodes()]
    return sub, sub_labels


@st.cache_data(show_spinner=False, max_entries=10)
def _compute_layout(layout: str, edge_list: list, node_list: list) -> dict:
    G_tmp = nx.Graph()
    G_tmp.add_nodes_from(node_list)
    G_tmp.add_edges_from(edge_list)
    n = len(node_list)
    k_val = 1.5 / max(1, np.sqrt(n))

    if layout == "spring":
        return nx.spring_layout(G_tmp, seed=42, k=k_val)
    elif layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G_tmp)
    elif layout == "circular":
        return nx.circular_layout(G_tmp)
    elif layout == "spectral":
        return nx.spectral_layout(G_tmp)
    return nx.spring_layout(G_tmp, seed=42, k=k_val)


def plot_graph_plotly(
    G: nx.Graph,
    labels: List[int],
    title: str = "Community Graph",
    layout: str = "spring",
    community_names: Optional[List[str]] = None,
) -> go.Figure:

    G_plot, labels_plot = _sample_graph(G, labels, GRAPH_RENDER_LIMIT)
    sampled = G_plot.number_of_nodes() < G.number_of_nodes()
    actual_title = title + (
        f" (showing {G_plot.number_of_nodes():,}/{G.number_of_nodes():,} nodes)" if sampled else ""
    )

    node_list = list(G_plot.nodes())
    edge_list = list(G_plot.edges())

    if len(edge_list) > EDGE_RENDER_LIMIT:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(edge_list), EDGE_RENDER_LIMIT, replace=False)
        edge_list = [edge_list[i] for i in idx]

    pos = _compute_layout(layout, list(G_plot.edges()), node_list)

    unique_labels = sorted(set(labels_plot))
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    color_map = {}
    for i, lab in enumerate(unique_labels):
        r, g, b, _ = cmap(i % 20)
        color_map[lab] = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

    edge_x, edge_y = [], []
    for u, v in edge_list:
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#444"),
        hoverinfo="none",
        showlegend=False,
    )

    node_traces = []
    for lab in unique_labels:
        group_nodes = [nd for nd, l in zip(node_list, labels_plot) if l == lab]
        if not group_nodes:
            continue
        valid_nodes = [nd for nd in group_nodes if nd in pos]
        x_vals = [pos[nd][0] for nd in valid_nodes]
        y_vals = [pos[nd][1] for nd in valid_nodes]
        degrees = [G_plot.degree(nd) for nd in valid_nodes]
        node_sizes = [max(6, min(22, 5 + d)) for d in degrees]
        name = community_names[lab] if (community_names and lab < len(community_names)) else f"Community {lab}"

        node_traces.append(go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers",
            marker=dict(
                size=node_sizes,
                color=color_map[lab],
                line=dict(width=0.8, color="#fff"),
                opacity=0.9,
            ),
            text=[f"Node: {nd}<br>Comm: {lab}<br>Degree: {G_plot.degree(nd)}" for nd in valid_nodes],
            hoverinfo="text",
            name=name,
        ))

    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title=dict(text=actual_title, font=dict(size=15, color="#e8e8e8"), x=0.5),
            paper_bgcolor="#0f0f1a",
            plot_bgcolor="#0f0f1a",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                bgcolor="rgba(20,20,40,0.85)",
                bordercolor="#444",
                borderwidth=1,
                font=dict(color="#ccc", size=10),
            ),
            margin=dict(b=20, l=10, r=10, t=50),
            hovermode="closest",
        )
    )
    return fig


def plot_modularity_bar(modularities: Dict[str, float]) -> go.Figure:
    names = list(modularities.keys())
    values = [modularities[n] for n in names]
    max_val = max(values) if values else 1
    colors = ["#00e5ff" if v == max_val else "#7b61ff" for v in values]

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(color="#e8e8e8"),
    ))
    fig.update_layout(
        title=dict(text="Modularity Comparison", font=dict(size=16, color="#e8e8e8"), x=0.5),
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#141428",
        xaxis=dict(color="#aaa", gridcolor="#222"),
        yaxis=dict(color="#aaa", gridcolor="#222", title="Modularity"),
        margin=dict(t=50, b=30, l=40, r=20),
        font=dict(color="#ccc"),
    )
    return fig


def plot_community_sizes(communities: List[List], algorithm: str) -> go.Figure:
    sizes = sorted([len(c) for c in communities], reverse=True)
    labels = [f"C{i}" for i in range(len(sizes))]

    fig = go.Figure(go.Bar(
        x=labels, y=sizes,
        marker=dict(
            color=sizes,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Size", tickfont=dict(color="#ccc")),
        ),
        text=sizes,
        textposition="outside",
        texttemplate="<b>%{text}</b>",
    ))
    fig.update_layout(
        title=dict(text=f"Community Sizes — {algorithm}", font=dict(size=15, color="#e8e8e8"), x=0.5),
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#141428",
        xaxis=dict(color="#aaa", gridcolor="#222", title="Community"),
        yaxis=dict(color="#aaa", gridcolor="#222", title="Nodes"),
        margin=dict(t=50, b=40, l=50, r=20),
        font=dict(color="#ccc"),
    )
    return fig


def generate_wordcloud_image(keywords: list) -> Optional[bytes]:
    try:
        from wordcloud import WordCloud
        freq = {kw: max(0.01, float(score)) for kw, score in keywords}
        if not freq:
            return None
        wc = WordCloud(
            width=600, height=280,
            background_color="#0f0f1a",
            colormap="plasma",
            max_words=40,
        ).generate_from_frequencies(freq)
        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def plot_learner_distribution(behavioral_df: pd.DataFrame) -> go.Figure:
    learner_counts = behavioral_df["learner_type"].value_counts()
    
    colors = {
        "Active": "#00e5ff",
        "Inactive": "#ff6b9d",
        "Average": "#ffd166"
    }
    bar_colors = [colors.get(lt, "#7b61ff") for lt in learner_counts.index]
    
    fig = go.Figure(go.Bar(
        x=learner_counts.index,
        y=learner_counts.values,
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=learner_counts.values,
        textposition="outside",
        textfont=dict(color="#e8e8e8"),
    ))
    fig.update_layout(
        title=dict(text="Learner Type Distribution", font=dict(size=16, color="#e8e8e8"), x=0.5),
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#141428",
        xaxis=dict(color="#aaa", gridcolor="#222"),
        yaxis=dict(color="#aaa", gridcolor="#222", title="Count"),
        margin=dict(t=50, b=30, l=40, r=20),
        font=dict(color="#ccc"),
    )
    return fig


def plot_learner_distribution_pie(behavioral_df: pd.DataFrame) -> go.Figure:
    learner_counts = behavioral_df["learner_type"].value_counts()
    
    colors = {
        "Active": "#00e5ff",
        "Inactive": "#ff6b9d",
        "Average": "#ffd166"
    }
    pie_colors = [colors.get(lt, "#7b61ff") for lt in learner_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=learner_counts.index,
        values=learner_counts.values,
        marker=dict(colors=pie_colors),
        textinfo="label+percent",
        textfont=dict(color="#e8e8e8"),
        hole=0.4,
    )])
    fig.update_layout(
        title=dict(text="Learner Type Distribution", font=dict(size=16, color="#e8e8e8"), x=0.5),
        paper_bgcolor="#0f0f1a",
        margin=dict(t=50, b=20, l=20, r=20),
        font=dict(color="#ccc"),
        showlegend=True,
        legend=dict(bgcolor="rgba(20,20,40,0.85)", bordercolor="#444", font=dict(color="#ccc")),
    )
    return fig


def plot_user_behavior_scatter(behavioral_df: pd.DataFrame) -> go.Figure:
    colors = {
        "Active": "#00e5ff",
        "Inactive": "#ff6b9d",
        "Average": "#ffd166"
    }
    
    fig = go.Figure()
    
    for learner_type in behavioral_df["learner_type"].unique():
        df_filtered = behavioral_df[behavioral_df["learner_type"] == learner_type]
        fig.add_trace(go.Scatter(
            x=df_filtered["total_actions"],
            y=df_filtered["avg_gap"],
            mode="markers",
            name=learner_type,
            marker=dict(
                size=df_filtered["unique_targets"] / 2,
                color=colors.get(learner_type, "#7b61ff"),
                opacity=0.7,
            ),
            text=df_filtered["userid"],
            hovertemplate="<b>User %{text}</b><br>Actions: %{x}<br>Gap: %{y}<br>Targets: %{marker.size}<extra></extra>",
        ))
    
    fig.update_layout(
        title=dict(text="User Behavior Analysis", font=dict(size=16, color="#e8e8e8"), x=0.5),
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#141428",
        xaxis=dict(title="Total Actions", color="#aaa", gridcolor="#222"),
        yaxis=dict(title="Avg Gap", color="#aaa", gridcolor="#222"),
        font=dict(color="#ccc"),
        showlegend=True,
        legend=dict(bgcolor="rgba(20,20,40,0.85)", bordercolor="#444", font=dict(color="#ccc")),
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig
    return fig


def plot_behavioral_heatmap(semantic_data: List[Dict]) -> go.Figure:
    communities = [s["community_id"] for s in semantic_data]
    metrics = ["avg_total_actions", "avg_avg_gap", "avg_duration", "avg_unique_targets"]
    metric_labels = ["Avg Actions", "Avg Gap", "Avg Duration", "Avg Targets"]
    
    z_values = []
    for s in semantic_data:
        row = [s.get(m, 0) for m in metrics]
        z_values.append(row)
    
    if not z_values or all(all(v == 0 for v in row) for row in z_values):
        return go.Figure()
    
    z_array = np.array(z_values)
    if z_array.size > 0 and z_array.max() > 0:
        z_normalized = (z_array - z_array.min(axis=0)) / (z_array.max(axis=0) - z_array.min(axis=0) + 1e-10)
    else:
        z_normalized = z_array
    
    fig = go.Figure(data=go.Heatmap(
        z=z_normalized.T if z_normalized.size > 0 else [[]],
        x=[f"C{c}" for c in communities],
        y=metric_labels,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="Normalized", tickfont=dict(color="#ccc")),
        hovertemplate="Community: %{x}<br>Metric: %{y}<br>Value: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Behavioral Metrics Heatmap by Community", font=dict(size=16, color="#e8e8e8"), x=0.5),
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#141428",
        xaxis=dict(color="#aaa", gridcolor="#222"),
        yaxis=dict(color="#aaa", gridcolor="#222"),
        margin=dict(t=50, b=40, l=100, r=20),
        font=dict(color="#ccc"),
    )
    return fig