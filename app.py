"""
GraphMind — Community Detection & Semantic Analysis
Production-ready Streamlit dashboard with full caching.
"""

import io
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go

from utils.graph_builder import build_graph_from_df, get_graph_stats, get_df_hash, detect_format, get_behavioral_data
from utils.community_detection import detect_communities, ALGORITHMS
from utils.metrics import community_summary, pairwise_comparison
from utils.behavioral_analysis import analyze_communities_behavioral, calculate_learner_statistics
from utils.visualization import (
    plot_graph_plotly,
    plot_modularity_bar,
    plot_community_sizes,
    plot_behavioral_heatmap,
    plot_learner_distribution_pie,
    plot_user_behavior_scatter,
)

# ════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════
st.set_page_config(
    page_title="GraphMind — Community Detection",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0a0a14; color: #d4d4e8; }
.stApp { background: #0a0a14; }
h1, h2, h3 { font-family: 'Space Mono', monospace; letter-spacing: -0.02em; }
[data-testid="stSidebar"] { background: #0e0e1e !important; border-right: 1px solid #1e1e3a; }
[data-testid="metric-container"] { background: linear-gradient(135deg, #13132a 0%, #1a1a35 100%); border: 1px solid #252550; border-radius: 12px; padding: 1rem 1.2rem; }
[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace; color: #00e5ff !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #888 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
.stTabs [data-baseweb="tab-list"] { background: #0e0e1e; border-bottom: 1px solid #1e1e3a; gap: 0; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #666; font-family: 'Space Mono', monospace; font-size: 0.78rem; padding: 0.7rem 1.2rem; border-bottom: 2px solid transparent; letter-spacing: 0.04em; }
.stTabs [aria-selected="true"] { color: #00e5ff !important; border-bottom: 2px solid #00e5ff !important; background: transparent !important; }
.stSelectbox > div > div { background: #13132a; border: 1px solid #252550; border-radius: 8px; color: #ccc; }
.stButton > button { background: linear-gradient(135deg, #7b61ff, #00e5ff); color: #000; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 0.8rem; border: none; border-radius: 8px; padding: 0.6rem 1.4rem; letter-spacing: 0.05em; transition: opacity 0.2s; }
.stButton > button:hover { opacity: 0.85; }
[data-testid="stFileUploader"] { background: #13132a; border: 1.5px dashed #252570; border-radius: 12px; padding: 1rem; }
.stDataFrame { border: 1px solid #252550; border-radius: 10px; }
hr { border-color: #1e1e3a; }
.algo-card { background: linear-gradient(135deg, #13132a, #1a1a35); border: 1px solid #252550; border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem; }
.hero-title { font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700; background: linear-gradient(90deg, #7b61ff, #00e5ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2; }
.subtitle { color: #666; font-size: 0.9rem; margin-top: 0.3rem; letter-spacing: 0.05em; text-transform: uppercase; font-family: 'Space Mono', monospace; }
.kw-pill { display: inline-block; background: rgba(123,97,255,0.15); border: 1px solid rgba(123,97,255,0.4); color: #b8aaff; font-size: 0.72rem; padding: 2px 10px; border-radius: 20px; margin: 2px; font-family: 'Space Mono', monospace; }
.speed-badge { display: inline-block; background: rgba(0,229,255,0.1); border: 1px solid rgba(0,229,255,0.3); color: #00e5ff; font-size: 0.68rem; padding: 1px 8px; border-radius: 20px; font-family: 'Space Mono', monospace; margin-left: 6px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# SESSION STATE INIT
# ════════════════════════════════════════════
_STATE_KEYS = ["G", "df", "node_texts", "algo_results", "gstats", "df_hash", "compare_results", "behavioral_df"]
for k in _STATE_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None

# ════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="hero-title">Graph<br>Mind</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Community Detection</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "CSV or TXT edge list",
        type=["csv", "txt"],
        help="Required: source, target. Optional: weight, text",
        label_visibility="collapsed",
    )

    st.markdown("#### ⚙️ Algorithm")
    algorithm = st.selectbox("Algorithm", list(ALGORITHMS.keys()), index=0, label_visibility="collapsed")

    st.markdown("#### 🗺️ Layout")
    layout = st.selectbox("Layout", ["spring", "kamada_kawai", "circular", "spectral"], label_visibility="collapsed")

    run_btn = st.button("▶ RUN DETECTION", use_container_width=True)
    st.markdown("---")
    st.markdown("#### 🔬 Compare All")
    compare_btn = st.button("⚡ COMPARE ALL ALGORITHMS", use_container_width=True)
    st.markdown("---")
    st.caption("GraphMind v2.0 · Optimised for speed")


# ════════════════════════════════════════════
# LANDING PAGE
# ════════════════════════════════════════════
if uploaded_file is None:
    st.markdown('<div class="hero-title" style="font-size:1.8rem">Welcome to GraphMind</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Graph-based community detection & semantic analysis</p>', unsafe_allow_html=True)
    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("⚡ Louvain", "#00e5ff", "Fast modularity optimization. Scales to 100k+ nodes."),
        ("🔬 Leiden", "#7b61ff", "Improved Louvain with guaranteed connectivity. C++ backend."),
        ("🏷️ Label Prop", "#ff6b9d", "Linear time O(n). Best for very large graphs."),
        ("✂️ Girvan-Newman", "#ffd166", "Edge betweenness method. Auto-sampled on large graphs."),
    ]
    for col, (name, color, desc) in zip([col1, col2, col3, col4], cards):
        col.markdown(f"""<div class="algo-card">
            <div style="font-family:'Space Mono',monospace;color:{color};font-size:0.9rem;margin-bottom:0.4rem">{name}</div>
            <div style="font-size:0.8rem;color:#888">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Expected CSV formats:**")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("**Edge List Format:**")
        st.code("source,target\nAlice,Bob\nBob,Carol\n\n# With optional columns:\nsource,target,weight,text", language="csv")
    
    with col_f2:
        st.markdown("**Behavioral Format:**")
        st.code("USERID,total_actions,avg_gap,duration,unique_targets,cluster,learner_type\n0,76,31498.28,2362371.0,24,1,Average\n1,26,48051.72,1201293.0,14,1,Average", language="csv")

    sample_G = nx.karate_club_graph()
    sample_df = pd.DataFrame([{"source": u, "target": v} for u, v in sample_G.edges()])
    st.download_button("⬇️ Download Karate Club Sample CSV", data=sample_df.to_csv(index=False),
                       file_name="karate_sample.csv", mime="text/csv")
    st.stop()


# ════════════════════════════════════════════
# LOAD + CACHE GRAPH  (only rebuilds when file changes)
# ════════════════════════════════════════════
try:
    df_raw = pd.read_csv(uploaded_file)
    new_hash = get_df_hash(df_raw)
    
    df_format = detect_format(df_raw)
    behavioral_df = get_behavioral_data(df_raw)

    if st.session_state["df_hash"] != new_hash:
        with st.spinner("Building graph..."):
            G, node_texts = build_graph_from_df(df_raw)
            gstats = get_graph_stats(G, new_hash)
            st.session_state.update({
                "G": G,
                "df": df_raw,
                "node_texts": node_texts,
                "gstats": gstats,
                "df_hash": new_hash,
                "algo_results": None,
                "compare_results": None,
                "behavioral_df": behavioral_df,
                "data_format": df_format,
            })

except Exception as e:
    st.error(f"❌ Error loading file: {e}")
    st.stop()

G             = st.session_state["G"]
df            = st.session_state["df"]
node_texts    = st.session_state["node_texts"]
gstats        = st.session_state["gstats"]
df_hash       = st.session_state["df_hash"]
behavioral_df = st.session_state.get("behavioral_df")
data_format   = st.session_state.get("data_format", "edge_list")

# ════════════════════════════════════════════
# GRAPH OVERVIEW METRICS
# ════════════════════════════════════════════
st.markdown('<div class="hero-title" style="font-size:1.3rem;margin-bottom:0.8rem">Graph Overview</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Nodes",         f"{gstats['num_nodes']:,}")
m2.metric("Edges",         f"{gstats['num_edges']:,}")
m3.metric("Density",       f"{gstats['density']:.4f}")
m4.metric("Avg Clustering",f"{gstats['avg_clustering']:.4f}")
m5.metric("Avg Degree",    f"{gstats['avg_degree']:.2f}")
m6.metric("Components",    gstats["num_components"])

with st.expander("📋 Dataset Preview", expanded=False):
    st.dataframe(df.head(200), use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════
# RUN DETECTION  (cached per algo+dataset)
# ════════════════════════════════════════════
if run_btn:
    cache_key = f"_detect_{df_hash}_{algorithm}"
    already_cached = cache_key in st.session_state and st.session_state[cache_key] is not None

    if not already_cached:
        with st.spinner(f"Running {algorithm}..."):
            try:
                labels, mod, communities = detect_communities(G, algorithm, cache_key=df_hash)
                summary = community_summary(G, communities, labels, mod)
                
                if data_format == "behavioral" and behavioral_df is not None:
                    semantic = analyze_communities_behavioral(
                        behavioral_df, communities, labels, list(G.nodes())
                    )
                else:
                    from utils.semantic_analysis import analyze_communities as analyze_semantic
                    semantic = analyze_semantic(communities, node_texts)
                    
                st.session_state["algo_results"] = {
                    "algorithm": algorithm, "labels": labels, "mod": mod,
                    "communities": communities, "summary": summary, "semantic": semantic,
                }
            except Exception as e:
                st.error(f"{algorithm} failed: {e}")
                st.stop()
    else:
        if (st.session_state["algo_results"] is None or
                st.session_state["algo_results"].get("algorithm") != algorithm):
            with st.spinner("Loading cached results..."):
                labels, mod, communities = detect_communities(G, algorithm, cache_key=df_hash)
                summary = community_summary(G, communities, labels, mod)
                
                if data_format == "behavioral" and behavioral_df is not None:
                    semantic = analyze_communities_behavioral(
                        behavioral_df, communities, labels, list(G.nodes())
                    )
                else:
                    from utils.semantic_analysis import analyze_communities as analyze_semantic
                    semantic = analyze_semantic(communities, node_texts)
                    
                st.session_state["algo_results"] = {
                    "algorithm": algorithm, "labels": labels, "mod": mod,
                    "communities": communities, "summary": summary, "semantic": semantic,
                }

# ════════════════════════════════════════════
# DISPLAY RESULTS
# ════════════════════════════════════════════
if st.session_state["algo_results"]:
    res         = st.session_state["algo_results"]
    alg         = res["algorithm"]
    labels      = res["labels"]
    communities = res["communities"]
    summary     = res["summary"]
    semantic    = res["semantic"]
    mod         = res["mod"]

    tabs = st.tabs(["🕸️ Graph", "📊 Metrics", "🔍 Community Analysis", "📥 Export"])

    # ── TAB 1 ──
    with tabs[0]:
        if data_format == "behavioral" and behavioral_df is not None:
            community_names = [f"C{s['community_id']} - {s['learner_type']}" for s in semantic]
        else:
            community_names = [s["label"] for s in semantic]
            
        fig_graph = plot_graph_plotly(
            G, labels,
            title=f"{alg} Communities",
            layout=layout,
            community_names=community_names,
        )
        st.plotly_chart(fig_graph, use_container_width=True)
        st.plotly_chart(plot_community_sizes(communities, alg), use_container_width=True)

    # ── TAB 2 ──
    with tabs[1]:
        st.markdown(f"### {alg} — Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Modularity",    f"{mod:.4f}")
        c2.metric("Communities",   summary["num_communities"])
        c3.metric("Largest Comm.", summary["largest_community"])
        c4.metric("Smallest",      summary["smallest_community"])
        c5.metric("Avg Size",      f"{summary['avg_community_size']:.1f}")
        st.markdown("---")

        if data_format == "behavioral" and behavioral_df is not None:
            table_data = [{
                "ID":           s["community_id"],
                "Learner Type": s["learner_type"],
                "Size":         s["size"],
                "Avg Actions":  s["avg_total_actions"],
                "Avg Gap":      f"{s['avg_avg_gap']:.1f}",
                "Avg Duration": f"{s['avg_duration']:.1f}",
                "Avg Targets":  s["avg_unique_targets"],
            } for s in semantic]
        else:
            table_data = [{
                "ID":           s["community_id"],
                "Label":        s["label"],
                "Size":         s["size"],
                "Topic":        s["topic"],
                "Top Keywords": ", ".join([kw for kw, _ in s["keywords"][:5]]) or "—",
            } for s in semantic]
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    # ── TAB 3 ──
    with tabs[2]:
        if data_format == "behavioral" and behavioral_df is not None:
            st.markdown("### Community Behavioral Analysis")
            st.markdown("""
            **Logic for Behavioral Classification (Assign Meaning - NO GUESSING):**
            - **Active Learners:** high total_actions, low avg_gap, high duration, high unique_targets
            - **Inactive / Dead Learners:** low total_actions, high avg_gap, low duration
            - **Average Learners:** in between everything
            
            :warning: **IMPORTANT:** Do NOT assume cluster 0 = Active. That's wrong unless data proves it.
            """)
            
            stats = calculate_learner_statistics(behavioral_df)
            
            st.markdown("#### Global Learner Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Actions", f"{stats['total_actions']['mean']:.1f}")
            col2.metric("Avg Gap", f"{stats['avg_gap']['mean']:.1f}")
            col3.metric("Avg Duration", f"{stats['duration']['mean']:.1f}")
            col4.metric("Avg Targets", f"{stats['unique_targets']['mean']:.1f}")
            
            st.markdown("---")
            st.markdown("#### Learner Type Distribution")
            st.plotly_chart(plot_learner_distribution_pie(behavioral_df), use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### User Behavior Analysis")
            st.plotly_chart(plot_user_behavior_scatter(behavioral_df), use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### Community Behavioral Heatmap")
            st.plotly_chart(plot_behavioral_heatmap(semantic), use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### Community Behavioral Profiles")
            
            for s in semantic:
                learner_type = s.get("learner_type", "Unknown")
                
                if "Active" in learner_type:
                    badge_color = "#00e5ff"
                    emoji = "✅"
                elif "Inactive" in learner_type:
                    badge_color = "#ff6b9d"
                    emoji = "❌"
                else:
                    badge_color = "#ffd166"
                    emoji = "⚡"
                
                with st.expander(f"{emoji} Community {s['community_id']} — {learner_type} ({s['size']} nodes)", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg Total Actions", f"{s['avg_total_actions']}")
                    c2.metric("Avg Gap", f"{s['avg_avg_gap']:.1f}")
                    c3.metric("Avg Duration", f"{s['avg_duration']:.1f}")
                    
                    c4, c5, c6 = st.columns(3)
                    c4.metric("Avg Unique Targets", f"{s['avg_unique_targets']}")
                    c5.markdown(f"**Description:** {s.get('description', 'N/A')}")
                    
                    st.markdown(f"<span style='background:{badge_color}20;color:{badge_color};padding:4px 12px;border-radius:20px;font-size:0.8rem'>{learner_type}</span>", unsafe_allow_html=True)
            
        elif node_texts:
            st.markdown("### Community Semantic Profiles")
            for s in semantic:
                with st.expander(f"Community {s['community_id']} — {s['label']} ({s['size']} nodes)", expanded=False):
                    cl, cr = st.columns([1, 2])
                    with cl:
                        st.markdown(f"**Topic:** `{s['topic']}`")
                        st.markdown(f"**Nodes:** {s['size']}")
                        st.markdown(f"**Sample:** {', '.join(str(n) for n in s['nodes'][:8])}")
                    with cr:
                        if s["keywords"]:
                            pills = " ".join([f'<span class="kw-pill">{kw} ({sc:.3f})</span>' for kw, sc in s["keywords"][:10]])
                            st.markdown(pills, unsafe_allow_html=True)
                            try:
                                from utils.visualization import generate_wordcloud_image
                                wc_img = generate_wordcloud_image(s["keywords"])
                                if wc_img:
                                    st.image(wc_img, use_column_width=True)
                            except:
                                pass
        else:
            st.info("Add a `text` column or behavioral data to enable analysis.")

    # ── TAB 4 ──
    with tabs[3]:
        st.markdown("### Export")
        ce1, ce2 = st.columns(2)
        with ce1:
            asgn_df = pd.DataFrame({"node": list(G.nodes()), "community": labels})
            st.download_button("⬇️ Community Assignments (CSV)",
                               data=asgn_df.to_csv(index=False),
                               file_name=f"{alg.lower().replace(' ','_')}_communities.csv",
                               mime="text/csv", use_container_width=True)
        with ce2:
            met_df = pd.DataFrame([{
                "algorithm": alg, "modularity": mod,
                "num_communities": summary["num_communities"],
                "largest_community": summary["largest_community"],
                "avg_community_size": summary["avg_community_size"],
                "graph_density": gstats["density"],
                "avg_clustering": gstats["avg_clustering"],
            }])
            st.download_button("⬇️ Metrics Summary (CSV)",
                               data=met_df.to_csv(index=False),
                               file_name=f"{alg.lower().replace(' ','_')}_metrics.csv",
                               mime="text/csv", use_container_width=True)

# ════════════════════════════════════════════
# COMPARE ALL  (each algo cached independently)
# ════════════════════════════════════════════
if compare_btn:
    st.markdown("---")
    st.markdown('<div class="hero-title" style="font-size:1.2rem;margin-bottom:1rem">⚡ Algorithm Comparison</div>', unsafe_allow_html=True)

    all_labels      = {}
    all_mods        = {}
    all_communities = {}

    prog = st.progress(0, text="Starting...")
    algo_names = list(ALGORITHMS.keys())

    for idx, aname in enumerate(algo_names):
        prog.progress(idx / len(algo_names), text=f"Running {aname}...")
        try:
            lbl, mv, comms = detect_communities(G, aname, cache_key=df_hash)
            all_labels[aname]      = lbl
            all_mods[aname]        = mv
            all_communities[aname] = comms
        except Exception as e:
            st.warning(f"⚠️ {aname} failed: {e}")

    prog.progress(1.0, text="✅ Done!")

    st.session_state["compare_results"] = {
        "labels": all_labels, "mods": all_mods, "communities": all_communities
    }

if st.session_state["compare_results"]:
    cr      = st.session_state["compare_results"]
    al      = cr["labels"]
    am      = cr["mods"]
    ac      = cr["communities"]

    st.plotly_chart(plot_modularity_bar(am), use_container_width=True)

    comp_rows = []
    for name in al:
        sizes = [len(c) for c in ac[name]]
        comp_rows.append({
            "Algorithm":   name,
            "Modularity":  round(am[name], 4),
            "Communities": len(ac[name]),
            "Largest":     max(sizes),
            "Avg Size":    round(float(np.mean(sizes)), 1),
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

    if len(al) >= 2:
        st.markdown("#### Pairwise Similarity")
        st.dataframe(pd.DataFrame(pairwise_comparison(al)), use_container_width=True)

    st.markdown("#### Side-by-Side Graphs")
    gcols = st.columns(2)
    for i, (name, lbl) in enumerate(al.items()):
        with gcols[i % 2]:
            if data_format == "behavioral" and behavioral_df is not None:
                compare_semantic = analyze_communities_behavioral(
                    behavioral_df, ac[name], lbl, list(G.nodes())
                )
                compare_names = [f"C{s['community_id']} - {s['learner_type']}" for s in compare_semantic]
            else:
                compare_names = None
            st.plotly_chart(plot_graph_plotly(G, lbl, title=name, layout=layout, community_names=compare_names), use_container_width=True)

    st.download_button("⬇️ Download Comparison CSV",
                       data=pd.DataFrame(comp_rows).to_csv(index=False),
                       file_name="algorithm_comparison.csv", mime="text/csv")