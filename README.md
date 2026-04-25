# GraphMind

Graph-based community detection & semantic analysis dashboard built with Streamlit.

## Overview

GraphMind is a production-ready Streamlit application for detecting communities in graph structures and analyzing their semantic or behavioral properties. It supports multiple community detection algorithms and provides interactive visualizations.

## Features

- **Multiple Algorithms**: Louvain, Leiden, Label Propagation, Girvan-Newman
- **Semantic Analysis**: Extract topics and keywords from community text data
- **Behavioral Analysis**: Analyze learner behavior patterns (for educational datasets)
- **Interactive Visualizations**: Plotly-based graph visualizations
- **Algorithm Comparison**: Compare all algorithms side-by-side
- **Export**: Download community assignments and metrics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Input Formats

### Edge List Format
```csv
source,target
Alice,Bob
Bob,Carol
```
Optional columns: `weight`, `text`

### Behavioral Format
```csv
USERID,total_actions,avg_gap,duration,unique_targets,cluster,learner_type
0,76,31498.28,2362371.0,24,1,Active
```

## Requirements

- streamlit>=1.32.0
- networkx>=3.2
- python-louvain>=0.16
- python-igraph>=0.11.3
- leidenalg>=0.10.2
- scikit-learn>=1.4.0
- numpy>=1.26.0
- pandas>=2.2.0
- matplotlib>=3.8.0
- plotly>=5.19.0
- scipy>=1.12.0
- sentence-transformers>=2.6.0
- wordcloud>=1.9.3
- nltk>=3.8.1