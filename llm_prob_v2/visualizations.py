# visualizations.py
# UMAP visualization for semantic analysis of generated sequences

import plotly.graph_objects as go
from typing import List, Dict


def create_umap_scatter(umap_points: List[Dict]) -> str:
    """
    Create an interactive 2D UMAP scatter plot.
    
    Args:
        umap_points: List of dicts with keys: sequence_id, text, x, y
    
    Returns:
        JSON string of the Plotly figure
    """
    # Create scatter plot directly from data
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[p["x"] for p in umap_points],
        y=[p["y"] for p in umap_points],
        mode='markers+text',
        marker=dict(
            size=12,
            color='green',  # Color by sequence ID
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sequence ID"),
            line=dict(width=1, color='white')
        ),
        text=[f"Seq {p['sequence_id']}" for p in umap_points],  # Label each point
        textposition="top center",
        textfont=dict(size=9),
        hovertemplate='<b>Sequence %{text}</b><br>' +
                      'X: %{x:.3f}<br>' +
                      'Y: %{y:.3f}<br>' +
                      '<b>Text:</b> %{customdata}<br>' +
                      '<extra></extra>',
        customdata=[p["text"] for p in umap_points],
        name='Sequences'
    ))
    
    # Layout
    fig.update_layout(
        title='UMAP: Semantic Similarity of Generated Sequences',
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        hovermode='closest',
        template='plotly_white',
        width=800,
        height=600,
        showlegend=False
    )
    
    return fig.to_json()
