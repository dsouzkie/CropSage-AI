"""
CropSage - Visualization Utilities
Charts and visual elements for the Streamlit app.
"""

import logging
from typing import List, Tuple

import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def create_confidence_chart(predictions: List[Tuple[str, float]]) -> go.Figure:
    """
    Create a horizontal bar chart showing prediction confidence scores.

    Args:
        predictions: List of (class_name, confidence) tuples

    Returns:
        Plotly Figure object
    """
    from utils.prediction import format_class_name

    # Reverse so highest confidence is at top
    names = [format_class_name(name) for name, _ in reversed(predictions)]
    confidences = [conf * 100 for _, conf in reversed(predictions)]

    # Color: green for healthy, orange/red for diseases
    colors = []
    for name, _ in reversed(predictions):
        if "healthy" in name.lower():
            colors.append("#2ecc71")  # Green
        else:
            colors.append("#e74c3c")  # Red

    fig = go.Figure(
        go.Bar(
            x=confidences,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{c:.1f}%" for c in confidences],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Confidence (%)",
        xaxis=dict(range=[0, 100]),
        height=max(200, len(predictions) * 60),
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(size=14),
    )

    return fig


def get_severity_color(confidence: float) -> str:
    """
    Return a color based on prediction confidence.

    Args:
        confidence: Prediction confidence (0.0 to 1.0)

    Returns:
        Hex color string
    """
    if confidence >= 0.90:
        return "#e74c3c"  # Red — high confidence disease
    elif confidence >= 0.70:
        return "#f39c12"  # Orange — moderate confidence
    else:
        return "#95a5a6"  # Grey — low confidence, uncertain
