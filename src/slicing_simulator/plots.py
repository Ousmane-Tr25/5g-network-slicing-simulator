"""Plotting utilities for the 5G slicing simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from slicing_simulator.assignment import assign_kmeans
from slicing_simulator.config import APP_TYPES, RANDOM_SEED


def create_method_comparison_figure(comparison: pd.DataFrame) -> Figure:
    """Create a bar chart comparing distribution methods."""
    fig = Figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(111)
    methods = comparison["method_name"].tolist()
    values = comparison["mean_utilization_percent"].tolist()
    ax.bar(methods, values)
    ax.set_title("Comparison of resource allocation methods")
    ax.set_ylabel("Mean resource utilization, %")
    ax.set_xlabel("Allocation method")
    ax.tick_params(axis="x", rotation=15)
    for i, value in enumerate(values):
        ax.text(i, value + 0.3, f"{value:.2f}%", ha="center")
    fig.tight_layout()
    return fig


def create_bandwidth_distribution_figure(data: pd.DataFrame) -> Figure:
    """Create a histogram of required bandwidth."""
    fig = Figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(111)
    ax.hist(data["B_r"], bins=20, edgecolor="black")
    ax.set_title("Distribution of required bandwidth")
    ax.set_xlabel("B_r, Mbps")
    ax.set_ylabel("Number of users")
    fig.tight_layout()
    return fig


def create_latency_distribution_figure(data: pd.DataFrame) -> Figure:
    """Create a histogram of allowed latency."""
    fig = Figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(111)
    ax.hist(data["L_r"], bins=20, edgecolor="black")
    ax.set_title("Distribution of allowed latency")
    ax.set_xlabel("L_r, ms")
    ax.set_ylabel("Number of users")
    fig.tight_layout()
    return fig


def create_app_type_distribution_figure(data: pd.DataFrame) -> Figure:
    """Create a bar chart of application types."""
    fig = Figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(111)
    counts = data["application"].value_counts().reindex(APP_TYPES.values())
    ax.bar(counts.index.tolist(), counts.values.tolist())
    ax.set_title("Application type distribution")
    ax.set_xlabel("Application type")
    ax.set_ylabel("Number of users")
    ax.tick_params(axis="x", rotation=15)
    for i, value in enumerate(counts.values):
        ax.text(i, value + 2, str(int(value)), ha="center")
    fig.tight_layout()
    return fig


def create_kmeans_figure(data: pd.DataFrame, seed: int = RANDOM_SEED) -> Figure:
    """Create a scatter plot of K-Means clusters."""
    _, model, scaler = assign_kmeans(data, seed=seed)
    labels = model.labels_
    centers_original = scaler.inverse_transform(model.cluster_centers_)

    fig = Figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data["B_r"], data["L_r"], c=labels, alpha=0.75)
    ax.scatter(
        centers_original[:, 0],
        centers_original[:, 1],
        marker="X",
        s=200,
        edgecolor="black",
        label="Cluster centers",
    )
    ax.set_title("K-Means clustering of user traffic")
    ax.set_xlabel("B_r, Mbps")
    ax.set_ylabel("L_r, ms")
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="Cluster label")
    fig.tight_layout()
    return fig


def create_regression_prediction_figure(regression_result: Dict[str, object]) -> Figure:
    """Create a chart comparing actual and predicted bandwidth values."""
    y_test = regression_result["y_test"]
    y_pred = regression_result["y_pred"]

    fig = Figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(111)
    indices = np.arange(len(y_test))
    ax.plot(indices, y_test, marker="o", linestyle="", label="Actual values")
    ax.plot(indices, y_pred, marker="x", linestyle="", label="Predicted values")
    ax.set_title("Actual vs predicted required bandwidth")
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("B_r, Mbps")
    ax.legend()
    fig.tight_layout()
    return fig


def create_correlation_matrix_figure(data: pd.DataFrame) -> Figure:
    """Create a correlation matrix for T_a, L_r and B_r."""
    corr = data[["T_a", "L_r", "B_r"]].corr()
    labels = ["T_a", "L_r", "B_r"]

    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)
    image = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_title("Feature correlation matrix")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: Path) -> None:
    """Save a matplotlib Figure object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
