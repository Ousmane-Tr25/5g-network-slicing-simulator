"""Metrics for 5G slicing simulations."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from slicing_simulator.assignment import assign_kmeans, assign_latency_based, assign_random
from slicing_simulator.config import METHOD_NAMES, RANDOM_SEED, SLICE_CONFIGS


def calculate_slice_metrics(data: pd.DataFrame, assignments: pd.Series) -> pd.DataFrame:
    """Calculate load and utilization for each slice.

    N_s = sum B_r for users assigned to slice s.
    Z_s = N_s / C_s * 100%.
    """
    if len(data) != len(assignments):
        raise ValueError("data and assignments must have the same length")
    if "B_r" not in data.columns:
        raise ValueError("data must contain the B_r column")

    rows = []
    for slice_id, config in SLICE_CONFIGS.items():
        users_in_slice = data[assignments == slice_id]
        load = float(users_in_slice["B_r"].sum())
        capacity = config.capacity_mbps
        utilization = (load / capacity) * 100.0
        rows.append(
            {
                "slice_id": slice_id,
                "slice_name": config.name,
                "capacity_mbps": capacity,
                "users_count": int(len(users_in_slice)),
                "load_mbps": round(load, 3),
                "utilization_percent": round(utilization, 3),
                "overloaded": bool(utilization > 100.0),
            }
        )
    return pd.DataFrame(rows)


def load_balance_score(metrics: pd.DataFrame) -> float:
    """Mean absolute deviation of utilization from average utilization.

    Lower values indicate a more balanced use of slice capacities.
    """
    mean_utilization = metrics["utilization_percent"].mean()
    score = (metrics["utilization_percent"] - mean_utilization).abs().mean()
    return round(float(score), 3)


def summarize_metrics(metrics: pd.DataFrame, method_key: str) -> Dict[str, float | int | str]:
    """Create a summary row for a distribution method."""
    return {
        "method_key": method_key,
        "method_name": METHOD_NAMES[method_key],
        "mean_load_mbps": round(float(metrics["load_mbps"].mean()), 3),
        "max_load_mbps": round(float(metrics["load_mbps"].max()), 3),
        "mean_utilization_percent": round(float(metrics["utilization_percent"].mean()), 3),
        "max_utilization_percent": round(float(metrics["utilization_percent"].max()), 3),
        "balance_score": load_balance_score(metrics),
        "overloaded_count": int(metrics["overloaded"].sum()),
    }


def compare_distribution_methods(
    data: pd.DataFrame, seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Compare latency-based, K-Means and random distribution strategies."""
    results = []
    detailed_metrics: Dict[str, pd.DataFrame] = {}

    latency_assignments = assign_latency_based(data)
    latency_metrics = calculate_slice_metrics(data, latency_assignments)
    detailed_metrics["latency"] = latency_metrics
    results.append(summarize_metrics(latency_metrics, "latency"))

    kmeans_assignments, _, _ = assign_kmeans(data, seed=seed)
    kmeans_metrics = calculate_slice_metrics(data, kmeans_assignments)
    detailed_metrics["kmeans"] = kmeans_metrics
    results.append(summarize_metrics(kmeans_metrics, "kmeans"))

    random_assignments = assign_random(data, seed=seed)
    random_metrics = calculate_slice_metrics(data, random_assignments)
    detailed_metrics["random"] = random_metrics
    results.append(summarize_metrics(random_metrics, "random"))

    return pd.DataFrame(results), detailed_metrics
