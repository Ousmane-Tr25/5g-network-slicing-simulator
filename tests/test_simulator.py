"""Basic tests for the 5G network slicing simulator."""

import pandas as pd

from slicing_simulator.assignment import assign_kmeans, assign_latency_based, assign_random
from slicing_simulator.data import generate_traffic
from slicing_simulator.metrics import calculate_slice_metrics, compare_distribution_methods
from slicing_simulator.ml import train_regression_model


def test_generate_traffic_shape_and_columns():
    data = generate_traffic(n_users=30, seed=42)
    assert len(data) == 30
    assert {"user_id", "B_r", "L_r", "T_a", "application"}.issubset(data.columns)
    assert data["B_r"].min() >= 1.0
    assert data["L_r"].min() >= 10.0


def test_latency_assignment_values():
    data = pd.DataFrame({"L_r": [20.0, 45.0, 80.0], "B_r": [5.0, 5.0, 5.0]})
    assignments = assign_latency_based(data)
    assert assignments.tolist() == [0, 1, 2]


def test_metrics_have_three_slices():
    data = generate_traffic(n_users=60, seed=42)
    assignments = assign_random(data, seed=42)
    metrics = calculate_slice_metrics(data, assignments)
    assert len(metrics) == 3
    assert {"slice_id", "load_mbps", "utilization_percent", "overloaded"}.issubset(metrics.columns)


def test_kmeans_assignment_length():
    data = generate_traffic(n_users=90, seed=42)
    assignments, _, _ = assign_kmeans(data, seed=42)
    assert len(assignments) == len(data)
    assert set(assignments.unique()).issubset({0, 1, 2})


def test_method_comparison_contains_three_methods():
    data = generate_traffic(n_users=90, seed=42)
    comparison, detailed = compare_distribution_methods(data, seed=42)
    assert len(comparison) == 3
    assert set(detailed.keys()) == {"latency", "kmeans", "random"}


def test_regression_model_outputs_metrics():
    data = generate_traffic(n_users=120, seed=42)
    result = train_regression_model(data, seed=42)
    assert "mse" in result
    assert "r2" in result
    assert result["mse"] >= 0.0
