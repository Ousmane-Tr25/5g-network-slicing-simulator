"""User-to-slice assignment strategies."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from slicing_simulator.config import RANDOM_SEED


def assign_latency_based(data: pd.DataFrame) -> pd.Series:
    """Assign users using a latency-threshold rule.

    - L_r < 30 ms -> slice 0;
    - 30 <= L_r < 60 ms -> slice 1;
    - L_r >= 60 ms -> slice 2.
    """
    _validate_columns(data, {"L_r"})
    conditions = [
        data["L_r"] < 30.0,
        (data["L_r"] >= 30.0) & (data["L_r"] < 60.0),
        data["L_r"] >= 60.0,
    ]
    assigned = np.select(conditions, [0, 1, 2], default=2)
    return pd.Series(assigned, index=data.index, name="slice_id")


def assign_random(data: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.Series:
    """Assign users randomly to one of three slices."""
    rng = np.random.default_rng(seed)
    assigned = rng.integers(low=0, high=3, size=len(data))
    return pd.Series(assigned, index=data.index, name="slice_id")


def assign_kmeans(
    data: pd.DataFrame, seed: int = RANDOM_SEED
) -> Tuple[pd.Series, KMeans, StandardScaler]:
    """Assign users using K-Means clustering.

    The clustering features are z_u = [B_r, L_r].
    Clusters are mapped to slices using average latency:
    - lowest average latency -> slice 0;
    - medium average latency -> slice 1;
    - highest average latency -> slice 2.
    """
    _validate_columns(data, {"B_r", "L_r"})

    features = data[["B_r", "L_r"]].to_numpy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = KMeans(n_clusters=3, random_state=seed, n_init=10)
    labels = model.fit_predict(features_scaled)

    temp = data.copy()
    temp["cluster"] = labels
    cluster_latency = temp.groupby("cluster")["L_r"].mean().sort_values()
    mapping = {
        int(cluster_latency.index[0]): 0,
        int(cluster_latency.index[1]): 1,
        int(cluster_latency.index[2]): 2,
    }
    assigned = pd.Series([mapping[int(label)] for label in labels], index=data.index, name="slice_id")
    return assigned, model, scaler


def _validate_columns(data: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
