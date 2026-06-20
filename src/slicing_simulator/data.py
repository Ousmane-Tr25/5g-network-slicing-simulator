"""Synthetic 5G traffic generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from slicing_simulator.config import APP_TYPES, DEFAULT_USERS, RANDOM_SEED


def generate_traffic(n_users: int = DEFAULT_USERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic 5G user traffic.

    Each user is described by:
    - B_r: required bandwidth in Mbps;
    - L_r: allowed latency in ms;
    - T_a: application type.

    T_a values:
    - 0: IoT;
    - 1: video;
    - 2: interactive applications.
    """
    if n_users <= 0:
        raise ValueError("n_users must be a positive integer")

    rng = np.random.default_rng(seed)
    user_ids = np.arange(1, n_users + 1)

    # Balanced application classes make method comparison easier to interpret.
    base_types = np.array([0, 1, 2] * (n_users // 3))
    remainder = n_users - len(base_types)
    if remainder > 0:
        extra_types = rng.choice([0, 1, 2], size=remainder, replace=True)
        app_types = np.concatenate([base_types, extra_types])
    else:
        app_types = base_types
    rng.shuffle(app_types)

    br_values = np.zeros(n_users)
    lr_values = np.zeros(n_users)

    for i, app_type in enumerate(app_types):
        if app_type == 0:
            # IoT: low bandwidth, high allowed latency.
            br_values[i] = rng.uniform(1.0, 4.0)
            lr_values[i] = rng.uniform(60.0, 100.0)
        elif app_type == 1:
            # Video: high bandwidth, medium latency.
            br_values[i] = rng.uniform(6.0, 10.0)
            lr_values[i] = rng.uniform(30.0, 70.0)
        else:
            # Interactive: medium/high bandwidth, low latency.
            br_values[i] = rng.uniform(3.0, 8.0)
            lr_values[i] = rng.uniform(10.0, 40.0)

    data = pd.DataFrame(
        {
            "user_id": user_ids,
            "B_r": np.round(br_values, 3),
            "L_r": np.round(lr_values, 3),
            "T_a": app_types,
        }
    )
    data["application"] = data["T_a"].map(APP_TYPES)
    return data
