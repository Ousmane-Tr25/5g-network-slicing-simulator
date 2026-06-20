"""Machine learning utilities for the slicing simulator."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from slicing_simulator.config import RANDOM_SEED


def train_regression_model(data: pd.DataFrame, seed: int = RANDOM_SEED) -> Dict[str, object]:
    """Train a Linear Regression model to estimate required bandwidth.

    Model:
        B_r_hat = theta_0 + theta_1 * T_a + theta_2 * L_r
    """
    required = {"T_a", "L_r", "B_r"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    x = data[["T_a", "L_r"]].to_numpy()
    y = data["B_r"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=seed
    )

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return {
        "model": model,
        "X_train": x_train,
        "X_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "mse": float(mean_squared_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "intercept": float(model.intercept_),
        "coefficients": model.coef_,
    }


def predict_bandwidth(model: LinearRegression, app_type: int, latency_ms: float) -> float:
    """Predict required bandwidth for one user request."""
    prediction = model.predict(np.array([[app_type, latency_ms]]))[0]
    return float(prediction)
