"""Command-line interface for the 5G slicing simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

from slicing_simulator.config import RANDOM_SEED
from slicing_simulator.data import generate_traffic
from slicing_simulator.metrics import compare_distribution_methods
from slicing_simulator.ml import train_regression_model
from slicing_simulator.plots import (
    create_app_type_distribution_figure,
    create_bandwidth_distribution_figure,
    create_correlation_matrix_figure,
    create_kmeans_figure,
    create_latency_distribution_figure,
    create_method_comparison_figure,
    create_regression_prediction_figure,
    save_figure,
)


def run_experiment(output_dir: str = "results", n_users: int = 500, seed: int = RANDOM_SEED) -> Path:
    """Run a full simulation experiment and save results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = generate_traffic(n_users=n_users, seed=seed)
    comparison, detailed_metrics = compare_distribution_methods(data, seed=seed)
    regression_result = train_regression_model(data, seed=seed)

    data.to_csv(output_path / "generated_users.csv", index=False, encoding="utf-8")
    comparison.to_csv(output_path / "method_comparison.csv", index=False, encoding="utf-8")

    for method_key, metrics in detailed_metrics.items():
        metrics.to_csv(output_path / f"slice_metrics_{method_key}.csv", index=False, encoding="utf-8")

    save_figure(create_method_comparison_figure(comparison), output_path / "figure_method_comparison.png")
    save_figure(create_bandwidth_distribution_figure(data), output_path / "figure_bandwidth_distribution.png")
    save_figure(create_latency_distribution_figure(data), output_path / "figure_latency_distribution.png")
    save_figure(create_app_type_distribution_figure(data), output_path / "figure_app_type_distribution.png")
    save_figure(create_kmeans_figure(data, seed=seed), output_path / "figure_kmeans.png")
    save_figure(
        create_regression_prediction_figure(regression_result),
        output_path / "figure_regression_prediction.png",
    )
    save_figure(create_correlation_matrix_figure(data), output_path / "figure_correlation_matrix.png")

    report_path = output_path / "experiment_report.txt"
    with report_path.open("w", encoding="utf-8") as file:
        file.write("5G Network Slicing Simulation Report\n")
        file.write("=" * 60 + "\n\n")
        file.write(f"Number of users: {n_users}\n")
        file.write(f"Random seed: {seed}\n\n")
        file.write("Method comparison:\n")
        file.write(comparison.to_string(index=False))
        file.write("\n\nLinear Regression:\n")
        file.write(f"MSE = {regression_result['mse']:.4f}\n")
        file.write(f"R^2 = {regression_result['r2']:.4f}\n")
        file.write(f"Intercept = {regression_result['intercept']:.4f}\n")
        file.write(f"Coefficients = {regression_result['coefficients']}\n")

    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run a 5G network slicing simulation experiment.")
    parser.add_argument("--users", type=int, default=500, help="Number of synthetic users to generate.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default="results", help="Output directory for reports and charts.")
    return parser.parse_args()


def main() -> None:
    """Main CLI entry point."""
    args = parse_args()
    output_path = run_experiment(output_dir=args.output, n_users=args.users, seed=args.seed)
    print(f"Experiment completed. Results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
