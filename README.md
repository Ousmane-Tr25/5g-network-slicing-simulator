# 5G Network Slicing Simulator

A clean Python project for simulating **5G network slicing** with different traffic types, resource allocation strategies, and indirect energy-efficiency indicators.

The project is designed as a GitHub portfolio project for Data Science / ML Engineering roles. It demonstrates data generation, clustering, regression, simulation metrics, visual analytics, testing, Docker, and reproducible experiment outputs.

## Why this project matters

Modern 5G networks serve different traffic types at the same time:

- **URLLC**: low-latency traffic for interactive and critical applications.
- **eMBB**: high-bandwidth traffic such as video and multimedia services.
- **mMTC**: massive IoT traffic with many low-bandwidth devices.

The simulator compares how different allocation methods distribute users across slices and how this affects resource utilization.

## Main features

- Synthetic 5G user traffic generation.
- User assignment to three network slices.
- Comparison of three allocation strategies:
  - latency-threshold baseline;
  - K-Means clustering;
  - random baseline.
- Slice-level metrics:
  - number of users per slice;
  - load in Mbps;
  - utilization percentage;
  - overload detection;
  - load-balance score.
- ML module:
  - K-Means for traffic grouping;
  - Linear Regression for basic bandwidth estimation.
- Reproducible charts and CSV reports.
- Unit tests and GitHub Actions workflow.
- Dockerfile for reproducible execution.

## Project structure

```text
5g-network-slicing-simulator/
├── src/slicing_simulator/
│   ├── assignment.py        # user-to-slice allocation methods
│   ├── cli.py               # command-line experiment runner
│   ├── config.py            # global constants and slice configuration
│   ├── data.py              # synthetic traffic generation
│   ├── metrics.py           # load, utilization and summary metrics
│   ├── ml.py                # regression model
│   └── plots.py             # result visualizations
├── scripts/
│   └── run_experiment.py    # simple experiment launcher
├── tests/
│   └── test_simulator.py    # basic tests
├── docs/
│   └── portfolio-card.md    # short project description for CV/LinkedIn
├── notebooks/
│   └── README.md            # notebook ideas
├── .github/workflows/
│   └── python-tests.yml     # CI tests
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1. Clone the repository
# git clone https://github.com/<your-github-username>/5g-network-slicing-simulator.git
# cd 5g-network-slicing-simulator

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Run the experiment
python -m slicing_simulator.cli --users 500 --output results
```

The command creates CSV files and PNG charts in the `results/` folder.

## Example command

```bash
python -m slicing_simulator.cli --users 500 --seed 42 --output results
```

## Output files

```text
results/
├── generated_users.csv
├── method_comparison.csv
├── slice_metrics_latency.csv
├── slice_metrics_kmeans.csv
├── slice_metrics_random.csv
├── experiment_report.txt
├── figure_method_comparison.png
├── figure_bandwidth_distribution.png
├── figure_latency_distribution.png
├── figure_app_type_distribution.png
├── figure_kmeans.png
├── figure_regression_prediction.png
└── figure_correlation_matrix.png
```

## Methods compared

### 1. Latency-threshold baseline

Users are assigned according to allowed latency:

- `L_r < 30 ms` → low-latency slice;
- `30 ms <= L_r < 60 ms` → broadband slice;
- `L_r >= 60 ms` → massive IoT slice.

### 2. K-Means allocation

Users are clustered using two features:

```text
z_u = [B_r, L_r]
```

where:

- `B_r` is required bandwidth in Mbps;
- `L_r` is allowed latency in ms.

Clusters are mapped to slices according to average latency.

### 3. Random baseline

Users are assigned randomly. This method is not practical, but it is useful as a baseline.

## Metrics

For each slice `s`, the simulator calculates:

```text
N_s = sum(B_r for users assigned to slice s)
Z_s = N_s / C_s * 100%
```

where:

- `N_s` is the slice load;
- `C_s` is slice capacity;
- `Z_s` is utilization percentage.

The simulator also computes:

- mean load;
- maximum load;
- mean utilization;
- maximum utilization;
- number of overloaded slices;
- load-balance score.

## How to run tests

```bash
pytest -q
```

## Docker usage

```bash
docker build -t 5g-network-slicing-simulator .
docker run --rm -v "$(pwd)/results:/app/results" 5g-network-slicing-simulator
```

## Portfolio value

This project shows the ability to:

- transform a research topic into clean production-style Python code;
- build a reproducible data science experiment;
- compare ML and baseline methods;
- write structured documentation;
- use Git, Docker, testing, and CI.

## Future improvements

- Add real or semi-real traffic datasets.
- Add time-series traffic forecasting.
- Add a reinforcement learning agent for dynamic slice orchestration.
- Add a direct power-consumption model for base stations and core network functions.
- Add a Streamlit dashboard.
- Integrate NS-3 simulation outputs.

## Author

Ousmane Traore

## Results preview

The repository includes a reproducible example experiment with 500 synthetic 5G users.

Run the experiment with:

python -m slicing_simulator.cli --users 500 --seed 42 --output results

### Method comparison

![Method comparison](examples/results/figure_method_comparison.png)

### K-Means traffic clustering

![K-Means clustering](examples/results/figure_kmeans.png)

### Linear regression baseline

![Regression prediction](examples/results/figure_regression_prediction.png)
