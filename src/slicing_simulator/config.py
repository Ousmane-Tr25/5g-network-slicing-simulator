"""Global configuration for the 5G slicing simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

RANDOM_SEED = 42
DEFAULT_USERS = 500

APP_TYPES = {
    0: "IoT",
    1: "Video",
    2: "Interactive",
}

METHOD_NAMES = {
    "latency": "Latency threshold",
    "kmeans": "K-Means",
    "random": "Random baseline",
}


@dataclass(frozen=True)
class SliceConfig:
    """Network slice configuration."""

    slice_id: int
    name: str
    capacity_mbps: float
    description: str


SLICE_CONFIGS: Dict[int, SliceConfig] = {
    0: SliceConfig(
        slice_id=0,
        name="Low-latency slice",
        capacity_mbps=5000.0,
        description="Interactive and latency-sensitive applications",
    ),
    1: SliceConfig(
        slice_id=1,
        name="Broadband slice",
        capacity_mbps=3000.0,
        description="Video and multimedia services",
    ),
    2: SliceConfig(
        slice_id=2,
        name="Massive IoT slice",
        capacity_mbps=2000.0,
        description="IoT devices, sensors and telemetry",
    ),
}
