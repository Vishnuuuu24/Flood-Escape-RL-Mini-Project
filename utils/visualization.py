"""Visualization utilities for Flood Escape RL experiments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

Numeric: TypeAlias = float | int
MetricsByAlgorithm: TypeAlias = Mapping[str, Mapping[str, Sequence[Numeric]]]
StateLike: TypeAlias = tuple[tuple[int, int], bytes]
TableLike: TypeAlias = Mapping[StateLike, Numeric | np.ndarray]
QTableLike: TypeAlias = Mapping[StateLike, np.ndarray]


def _ensure_plots_dir(output_dir: str | Path | None = None) -> Path:
    """Return writable plots directory, creating it if necessary."""
    if output_dir is None:
        resolved_dir = Path("results") / "plots"
    else:
        resolved_dir = Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    return resolved_dir


def _ensure_parent_dir(output_path: str | Path) -> Path:
    """Return output path after ensuring parent directory exists."""
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path


def _running_mean(values: Sequence[Numeric], window_size: int) -> np.ndarray:
    """Compute a causal running mean for 1D values."""
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if series.size == 0:
        return series

    window = max(1, min(int(window_size), int(series.size)))
    cumulative_sum = np.cumsum(np.insert(series, 0, 0.0))
    smoothed = np.empty_like(series)

    for index in range(series.size):
        start_index = max(0, index - window + 1)
        count = index - start_index + 1
        smoothed[index] = (cumulative_sum[index + 1] - cumulative_sum[start_index]) / count

    return smoothed


def _extract_position(state: StateLike) -> tuple[int, int] | None:
    """Extract (x, y) from a state key safely."""
    try:
        x, y = state[0]
    except (TypeError, ValueError, IndexError):
        return None
    return int(x), int(y)


def _aggregate_scalar_values(table: TableLike, grid_size: int) -> dict[tuple[int, int], float]:
    """Aggregate values by (x, y) using mean across repeated states."""
    grouped_values: dict[tuple[int, int], list[float]] = defaultdict(list)

    for state, value in table.items():
        position = _extract_position(state)
        if position is None:
            continue

        x, y = position
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            continue

        if isinstance(value, np.ndarray):
            q_values = np.asarray(value, dtype=np.float64).reshape(-1)
            if q_values.size == 0 or not np.all(np.isfinite(q_values)):
                continue
            scalar = float(np.max(q_values))
        else:
            scalar = float(value)
            if not np.isfinite(scalar):
                continue

        grouped_values[(x, y)].append(scalar)

    return {
        position: float(np.mean(values))
        for position, values in grouped_values.items()
        if len(values) > 0
    }


def _aggregate_q_values(q_table: QTableLike, grid_size: int) -> dict[tuple[int, int], np.ndarray]:
    """Aggregate Q-vectors by (x, y) using mean across repeated states."""
    grouped: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)

    for state, q_values in q_table.items():
        position = _extract_position(state)
        if position is None:
            continue

        x, y = position
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            continue

        q_array = np.asarray(q_values, dtype=np.float64).reshape(-1)
        if q_array.size == 0 or not np.all(np.isfinite(q_array)):
            continue

        existing = grouped.get((x, y))
        if existing and existing[0].shape[0] != q_array.shape[0]:
            continue

        grouped[(x, y)].append(q_array)

    aggregated: dict[tuple[int, int], np.ndarray] = {}
    for position, q_rows in grouped.items():
        if q_rows:
            aggregated[position] = np.mean(np.vstack(q_rows), axis=0)
    return aggregated


def plot_learning_curves(
    metrics_by_algo: MetricsByAlgorithm,
    output_dir: str | Path,
    smooth_window: int,
) -> Path:
    """Compare algorithms with smoothed reward and success-rate curves."""
    if not metrics_by_algo:
        raise ValueError("metrics_by_algo must not be empty.")

    target_dir = _ensure_plots_dir(output_dir)
    output_path = target_dir / "learning_curves.png"

    sns.set_theme(style="whitegrid")
    figure, (ax_reward, ax_success) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for algo_name, metrics in metrics_by_algo.items():
        rewards = metrics.get("reward_per_episode", ())
        successes = metrics.get("success_per_episode", ())

        smoothed_rewards = _running_mean(rewards, smooth_window)
        smoothed_success = _running_mean(successes, smooth_window)

        reward_x = np.arange(1, smoothed_rewards.size + 1)
        success_x = np.arange(1, smoothed_success.size + 1)

        ax_reward.plot(reward_x, smoothed_rewards, label=algo_name, linewidth=2)
        ax_success.plot(success_x, smoothed_success, label=algo_name, linewidth=2)

    ax_reward.set_title("Smoothed Episode Reward")
    ax_reward.set_ylabel("Reward")
    ax_reward.legend(loc="best")

    ax_success.set_title("Smoothed Success Rate")
    ax_success.set_xlabel("Episode")
    ax_success.set_ylabel("Success")
    ax_success.set_ylim(-0.05, 1.05)
    ax_success.legend(loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_value_heatmap(
    table: TableLike,
    output_path: str | Path,
    grid_size: int = 6,
    title: str = "State Value Heatmap",
) -> Path:
    """Plot a value heatmap from V-table or Q-table, aggregated on (x, y)."""
    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")

    resolved_output_path = _ensure_parent_dir(output_path)
    aggregated_values = _aggregate_scalar_values(table, grid_size)

    heatmap_data = np.full((grid_size, grid_size), np.nan, dtype=np.float64)
    for (x, y), value in aggregated_values.items():
        heatmap_data[x, y] = value

    all_nan = np.isnan(heatmap_data).all()
    if all_nan:
        heatmap_data = np.zeros((grid_size, grid_size), dtype=np.float64)

    sns.set_theme(style="white")
    figure, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heatmap_data,
        mask=np.isnan(heatmap_data),
        cmap="viridis",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Value"},
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("y")
    ax.set_ylabel("x")

    figure.tight_layout()
    figure.savefig(resolved_output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return resolved_output_path


def plot_policy(
    q_table: QTableLike,
    output_path: str | Path,
    grid_size: int = 6,
    title: str = "Greedy Policy",
) -> Path:
    """Plot greedy action arrows from argmax Q-values, aggregated on (x, y)."""
    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")

    resolved_output_path = _ensure_parent_dir(output_path)
    aggregated_q = _aggregate_q_values(q_table, grid_size)

    row_indices, col_indices = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    u = np.zeros((grid_size, grid_size), dtype=np.float64)
    v = np.zeros((grid_size, grid_size), dtype=np.float64)

    action_to_delta = {
        0: (0.0, -1.0),  # up
        1: (0.0, 1.0),  # down
        2: (-1.0, 0.0),  # left
        3: (1.0, 0.0),  # right
    }

    for x in range(grid_size):
        for y in range(grid_size):
            q_values = aggregated_q.get((x, y))
            if q_values is None or q_values.size < 4:
                continue

            best_action = int(np.argmax(q_values[:4]))
            dx, dy = action_to_delta[best_action]
            u[x, y] = dx
            v[x, y] = dy

    sns.set_theme(style="ticks")
    figure, ax = plt.subplots(figsize=(8, 8))
    ax.quiver(
        col_indices,
        row_indices,
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=1.7,
        color="#1f77b4",
        width=0.006,
    )

    ax.set_title(title)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.5, alpha=0.4)

    figure.tight_layout()
    figure.savefig(resolved_output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return resolved_output_path
