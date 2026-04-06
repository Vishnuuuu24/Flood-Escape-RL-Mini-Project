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
AlgorithmMetrics: TypeAlias = Mapping[str, Sequence[Numeric]]
MetricsByAlgorithm: TypeAlias = Mapping[str, Mapping[str, Sequence[Numeric]]]
StateLike: TypeAlias = tuple[tuple[int, int], bytes]
TableLike: TypeAlias = Mapping[StateLike, Numeric | np.ndarray]
QTableLike: TypeAlias = Mapping[StateLike, np.ndarray]
PolicyByPosition: TypeAlias = Mapping[tuple[int, int], int]
RolloutRecord: TypeAlias = Mapping[str, object]

ACTION_LABELS = {
    0: "U",
    1: "D",
    2: "L",
    3: "R",
}


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


def plot_algorithm_learning_curves(
    algo_name: str,
    metrics: AlgorithmMetrics,
    output_dir: str | Path,
    smooth_window: int,
) -> Path:
    """Plot detailed individual learning curves for a single algorithm."""
    target_dir = _ensure_plots_dir(output_dir)
    output_path = target_dir / f"learning_{algo_name.lower()}.png"

    rewards = np.asarray(metrics.get("reward_per_episode", ()), dtype=np.float64)
    successes = np.asarray(metrics.get("success_per_episode", ()), dtype=np.float64)
    steps = np.asarray(metrics.get("steps_per_episode", ()), dtype=np.float64)

    if rewards.size == 0:
        raise ValueError(f"No reward data available for algorithm '{algo_name}'.")

    rewards_smoothed = _running_mean(rewards, smooth_window)
    success_smoothed = _running_mean(successes, smooth_window)
    steps_smoothed = _running_mean(steps, smooth_window)

    x_values = np.arange(1, rewards.size + 1)

    sns.set_theme(style="whitegrid")
    figure, (ax_reward, ax_success, ax_steps) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    ax_reward.plot(x_values, rewards, alpha=0.25, linewidth=1.0, label="Raw")
    ax_reward.plot(x_values, rewards_smoothed, linewidth=2.0, label="Smoothed")
    ax_reward.set_title(f"{algo_name}: Reward per Episode")
    ax_reward.set_ylabel("Reward")
    ax_reward.legend(loc="best")

    ax_success.step(x_values, successes, where="mid", alpha=0.25, linewidth=1.0, label="Raw")
    ax_success.plot(x_values, success_smoothed, linewidth=2.0, label="Smoothed")
    ax_success.set_title(f"{algo_name}: Success per Episode")
    ax_success.set_ylabel("Success")
    ax_success.set_ylim(-0.05, 1.05)
    ax_success.legend(loc="best")

    ax_steps.plot(x_values, steps, alpha=0.25, linewidth=1.0, label="Raw")
    ax_steps.plot(x_values, steps_smoothed, linewidth=2.0, label="Smoothed")
    ax_steps.set_title(f"{algo_name}: Steps per Episode")
    ax_steps.set_xlabel("Episode")
    ax_steps.set_ylabel("Steps")
    ax_steps.legend(loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_steps_comparison(
    metrics_by_algo: MetricsByAlgorithm,
    output_dir: str | Path,
    smooth_window: int,
) -> Path:
    """Plot smoothed steps-per-episode comparison across algorithms."""
    if not metrics_by_algo:
        raise ValueError("metrics_by_algo must not be empty.")

    target_dir = _ensure_plots_dir(output_dir)
    output_path = target_dir / "steps_comparison.png"

    sns.set_theme(style="whitegrid")
    figure, ax = plt.subplots(figsize=(12, 5))

    for algo_name, metrics in metrics_by_algo.items():
        steps = metrics.get("steps_per_episode", ())
        steps_smoothed = _running_mean(steps, smooth_window)
        x_values = np.arange(1, steps_smoothed.size + 1)
        ax.plot(x_values, steps_smoothed, linewidth=2.0, label=algo_name)

    ax.set_title("Smoothed Steps per Episode (All Algorithms)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend(loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_summary_metrics(
    metrics_by_algo: MetricsByAlgorithm,
    output_dir: str | Path,
    tail_fraction: float = 0.20,
) -> Path:
    """Plot compact bar summaries over the final fraction of episodes."""
    if not metrics_by_algo:
        raise ValueError("metrics_by_algo must not be empty.")

    target_dir = _ensure_plots_dir(output_dir)
    output_path = target_dir / "summary_metrics.png"

    names = list(metrics_by_algo.keys())
    avg_reward: list[float] = []
    avg_success: list[float] = []
    avg_steps: list[float] = []

    for algo_name in names:
        metrics = metrics_by_algo[algo_name]

        rewards = np.asarray(metrics.get("reward_per_episode", ()), dtype=np.float64)
        successes = np.asarray(metrics.get("success_per_episode", ()), dtype=np.float64)
        steps = np.asarray(metrics.get("steps_per_episode", ()), dtype=np.float64)

        window = max(1, int(np.ceil(rewards.size * tail_fraction)))

        avg_reward.append(float(np.mean(rewards[-window:])))
        avg_success.append(float(np.mean(successes[-window:])))
        avg_steps.append(float(np.mean(steps[-window:])))

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(14, 5))
    positions = np.arange(len(names))

    axes[0].bar(positions, avg_reward, color="#1f77b4")
    axes[0].set_title("Avg Reward (Final 20%)")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(names, rotation=20, ha="right")

    axes[1].bar(positions, avg_success, color="#2ca02c")
    axes[1].set_title("Avg Success (Final 20%)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(names, rotation=20, ha="right")

    axes[2].bar(positions, avg_steps, color="#ff7f0e")
    axes[2].set_title("Avg Steps (Final 20%)")
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(names, rotation=20, ha="right")

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


def plot_policy_grid_image(
    algo_name: str,
    policy: PolicyByPosition,
    output_path: str | Path,
    grid_size: int = 6,
    goal_position: tuple[int, int] = (5, 5),
) -> Path:
    """Render a terminal-style policy grid as a clean image artifact."""
    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")

    resolved_output_path = _ensure_parent_dir(output_path)

    grid_symbols = np.full((grid_size, grid_size), ".", dtype="<U1")
    for (x, y), action in policy.items():
        if 0 <= x < grid_size and 0 <= y < grid_size:
            grid_symbols[x, y] = ACTION_LABELS.get(int(action), ".")

    gx, gy = goal_position
    if 0 <= gx < grid_size and 0 <= gy < grid_size:
        grid_symbols[gx, gy] = "G"

    figure, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.axis("off")

    table = ax.table(
        cellText=grid_symbols,
        cellLoc="center",
        loc="center",
        colLabels=[f"y{idx}" for idx in range(grid_size)],
        rowLabels=[f"x{idx}" for idx in range(grid_size)],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.1, 1.55)

    for x in range(grid_size):
        for y in range(grid_size):
            cell = table[(x + 1, y)]
            symbol = grid_symbols[x, y]

            if symbol == "G":
                cell.set_facecolor("#ffe082")
            elif symbol == ".":
                cell.set_facecolor("#f5f5f5")
            else:
                cell.set_facecolor("#e8f4fd")

    ax.set_title(
        f"{algo_name} Policy Grid\nLegend: U=Up, D=Down, L=Left, R=Right, G=Goal, .=Unknown",
        fontsize=11,
        pad=16,
    )

    figure.tight_layout()
    figure.savefig(resolved_output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return resolved_output_path


def plot_environment_rollouts(
    rollouts_by_algo: Mapping[str, RolloutRecord],
    output_path: str | Path,
    grid_size: int = 6,
    goal_position: tuple[int, int] = (5, 5),
) -> Path:
    """Visualize environment rollouts for all algorithms on one panel."""
    if not rollouts_by_algo:
        raise ValueError("rollouts_by_algo must not be empty.")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")

    resolved_output_path = _ensure_parent_dir(output_path)
    algo_names = list(rollouts_by_algo.keys())
    n_plots = len(algo_names)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))

    figure, axes = plt.subplots(n_rows, n_cols, figsize=(11, 4.8 * n_rows))
    axes_array = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for index, algo_name in enumerate(algo_names):
        row = index // n_cols
        col = index % n_cols
        ax = axes_array[row, col]

        rollout = rollouts_by_algo[algo_name]
        path = np.asarray(rollout.get("path", []), dtype=np.int64)
        flood_map = np.asarray(rollout.get("flood_map", np.zeros((grid_size, grid_size))), dtype=np.float64)
        success = bool(rollout.get("success", False))
        steps = int(rollout.get("steps", 0))

        if flood_map.shape != (grid_size, grid_size):
            flood_map = np.zeros((grid_size, grid_size), dtype=np.float64)

        ax.imshow(flood_map, cmap="Blues", vmin=0.0, vmax=1.0, alpha=0.38)

        ax.set_xticks(np.arange(grid_size))
        ax.set_yticks(np.arange(grid_size))
        ax.set_xticklabels([f"y{idx}" for idx in range(grid_size)], fontsize=8)
        ax.set_yticklabels([f"x{idx}" for idx in range(grid_size)], fontsize=8)
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)
        ax.grid(True, color="lightgray", linewidth=0.6)

        gy, gx = goal_position[1], goal_position[0]
        ax.scatter(gy, gx, marker="*", s=180, color="#f9a825", edgecolor="black", linewidth=0.6, zorder=4)

        if path.ndim == 2 and path.shape[0] > 0 and path.shape[1] >= 2:
            path_y = path[:, 1]
            path_x = path[:, 0]

            ax.plot(path_y, path_x, color="#263238", linewidth=1.8, alpha=0.9, zorder=3)
            ax.scatter(path_y[0], path_x[0], marker="o", s=60, color="#2e7d32", zorder=5, label="Start")
            end_color = "#2e7d32" if success else "#c62828"
            ax.scatter(path_y[-1], path_x[-1], marker="X", s=70, color=end_color, zorder=5, label="End")

        status_text = "Success" if success else "Fail/Timeout"
        ax.set_title(f"{algo_name}: {status_text} | Steps={steps}", fontsize=10)

    for index in range(n_plots, n_rows * n_cols):
        row = index // n_cols
        col = index % n_cols
        axes_array[row, col].axis("off")

    figure.suptitle("Environment Rollout Overview (Greedy Policy)", fontsize=13, y=0.995)
    figure.tight_layout()
    figure.savefig(resolved_output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return resolved_output_path
