"""Training entry point for Flood Escape Phase 4/5 experiments."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from algorithms.base_agent import StateKey, observation_to_state_key
from algorithms.monte_carlo import MonteCarloControl
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from algorithms.td_learning import TDPrediction
from env import FloodEscapeEnv
from utils import (
    plot_algorithm_learning_curves,
    plot_learning_curves,
    plot_policy,
    plot_steps_comparison,
    plot_summary_metrics,
    plot_value_heatmap,
)

MetricValue = float | int
Metrics = dict[str, list[MetricValue]]
MetricsByAlgorithm = dict[str, Metrics]


def _init_metrics() -> Metrics:
    return {
        "reward_per_episode": [],
        "success_per_episode": [],
        "steps_per_episode": [],
    }


def _episode_seed(base_seed: int, algo_offset: int, episode_index: int) -> int:
    return base_seed + (algo_offset * 100_000) + episode_index


def _is_success(terminated: bool, final_state: StateKey, goal_position: tuple[int, int]) -> int:
    return 1 if terminated and final_state[0] == goal_position else 0


def _td_behavior_action(
    state: StateKey,
    td_agent: TDPrediction,
    rng: np.random.Generator,
    *,
    epsilon: float,
    grid_size: int,
) -> int:
    """Epsilon-greedy behavior policy for TD(0) over one-step value lookahead."""
    if rng.random() < epsilon:
        return int(rng.integers(0, 4))

    x, y = state[0]
    flood_bytes = state[1]

    candidates = {
        0: (max(0, x - 1), y),
        1: (min(grid_size - 1, x + 1), y),
        2: (x, max(0, y - 1)),
        3: (x, min(grid_size - 1, y + 1)),
    }

    values = np.empty(4, dtype=np.float64)
    for action, next_pos in candidates.items():
        next_state: StateKey = (next_pos, flood_bytes)
        values[action] = td_agent.value(next_state)

    max_value = float(np.max(values))
    best_actions = np.flatnonzero(np.isclose(values, max_value))
    return int(best_actions[int(rng.integers(0, len(best_actions)))])


def _run_monte_carlo(episodes: int, seed: int) -> tuple[Metrics, dict[StateKey, np.ndarray]]:
    env = FloodEscapeEnv()
    agent = MonteCarloControl(n_actions=env.action_space.n, seed=seed)
    metrics = _init_metrics()

    for episode in range(episodes):
        obs, _ = env.reset(seed=_episode_seed(seed, 0, episode))
        state = observation_to_state_key(obs)
        done = False

        episode_reward = 0.0
        episode_steps = 0
        agent.start_episode()

        while not done:
            action = agent.select_action(state, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = observation_to_state_key(next_obs)

            agent.record_transition(state, action, reward)

            episode_reward += float(reward)
            episode_steps += 1
            done = bool(terminated or truncated)
            state = next_state

        agent.end_episode()
        metrics["reward_per_episode"].append(episode_reward)
        metrics["success_per_episode"].append(_is_success(bool(terminated), state, env.goal_position))
        metrics["steps_per_episode"].append(episode_steps)

    return metrics, agent.q_table


def _run_sarsa(episodes: int, seed: int) -> tuple[Metrics, dict[StateKey, np.ndarray]]:
    env = FloodEscapeEnv()
    agent = SARSAAgent(n_actions=env.action_space.n, seed=seed)
    metrics = _init_metrics()

    for episode in range(episodes):
        obs, _ = env.reset(seed=_episode_seed(seed, 1, episode))
        state = observation_to_state_key(obs)
        action = agent.select_action(state, explore=True)

        done = False
        episode_reward = 0.0
        episode_steps = 0

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = observation_to_state_key(next_obs)

            done = bool(terminated or truncated)
            next_action = 0 if done else agent.select_action(next_state, explore=True)
            agent.update(state, action, reward, next_state, next_action, done)

            episode_reward += float(reward)
            episode_steps += 1
            state = next_state
            action = next_action

        agent.decay_hyperparameters()
        metrics["reward_per_episode"].append(episode_reward)
        metrics["success_per_episode"].append(_is_success(bool(terminated), state, env.goal_position))
        metrics["steps_per_episode"].append(episode_steps)

    return metrics, agent.q_table


def _run_q_learning(episodes: int, seed: int) -> tuple[Metrics, dict[StateKey, np.ndarray]]:
    env = FloodEscapeEnv()
    agent = QLearningAgent(n_actions=env.action_space.n, seed=seed)
    metrics = _init_metrics()

    for episode in range(episodes):
        obs, _ = env.reset(seed=_episode_seed(seed, 2, episode))
        state = observation_to_state_key(obs)

        done = False
        episode_reward = 0.0
        episode_steps = 0

        while not done:
            action = agent.select_action(state, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = observation_to_state_key(next_obs)

            done = bool(terminated or truncated)
            agent.update(state, action, reward, next_state, done)

            episode_reward += float(reward)
            episode_steps += 1
            state = next_state

        agent.decay_hyperparameters()
        metrics["reward_per_episode"].append(episode_reward)
        metrics["success_per_episode"].append(_is_success(bool(terminated), state, env.goal_position))
        metrics["steps_per_episode"].append(episode_steps)

    return metrics, agent.q_table


def _run_td_prediction(episodes: int, seed: int) -> tuple[Metrics, dict[StateKey, float]]:
    env = FloodEscapeEnv()
    agent = TDPrediction()
    metrics = _init_metrics()
    rng = np.random.default_rng(seed)

    for episode in range(episodes):
        obs, _ = env.reset(seed=_episode_seed(seed, 3, episode))
        state = observation_to_state_key(obs)

        done = False
        episode_reward = 0.0
        episode_steps = 0

        while not done:
            action = _td_behavior_action(
                state,
                agent,
                rng,
                epsilon=0.10,
                grid_size=env.grid_size,
            )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = observation_to_state_key(next_obs)

            done = bool(terminated or truncated)
            agent.update(state, reward, next_state, done)

            episode_reward += float(reward)
            episode_steps += 1
            state = next_state

        agent.decay_alpha()
        metrics["reward_per_episode"].append(episode_reward)
        metrics["success_per_episode"].append(_is_success(bool(terminated), state, env.goal_position))
        metrics["steps_per_episode"].append(episode_steps)

    return metrics, agent.v_table


def run_all_experiments(
    episodes: int,
    seed: int,
    smooth_window: int,
    output_dir: str | Path,
) -> tuple[MetricsByAlgorithm, list[Path]]:
    """Run all required training loops and save plot artifacts."""
    trainers: list[tuple[str, Callable[[int, int], tuple[Metrics, dict[StateKey, Any]]]]] = [
        ("MonteCarloControl", _run_monte_carlo),
        ("TDPrediction", _run_td_prediction),
        ("SARSAAgent", _run_sarsa),
        ("QLearningAgent", _run_q_learning),
    ]

    metrics_by_algo: MetricsByAlgorithm = {}
    generated_plots: list[Path] = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for algo_name, train_fn in trainers:
        metrics, table = train_fn(episodes, seed)
        metrics_by_algo[algo_name] = metrics

        generated_plots.append(
            plot_algorithm_learning_curves(algo_name, metrics, output_path, smooth_window)
        )

        heatmap_file = output_path / f"value_heatmap_{algo_name.lower()}.png"
        generated_plots.append(
            plot_value_heatmap(table, heatmap_file, grid_size=6, title=f"{algo_name} Value Heatmap")
        )

        if algo_name != "TDPrediction":
            policy_file = output_path / f"policy_{algo_name.lower()}.png"
            generated_plots.append(
                plot_policy(table, policy_file, grid_size=6, title=f"{algo_name} Greedy Policy")
            )

    generated_plots.append(plot_learning_curves(metrics_by_algo, output_path, smooth_window))
    generated_plots.append(plot_steps_comparison(metrics_by_algo, output_path, smooth_window))
    generated_plots.append(plot_summary_metrics(metrics_by_algo, output_path))
    return metrics_by_algo, generated_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Flood Escape RL experiments.")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=100,
        help="Smoothing window size for learning curves.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Directory where plots are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = max(1, int(args.episodes))
    smooth_window = max(1, int(args.smooth_window))

    metrics_by_algo, generated_plots = run_all_experiments(
        episodes=episodes,
        seed=int(args.seed),
        smooth_window=smooth_window,
        output_dir=args.output_dir,
    )

    print(
        "Completed runs:",
        ", ".join(f"{name}({len(metrics['reward_per_episode'])})" for name, metrics in metrics_by_algo.items()),
    )
    print("Generated plots:")
    for plot_path in generated_plots:
        print(f"- {plot_path}")


if __name__ == "__main__":
    main()
