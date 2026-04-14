"""Training entry point for Flood Escape Phase 4/5 experiments."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from algorithms.base_agent import LOCAL_SENSOR_RADIUS, StateKey, observation_to_state_key
from algorithms.dyna_q import DynaQAgent
from algorithms.monte_carlo import MonteCarloControl
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from algorithms.td_learning import TDPrediction
from env import FloodEscapeEnv
from utils import (
    plot_algorithm_learning_curves,
    plot_environment_rollouts,
    plot_learning_curves,
    plot_policy,
    plot_policy_grid_image,
    plot_steps_comparison,
    plot_summary_metrics,
    plot_value_heatmap,
)

MetricValue = float | int
Metrics = dict[str, list[MetricValue]]
MetricsByAlgorithm = dict[str, Metrics]
PolicyByPosition = dict[tuple[int, int], int]

TABLE_ROW_WIDTH = {
    "Q-table": 4,
    "V-table": 1,
}

ACTION_LABELS = {
    0: "U",
    1: "D",
    2: "L",
    3: "R",
}


def _init_metrics() -> Metrics:
    return {
        "reward_per_episode": [],
        "success_per_episode": [],
        "steps_per_episode": [],
    }


def _episode_seed(base_seed: int, _algo_offset: int, episode_index: int) -> int:
    """Generate per-episode seeds shared across algorithms for fair comparison."""
    return base_seed + episode_index


def _is_success(
    *,
    terminated: bool,
    truncated: bool,
    final_state: StateKey,
    goal_position: tuple[int, int],
) -> int:
    return 1 if (terminated or truncated) and final_state[0] == goal_position else 0


def _td_behavior_action(
    observation: dict[str, np.ndarray],
    td_agent: TDPrediction,
    rng: np.random.Generator,
    *,
    epsilon: float,
    move_success_prob: float,
) -> int:
    """Epsilon-greedy behavior policy for TD(0) over one-step value lookahead."""
    if rng.random() < epsilon:
        return int(rng.integers(0, 4))

    agent_position = np.asarray(observation["agent"], dtype=np.int64).reshape(-1)
    x, y = int(agent_position[0]), int(agent_position[1])
    flood_grid = np.asarray(observation["flood"], dtype=np.uint8)
    grid_size = int(flood_grid.shape[0])

    def _state_for_position(position: tuple[int, int]) -> StateKey:
        return observation_to_state_key(
            {
                "agent": np.asarray(position, dtype=np.int64),
                "flood": flood_grid,
            }
        )

    # 1. Identify all 4 possible intended next-positions.
    intended_positions = {
        0: (max(0, x - 1), y),
        1: (min(grid_size - 1, x + 1), y),
        2: (x, max(0, y - 1)),
        3: (x, min(grid_size - 1, y + 1)),
    }

    # 2. Get the values of all possible neighbors for weighted randomness.
    # The agent gets a random neighboring cell 20% of the time on move failure.
    neighbor_positions = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid_size and 0 <= ny < grid_size:
            neighbor_positions.append((nx, ny))

    neighbor_values = [td_agent.value(_state_for_position(pos)) for pos in neighbor_positions]
    avg_neighbor_value = float(np.mean(neighbor_values)) if neighbor_positions else 0.0

    # 3. Compute expected value for each action: p * V(intended) + (1-p) * E[V(random_neighbor)].
    action_expectations = np.empty(4, dtype=np.float64)
    for action, pos in intended_positions.items():
        v_intended = td_agent.value(_state_for_position(pos))
        action_expectations[action] = (move_success_prob * v_intended) + ((1.0 - move_success_prob) * avg_neighbor_value)

    max_value = float(np.max(action_expectations))
    best_actions = np.flatnonzero(np.isclose(action_expectations, max_value))
    return int(best_actions[int(rng.integers(0, len(best_actions)))])


def _run_monte_carlo(
    episodes: int,
    seed: int,
) -> tuple[Metrics, dict[StateKey, np.ndarray], set[StateKey]]:
    env = FloodEscapeEnv()
    agent = MonteCarloControl(n_actions=env.action_space.n, seed=seed)
    metrics = _init_metrics()
    visited_states: set[StateKey] = set()

    for episode in range(episodes):
        obs, _ = env.reset(seed=_episode_seed(seed, 0, episode))
        state = observation_to_state_key(obs)
        visited_states.add(state)
        done = False
        terminated = False
        truncated = False

        episode_reward = 0.0
        episode_steps = 0
        agent.start_episode()

        while not done:
            action = agent.select_action(state, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = observation_to_state_key(next_obs)
            visited_states.add(next_state)

            agent.record_transition(state, action, reward)

            episode_reward += float(reward)
            episode_steps += 1
            done = bool(terminated or truncated)
            state = next_state

        agent.end_episode(
            terminal_state=state if bool(terminated) else None,
            terminated=bool(terminated),
        )
        metrics["reward_per_episode"].append(episode_reward)
        metrics["success_per_episode"].append(
            _is_success(
                terminated=bool(terminated),
                truncated=bool(truncated),
                final_state=state,
                goal_position=env.goal_position,
            )
        )
        metrics["steps_per_episode"].append(episode_steps)

    return metrics, agent.q_table, visited_states


def _run_sarsa(
    episodes: int,
    seed: int,
) -> tuple[Metrics, dict[StateKey, np.ndarray], set[StateKey]]:
    env = FloodEscapeEnv()
    agent = SARSAAgent(n_actions=env.action_space.n, seed=seed)
    metrics = _init_metrics()
    visited_states: set[StateKey] = set()

    for episode in range(episodes):
        obs, _ = env.reset(seed=_episode_seed(seed, 1, episode))
        state = observation_to_state_key(obs)
        visited_states.add(state)
        action = agent.select_action(state, explore=True)

        done = False
        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_steps = 0

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = observation_to_state_key(next_obs)
            visited_states.add(next_state)

            done = bool(terminated or truncated)
            next_action = 0 if terminated else agent.select_action(next_state, explore=True)
            agent.update(
                state,
                action,
                reward,
                next_state,
                next_action,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )

            episode_reward += float(reward)
            episode_steps += 1
            state = next_state
            action = next_action

        agent.decay_hyperparameters()
        metrics["reward_per_episode"].append(episode_reward)
        metrics["success_per_episode"].append(
            _is_success(
                terminated=bool(terminated),
                truncated=bool(truncated),
                final_state=state,
                goal_position=env.goal_position,
            )
        )
        metrics["steps_per_episode"].append(episode_steps)

    return metrics, agent.q_table, visited_states

def _run_q_learning(
    episodes: int,
    seed: int,
) -> tuple[Metrics, dict[StateKey, np.ndarray], set[StateKey]]:
    env = FloodEscapeEnv()
    agent = QLearningAgent(n_actions=env.action_space.n, seed=seed)
    metrics = _init_metrics()
    visited_states: set[StateKey] = set()

    for episode in range(episodes):
        obs, _ = env.reset(seed=_episode_seed(seed, 2, episode))
        state = observation_to_state_key(obs)
        visited_states.add(state)

        done = False
        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_steps = 0

        while not done:
            action = agent.select_action(state, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = observation_to_state_key(next_obs)
            visited_states.add(next_state)

            done = bool(terminated or truncated)
            agent.update(
                state,
                action,
                reward,
                next_state,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )

            episode_reward += float(reward)
            episode_steps += 1
            state = next_state

        agent.decay_hyperparameters()
        metrics["reward_per_episode"].append(episode_reward)
        metrics["success_per_episode"].append(
            _is_success(
                terminated=bool(terminated),
                truncated=bool(truncated),
                final_state=state,
                goal_position=env.goal_position,
            )
        )
        metrics["steps_per_episode"].append(episode_steps)

    return metrics, agent.q_table, visited_states


def _run_dyna_q(
    episodes: int,
    seed: int,
) -> tuple[Metrics, dict[StateKey, np.ndarray], set[StateKey]]:
    env = FloodEscapeEnv()
    agent = DynaQAgent(n_actions=env.action_space.n, seed=seed, planning_steps=20)
    metrics = _init_metrics()
    visited_states: set[StateKey] = set()

    for episode in range(episodes):
        obs, _ = env.reset(seed=_episode_seed(seed, 4, episode))
        state = observation_to_state_key(obs)
        visited_states.add(state)

        done = False
        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_steps = 0

        while not done:
            action = agent.select_action(state, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = observation_to_state_key(next_obs)
            visited_states.add(next_state)

            done = bool(terminated or truncated)
            agent.update(
                state,
                action,
                reward,
                next_state,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )

            episode_reward += float(reward)
            episode_steps += 1
            state = next_state

        agent.decay_hyperparameters()
        metrics["reward_per_episode"].append(episode_reward)
        metrics["success_per_episode"].append(
            _is_success(
                terminated=bool(terminated),
                truncated=bool(truncated),
                final_state=state,
                goal_position=env.goal_position,
            )
        )
        metrics["steps_per_episode"].append(episode_steps)

    return metrics, agent.q_table, visited_states


def _run_td_prediction(
    episodes: int,
    seed: int,
) -> tuple[Metrics, dict[StateKey, float], set[StateKey]]:
    env = FloodEscapeEnv()
    agent = TDPrediction()
    metrics = _init_metrics()
    rng = np.random.default_rng(seed)
    visited_states: set[StateKey] = set()

    for episode in range(episodes):
        obs, _ = env.reset(seed=_episode_seed(seed, 3, episode))
        state = observation_to_state_key(obs)
        visited_states.add(state)

        done = False
        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_steps = 0

        while not done:
            action = _td_behavior_action(
                obs,
                agent,
                rng,
                epsilon=agent.epsilon,
                move_success_prob=env.move_success_prob,
            )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = observation_to_state_key(next_obs)
            visited_states.add(next_state)

            done = bool(terminated or truncated)
            agent.update(
                state,
                reward,
                next_state,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )

            episode_reward += float(reward)
            episode_steps += 1
            obs = next_obs
            state = next_state

        agent.decay_hyperparameters()
        metrics["reward_per_episode"].append(episode_reward)
        metrics["success_per_episode"].append(
            _is_success(
                terminated=bool(terminated),
                truncated=bool(truncated),
                final_state=state,
                goal_position=env.goal_position,
            )
        )
        metrics["steps_per_episode"].append(episode_steps)

    return metrics, agent.v_table, visited_states


def _compute_state_space_caps(
    *,
    grid_size: int,
    local_sensor_radius: int,
) -> dict[str, int]:
    positions = grid_size * grid_size
    full_flood_bits = grid_size * grid_size
    local_sensor_width = (2 * local_sensor_radius) + 1
    local_sensor_bits = local_sensor_width * local_sensor_width

    return {
        "positions": positions,
        "full_flood_bits": full_flood_bits,
        "local_sensor_width": local_sensor_width,
        "local_sensor_bits": local_sensor_bits,
        "old_cap": positions * (2**full_flood_bits),
        "new_cap": positions * (2**local_sensor_bits),
    }


def _compute_sparsity_metrics(
    *,
    unique_states_visited: int,
    table_rows: int,
    old_cap: int,
    new_cap: int,
) -> dict[str, float | int]:
    practical_occupancy_new = (table_rows / new_cap) if new_cap else 0.0
    visited_coverage_new = (unique_states_visited / new_cap) if new_cap else 0.0
    baseline_occupancy_old = (table_rows / old_cap) if old_cap else 0.0

    return {
        "unique_states_visited": unique_states_visited,
        "table_rows": table_rows,
        "practical_occupancy_new": practical_occupancy_new,
        "visited_coverage_new": visited_coverage_new,
        "baseline_occupancy_old": baseline_occupancy_old,
    }


def _format_sparsity_section(
    *,
    sparsity_by_algo: dict[str, dict[str, float | int | str]],
    caps: dict[str, int],
    scenario_label: str | None = None,
) -> str:
    reduction_factor = caps["old_cap"] / caps["new_cap"] if caps["new_cap"] else float("inf")

    lines = [
        "Phase 3 Convergence Sparsity and Coverage",
        (
            f"Scenario context: {scenario_label}"
            if scenario_label
            else "Scenario context: Scenario-conditioned (deterministic base fallback flood: none)"
        ),
        "State-space baseline comparison:",
        f"- Old cap (full flood map, {caps['full_flood_bits']} bits): {caps['old_cap']}",
        (
            "- New cap (local sensor "
            f"{caps['local_sensor_width']}x{caps['local_sensor_width']}, {caps['local_sensor_bits']} bits): "
            f"{caps['new_cap']}"
        ),
        f"- Reduction factor (old/new): {reduction_factor:.3e}",
        "Formulas:",
        "- old_cap = |positions| * 2^(grid_size^2)",
        "- new_cap = |positions| * 2^((2*sensor_radius+1)^2)",
        "- Practical occupancy ratio (table/new cap): table_rows / new_cap",
        "- Coverage ratio (visited/new cap): unique_states_visited / new_cap",
        "- Old baseline occupancy (table/old cap): table_rows / old_cap",
    ]

    for algo_name, metrics in sparsity_by_algo.items():
        table_kind = str(metrics["table_kind"])
        table_rows = int(metrics["table_rows"])
        table_width = TABLE_ROW_WIDTH[table_kind]
        table_entries = table_rows * table_width

        lines.extend(
            [
                "",
                f"{algo_name} sparsity",
                f"- Unique states visited: {int(metrics['unique_states_visited'])}",
                f"- Table rows: {table_rows}",
                f"- {table_kind} size (rows x values): {table_rows} x {table_width} = {table_entries}",
                f"- Practical occupancy ratio (table/new cap): {float(metrics['practical_occupancy_new']):.6e}",
                f"- Coverage ratio (visited/new cap): {float(metrics['visited_coverage_new']):.6e}",
                f"- Old baseline occupancy (table/old cap): {float(metrics['baseline_occupancy_old']):.6e}",
            ]
        )

    return "\n".join(lines)


def _scenario_state_key(position: tuple[int, int], scenario_flood_map: np.ndarray) -> StateKey:
    """Build a state key query for one position under a fixed flood scenario."""
    observation = {
        "agent": np.asarray(position, dtype=np.int64),
        "flood": scenario_flood_map,
    }
    return observation_to_state_key(observation)


def _extract_q_by_position_for_scenario(
    q_table: dict[StateKey, np.ndarray],
    grid_size: int,
    scenario_flood_map: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    extracted: dict[tuple[int, int], np.ndarray] = {}

    for x in range(grid_size):
        for y in range(grid_size):
            state_key = _scenario_state_key((x, y), scenario_flood_map)
            q_values = q_table.get(state_key)
            if q_values is None:
                continue

            q_array = np.asarray(q_values, dtype=np.float64).reshape(-1)
            if q_array.size < 4 or not np.all(np.isfinite(q_array[:4])):
                continue

            extracted[(x, y)] = q_array[:4]

    return extracted


def _extract_v_by_position_for_scenario(
    v_table: dict[StateKey, float],
    grid_size: int,
    scenario_flood_map: np.ndarray,
) -> dict[tuple[int, int], float]:
    extracted: dict[tuple[int, int], float] = {}

    for x in range(grid_size):
        for y in range(grid_size):
            state_key = _scenario_state_key((x, y), scenario_flood_map)
            value = v_table.get(state_key)
            if value is None:
                continue

            numeric_value = float(value)
            if np.isfinite(numeric_value):
                extracted[(x, y)] = numeric_value

    return extracted


def _resolve_policy_scenario_flood_map(
    scenario_flood_map: np.ndarray | None,
    grid_size: int,
) -> np.ndarray:
    if scenario_flood_map is None:
        return np.zeros((grid_size, grid_size), dtype=np.uint8)

    resolved_map = np.asarray(scenario_flood_map, dtype=np.uint8)
    if resolved_map.shape != (grid_size, grid_size):
        raise ValueError("scenario_flood_map shape must match (grid_size, grid_size).")
    return resolved_map


def _derive_policy_from_q_table(
    q_table: dict[StateKey, np.ndarray],
    grid_size: int,
    scenario_flood_map: np.ndarray | None = None,
) -> PolicyByPosition:
    resolved_scenario_map = _resolve_policy_scenario_flood_map(scenario_flood_map, grid_size)
    aggregated_q = _extract_q_by_position_for_scenario(q_table, grid_size, resolved_scenario_map)
    policy: PolicyByPosition = {}

    for position, q_values in aggregated_q.items():
        best_action = int(np.argmax(q_values))
        policy[position] = best_action

    return policy


def _derive_policy_from_v_table(
    v_table: dict[StateKey, float],
    grid_size: int,
    scenario_flood_map: np.ndarray | None = None,
) -> PolicyByPosition:
    resolved_scenario_map = _resolve_policy_scenario_flood_map(scenario_flood_map, grid_size)
    aggregated_v = _extract_v_by_position_for_scenario(v_table, grid_size, resolved_scenario_map)
    policy: PolicyByPosition = {}

    for x in range(grid_size):
        for y in range(grid_size):
            candidate_positions = {
                0: (max(0, x - 1), y),
                1: (min(grid_size - 1, x + 1), y),
                2: (x, max(0, y - 1)),
                3: (x, min(grid_size - 1, y + 1)),
            }

            candidate_values = np.array(
                [aggregated_v.get(candidate_positions[action], float("-inf")) for action in range(4)],
                dtype=np.float64,
            )

            if not np.isfinite(candidate_values).any():
                continue

            best_action = int(np.argmax(candidate_values))
            policy[(x, y)] = best_action

    return policy


def _format_policy_grid(
    algo_name: str,
    policy: PolicyByPosition,
    grid_size: int,
    goal_position: tuple[int, int],
    scenario_label: str | None = None,
) -> str:
    header = "    " + " ".join(f"y{idx}" for idx in range(grid_size))
    title = f"{algo_name} policy (greedy)"
    if scenario_label:
        title = f"{title} - {scenario_label}"
    lines = [title, "Legend: U=Up D=Down L=Left R=Right G=Goal .=Unknown", header]

    for x in range(grid_size):
        row_tokens: list[str] = []
        for y in range(grid_size):
            position = (x, y)
            if position == goal_position:
                token = "G"
            else:
                action = policy.get(position)
                token = ACTION_LABELS.get(action, ".")
            row_tokens.append(token)
        lines.append(f"x{x}: " + " ".join(row_tokens))

    return "\n".join(lines)


def _write_policy_report(sections: list[str], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return output_path


def _simulate_policy_rollout(
    policy: PolicyByPosition,
    *,
    seed: int,
    max_steps: int = 100,
) -> dict[str, object]:
    """Simulate one greedy rollout for environment-level visualization."""
    env = FloodEscapeEnv(max_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    path: list[tuple[int, int]] = [
        (int(obs["agent"][0]), int(obs["agent"][1]))
    ]

    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):
        current_position = (int(obs["agent"][0]), int(obs["agent"][1]))
        action = int(policy.get(current_position, 0))

        obs, _, terminated, truncated, _ = env.step(action)
        path.append((int(obs["agent"][0]), int(obs["agent"][1])))
        steps += 1

    final_position = (int(obs["agent"][0]), int(obs["agent"][1]))
    success = bool(terminated and final_position == env.goal_position)

    return {
        "path": path,
        "flood_map": np.asarray(obs["flood"], dtype=np.float64),
        "success": success,
        "steps": steps,
    }


def run_all_experiments(
    episodes: int,
    seed: int,
    smooth_window: int,
    output_dir: str | Path,
) -> tuple[MetricsByAlgorithm, list[Path]]:
    """Run all required training loops and save plot artifacts."""
    trainers: list[
        tuple[str, Callable[[int, int], tuple[Metrics, dict[StateKey, Any], set[StateKey]]]]
    ] = [
        ("MonteCarloControl", _run_monte_carlo),
        ("TDPrediction", _run_td_prediction),
        ("SARSAAgent", _run_sarsa),
        ("QLearningAgent", _run_q_learning),
        ("DynaQAgent", _run_dyna_q),
    ]

    metrics_by_algo: MetricsByAlgorithm = {}
    generated_plots: list[Path] = []
    derived_policies: dict[str, PolicyByPosition] = {}
    policy_sections: list[str] = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    grid_size = 6
    goal_position = (grid_size - 1, grid_size - 1)
    sparsity_by_algo: dict[str, dict[str, float | int | str]] = {}

    scenario_env = FloodEscapeEnv()
    scenario_obs, _ = scenario_env.reset(seed=seed)
    scenario_flood_map = np.asarray(scenario_obs["flood"], dtype=np.uint8)
    grid_size = int(scenario_env.grid_size)
    goal_position = tuple(scenario_env.goal_position)
    caps = _compute_state_space_caps(
        grid_size=grid_size,
        local_sensor_radius=LOCAL_SENSOR_RADIUS,
    )

    flooded_cells = [(int(x), int(y)) for x, y in np.argwhere(scenario_flood_map == 1)]
    flood_cells_text = ", ".join(f"({x},{y})" for x, y in flooded_cells) if flooded_cells else "none"
    scenario_label = f"Scenario-conditioned (base reset flood: {flood_cells_text})"

    for algo_name, train_fn in trainers:
        metrics, table, visited_states = train_fn(episodes, seed)
        metrics_by_algo[algo_name] = metrics

        table_kind = "V-table" if algo_name == "TDPrediction" else "Q-table"
        sparsity_metrics = _compute_sparsity_metrics(
            unique_states_visited=len(visited_states),
            table_rows=len(table),
            old_cap=int(caps["old_cap"]),
            new_cap=int(caps["new_cap"]),
        )
        sparsity_by_algo[algo_name] = {
            "table_kind": table_kind,
            **sparsity_metrics,
        }

        if algo_name == "TDPrediction":
            derived_policy = _derive_policy_from_v_table(
                table,
                grid_size,
                scenario_flood_map=scenario_flood_map,
            )
        else:
            derived_policy = _derive_policy_from_q_table(
                table,
                grid_size,
                scenario_flood_map=scenario_flood_map,
            )
        derived_policies[algo_name] = derived_policy

        policy_sections.append(
            _format_policy_grid(
                algo_name=algo_name,
                scenario_label=scenario_label,
                policy=derived_policy,
                grid_size=grid_size,
                goal_position=goal_position,
            )
        )

        generated_plots.append(
            plot_algorithm_learning_curves(algo_name, metrics, output_path, smooth_window)
        )

        policy_grid_file = output_path / f"policy_grid_{algo_name.lower()}.png"
        generated_plots.append(
            plot_policy_grid_image(
                algo_name=algo_name,
                policy=derived_policy,
                output_path=policy_grid_file,
                grid_size=grid_size,
                goal_position=goal_position,
                scenario_label=scenario_label,
            )
        )

        heatmap_file = output_path / f"value_heatmap_{algo_name.lower()}.png"
        generated_plots.append(
            plot_value_heatmap(
                table,
                heatmap_file,
                grid_size=grid_size,
                title=f"{algo_name} Value Heatmap",
                scenario_flood_map=scenario_flood_map,
                scenario_label=scenario_label,
            )
        )

        if algo_name != "TDPrediction":
            policy_file = output_path / f"policy_{algo_name.lower()}.png"
            generated_plots.append(
                plot_policy(
                    table,
                    policy_file,
                    grid_size=grid_size,
                    title=f"{algo_name} Greedy Policy",
                    scenario_flood_map=scenario_flood_map,
                    scenario_label=scenario_label,
                )
            )

    generated_plots.append(plot_learning_curves(metrics_by_algo, output_path, smooth_window))
    generated_plots.append(plot_steps_comparison(metrics_by_algo, output_path, smooth_window))
    generated_plots.append(plot_summary_metrics(metrics_by_algo, output_path))

    rollout_seed = _episode_seed(seed, 8, 0)
    rollout_views = {
        algo_name: _simulate_policy_rollout(
            policy,
            seed=rollout_seed,
            max_steps=100,
        )
        for algo_name, policy in derived_policies.items()
    }
    generated_plots.append(
        plot_environment_rollouts(
            rollout_views,
            output_path / "environment_rollouts.png",
            grid_size=grid_size,
            goal_position=goal_position,
        )
    )

    policy_report_path = output_path.parent / "tables" / "policies_report.txt"
    policy_sections.append(
        _format_sparsity_section(
            sparsity_by_algo=sparsity_by_algo,
            caps=caps,
            scenario_label=scenario_label,
        )
    )
    _write_policy_report(policy_sections, policy_report_path)

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

    policy_report_path = Path(args.output_dir).parent / "tables" / "policies_report.txt"
    if policy_report_path.exists():
        print(f"Policy report: {policy_report_path}")
        print("\n" + policy_report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
