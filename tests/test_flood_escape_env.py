"""Focused tests for FloodEscapeEnv Phase 2 requirements."""

from __future__ import annotations

import pytest

from env import FloodEscapeEnv


def test_reset_never_spawns_on_goal_or_flood_across_many_seeds() -> None:
    env = FloodEscapeEnv()

    for seed in range(250):
        obs, _ = env.reset(seed=seed)
        agent_pos = tuple(int(v) for v in obs["agent"])

        assert agent_pos != env.goal_position
        assert env.flood_map[agent_pos] == 0


def test_agent_movement_never_leaves_grid_under_stochasticity() -> None:
    env = FloodEscapeEnv(max_steps=5000, flood_spread_prob=0.0)
    obs, _ = env.reset(seed=3)

    for step_index in range(1500):
        action = step_index % 4
        obs, _, terminated, truncated, _ = env.step(action)

        x, y = (int(obs["agent"][0]), int(obs["agent"][1]))
        assert 0 <= x <= 5
        assert 0 <= y <= 5

        if terminated or truncated:
            obs, _ = env.reset(seed=1000 + step_index)


def test_flood_spread_never_overwrites_goal_cell() -> None:
    env = FloodEscapeEnv(move_success_prob=1.0, flood_spread_prob=1.0)
    env.reset(seed=11)

    goal_x, goal_y = env.goal_position
    env.flood_map.fill(0)

    neighbor = (goal_x - 1, goal_y) if goal_x > 0 else (goal_x + 1, goal_y)
    env.flood_map[neighbor] = 1

    env.step(0)

    assert env.flood_map[env.goal_position] == 0


def test_episode_terminates_on_goal_and_flood() -> None:
    env = FloodEscapeEnv(move_success_prob=1.0, flood_spread_prob=0.0, max_steps=50)

    env.reset(seed=19)
    goal_x, goal_y = env.goal_position
    if goal_x > 0:
        env.agent_position = (goal_x - 1, goal_y)
        action_to_goal = 1
    else:
        env.agent_position = (goal_x + 1, goal_y)
        action_to_goal = 0

    _, _, terminated_goal, truncated_goal, _ = env.step(action_to_goal)
    assert terminated_goal is True
    assert truncated_goal is False

    env.reset(seed=20)
    env.flood_map.fill(0)
    env.agent_position = (1, 0)
    env.flood_map[1, 1] = 1

    _, _, terminated_flood, truncated_flood, _ = env.step(3)
    assert terminated_flood is True
    assert truncated_flood is False


def test_flood_spread_probability_applies_once_per_target_cell() -> None:
    spread_prob = 0.25
    trials = 3000
    env = FloodEscapeEnv(flood_spread_prob=spread_prob)

    infected_count = 0
    for seed in range(trials):
        env.reset(seed=seed)
        env.flood_map.fill(0)
        env.flood_map[2, 2] = 1
        env.flood_map[2, 4] = 1

        # Cell (2, 3) has two flooded neighbors and must still get one draw.
        env._spread_flood()
        infected_count += int(env.flood_map[2, 3])

    observed_rate = infected_count / trials
    assert abs(observed_rate - spread_prob) < 0.06
    assert observed_rate < 0.33


def test_two_flooded_neighbors_do_not_compound_target_spread_probability() -> None:
    spread_prob = 0.30
    trials = 3000
    env = FloodEscapeEnv(flood_spread_prob=spread_prob)

    single_neighbor_hits = 0
    double_neighbor_hits = 0
    for seed in range(trials):
        env.reset(seed=10000 + seed)
        env.flood_map.fill(0)
        env.flood_map[2, 2] = 1
        env.flood_map[2, 4] = 1

        env._spread_flood()

        # (1, 2) has one flooded neighbor; (2, 3) has two flooded neighbors.
        single_neighbor_hits += int(env.flood_map[1, 2])
        double_neighbor_hits += int(env.flood_map[2, 3])

    single_neighbor_rate = single_neighbor_hits / trials
    double_neighbor_rate = double_neighbor_hits / trials

    assert abs(single_neighbor_rate - spread_prob) < 0.07
    assert abs(double_neighbor_rate - spread_prob) < 0.07
    assert abs(double_neighbor_rate - single_neighbor_rate) < 0.08


def test_default_flood_spread_probability_is_balanced_for_learning_signal() -> None:
    env = FloodEscapeEnv()
    assert env.flood_spread_prob == pytest.approx(0.5)


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        ({"max_steps": 0}, "max_steps must be at least 1."),
        ({"move_success_prob": -0.1}, "move_success_prob must be in [0.0, 1.0]."),
        ({"move_success_prob": 1.1}, "move_success_prob must be in [0.0, 1.0]."),
        ({"flood_spread_prob": -0.1}, "flood_spread_prob must be in [0.0, 1.0]."),
        ({"flood_spread_prob": 1.1}, "flood_spread_prob must be in [0.0, 1.0]."),
    ],
)
def test_constructor_validates_core_parameters(kwargs: dict[str, float | int], expected_message: str) -> None:
    with pytest.raises(ValueError) as error_info:
        FloodEscapeEnv(**kwargs)
    assert str(error_info.value) == expected_message
