"""Focused tests for FloodEscapeEnv Phase 2 requirements."""

from __future__ import annotations

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
