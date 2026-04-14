"""Verification tests for Phase 3 RL algorithm correctness guarantees."""

from __future__ import annotations

import numpy as np
import pytest
from algorithms import DynaQAgent, MonteCarloControl, QLearningAgent, SARSAAgent, observation_to_state_key


def _state(agent_xy: tuple[int, int], flood_grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    """Build canonical hashable StateKey fixtures for tests."""
    return (agent_xy, np.ascontiguousarray(flood_grid.astype(np.uint8)).tobytes())


@pytest.mark.parametrize("agent_cls", [QLearningAgent, SARSAAgent, DynaQAgent])
def test_terminal_state_q_values_remain_zeros_and_never_update(agent_cls: type) -> None:
    flood = np.zeros((2, 2), dtype=np.uint8)
    terminal_state = _state((1, 1), flood)

    agent = agent_cls(n_actions=4, alpha=1.0, gamma=0.9, epsilon=0.2)
    agent.mark_terminal_state(terminal_state)

    assert np.array_equal(agent.q_values(terminal_state), np.zeros(4, dtype=np.float64))

    for _ in range(20):
        agent.update_q_value(terminal_state, action=2, target=123.456)

    assert np.array_equal(agent.q_values(terminal_state), np.zeros(4, dtype=np.float64))


def test_q_learning_uses_off_policy_max_backup() -> None:
    flood = np.zeros((2, 2), dtype=np.uint8)
    state = _state((0, 0), flood)
    next_state = _state((0, 1), flood)

    agent = QLearningAgent(n_actions=3, alpha=1.0, gamma=0.9)
    agent.q_table[next_state] = np.array([1.0, 10.0, 3.0], dtype=np.float64)

    agent.update(
        state=state,
        action=0,
        reward=2.0,
        next_state=next_state,
        terminated=False,
        truncated=False,
    )

    expected = 2.0 + 0.9 * 10.0
    assert agent.q_values(state)[0] == pytest.approx(expected)


def test_q_learning_truncation_does_not_zero_bootstrap_target() -> None:
    flood = np.zeros((2, 2), dtype=np.uint8)
    state = _state((0, 0), flood)
    next_state = _state((0, 1), flood)

    agent = QLearningAgent(n_actions=3, alpha=1.0, gamma=0.5)
    agent.q_table[next_state] = np.array([4.0, 1.0, 0.0], dtype=np.float64)

    agent.update(
        state=state,
        action=1,
        reward=2.0,
        next_state=next_state,
        terminated=False,
        truncated=True,
    )

    expected = 2.0 + 0.5 * 4.0
    assert agent.q_values(state)[1] == pytest.approx(expected)
    assert not agent.is_terminal_state(next_state)


def test_sarsa_uses_on_policy_next_action_backup() -> None:
    flood = np.zeros((2, 2), dtype=np.uint8)
    state = _state((0, 0), flood)
    next_state = _state((0, 1), flood)

    agent = SARSAAgent(n_actions=3, alpha=1.0, gamma=0.9)
    agent.q_table[next_state] = np.array([1.0, 10.0, 3.0], dtype=np.float64)

    agent.update(
        state=state,
        action=0,
        reward=2.0,
        next_state=next_state,
        next_action=2,
        terminated=False,
        truncated=False,
    )

    expected = 2.0 + 0.9 * 3.0
    assert agent.q_values(state)[0] == pytest.approx(expected)


def test_monte_carlo_marks_terminal_state_at_episode_end() -> None:
    flood = np.zeros((2, 2), dtype=np.uint8)
    start_state = _state((0, 0), flood)
    terminal_state = _state((1, 1), flood)

    agent = MonteCarloControl(n_actions=4, gamma=1.0, alpha=1.0)
    agent.start_episode()
    agent.record_transition(start_state, action=2, reward=1.0)

    agent.end_episode(terminal_state=terminal_state, terminated=True)

    assert agent.is_terminal_state(terminal_state)
    assert np.array_equal(agent.q_values(terminal_state), np.zeros(4, dtype=np.float64))


def test_monte_carlo_marks_terminal_state_even_for_empty_episode() -> None:
    flood = np.zeros((2, 2), dtype=np.uint8)
    terminal_state = _state((1, 1), flood)

    agent = MonteCarloControl(n_actions=4)
    agent.start_episode()
    agent.end_episode(terminal_state=terminal_state, terminated=True)

    assert agent.is_terminal_state(terminal_state)
    assert np.array_equal(agent.q_values(terminal_state), np.zeros(4, dtype=np.float64))


def test_observation_to_state_key_uses_compact_local_sensor_and_is_hashable() -> None:
    observation = {
        "agent": np.array([2, 2], dtype=np.int32),
        "flood": np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    }

    key = observation_to_state_key(observation)

    hash(key)
    assert key == ((2, 2), bytes([0xD5, 0x00]))
    assert isinstance(key, tuple)
    assert isinstance(key[0], tuple)
    assert isinstance(key[1], bytes)
    assert len(key[1]) == 2


def test_observation_to_state_key_ignores_flood_cells_outside_local_neighborhood() -> None:
    flood_a = np.zeros((6, 6), dtype=np.uint8)
    flood_b = np.zeros((6, 6), dtype=np.uint8)

    flood_a[2, 2] = 1
    flood_b[2, 2] = 1
    flood_a[0, 0] = 1
    flood_b[5, 5] = 1

    key_a = observation_to_state_key({"agent": np.array([2, 2]), "flood": flood_a})
    key_b = observation_to_state_key({"agent": np.array([2, 2]), "flood": flood_b})

    assert key_a == key_b


def test_observation_to_state_key_preserves_agent_position_in_state() -> None:
    flood = np.zeros((6, 6), dtype=np.uint8)

    key_a = observation_to_state_key({"agent": np.array([1, 1]), "flood": flood})
    key_b = observation_to_state_key({"agent": np.array([1, 2]), "flood": flood})

    assert key_a[0] == (1, 1)
    assert key_b[0] == (1, 2)
    assert key_a != key_b


def test_epsilon_decay_converges_to_min_and_never_goes_below_floor() -> None:
    agent = QLearningAgent(
        n_actions=4,
        epsilon=1.0,
        epsilon_decay=0.99,
        min_epsilon=0.05,
    )

    for _ in range(1000):
        current = agent.decay_epsilon()
        assert current >= agent.min_epsilon

    assert agent.epsilon == pytest.approx(agent.min_epsilon)


def test_dyna_q_planning_accelerates_value_update() -> None:
    flood = np.zeros((2, 2), dtype=np.uint8)
    state = _state((0, 0), flood)
    next_state = _state((0, 1), flood)

    agent = DynaQAgent(n_actions=4, alpha=0.5, gamma=0.0, planning_steps=10)
    agent.update(
        state=state,
        action=0,
        reward=1.0,
        next_state=next_state,
        terminated=False,
        truncated=False,
    )

    # Real update alone would set Q to 0.5; planning should push it closer to 1.0.
    assert agent.q_values(state)[0] > 0.5


def test_dyna_q_truncation_does_not_zero_bootstrap_target() -> None:
    flood = np.zeros((2, 2), dtype=np.uint8)
    state = _state((0, 0), flood)
    next_state = _state((0, 1), flood)

    agent = DynaQAgent(n_actions=3, alpha=1.0, gamma=0.5, planning_steps=0)
    agent.q_table[next_state] = np.array([4.0, 1.0, 0.0], dtype=np.float64)

    agent.update(
        state=state,
        action=1,
        reward=2.0,
        next_state=next_state,
        terminated=False,
        truncated=True,
    )

    expected = 2.0 + 0.5 * 4.0
    assert agent.q_values(state)[1] == pytest.approx(expected)
    assert not agent.is_terminal_state(next_state)


def test_dyna_q_model_keeps_multiple_outcomes_and_samples_stochastically() -> None:
    flood = np.zeros((2, 2), dtype=np.uint8)
    state = _state((0, 0), flood)
    next_state_a = _state((0, 1), flood)
    next_state_b = _state((1, 0), flood)

    agent = DynaQAgent(n_actions=2, alpha=1.0, gamma=0.0, planning_steps=0, seed=7)
    agent.update(
        state=state,
        action=0,
        reward=1.0,
        next_state=next_state_a,
        terminated=False,
        truncated=False,
    )
    agent.update(
        state=state,
        action=0,
        reward=3.0,
        next_state=next_state_b,
        terminated=False,
        truncated=False,
    )

    model_key = (state, 0)
    assert len(agent.model[model_key]) == 2

    sampled_next_states: set[tuple[tuple[int, int], bytes]] = set()
    sampled_rewards: set[float] = set()
    for _ in range(200):
        sampled_next_state, sampled_reward, sampled_terminated = agent._sample_model_transition(model_key)
        sampled_next_states.add(sampled_next_state)
        sampled_rewards.add(sampled_reward)
        assert sampled_terminated is False

    assert sampled_next_states == {next_state_a, next_state_b}
    assert sampled_rewards == {1.0, 3.0}


def test_dyna_q_conflicting_terminal_flags_resolve_to_terminal_outcome() -> None:
    flood = np.zeros((2, 2), dtype=np.uint8)
    state = _state((0, 0), flood)
    next_state = _state((0, 1), flood)

    agent = DynaQAgent(n_actions=2, planning_steps=0, seed=9)
    model_key = (state, 0)
    agent.model[model_key] = {
        (next_state, 1.0, False): 5,
        (next_state, 1.0, True): 1,
    }

    for _ in range(200):
        sampled_next_state, sampled_reward, sampled_terminated = agent._sample_model_transition(model_key)
        assert sampled_next_state == next_state
        assert sampled_reward == 1.0
        assert sampled_terminated is True
