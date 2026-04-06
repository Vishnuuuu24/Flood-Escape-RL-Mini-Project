"""Verification tests for Phase 3 RL algorithm correctness guarantees."""

from __future__ import annotations

import numpy as np
import pytest
from algorithms import QLearningAgent, SARSAAgent, observation_to_state_key


def _state(agent_xy: tuple[int, int], flood_grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    """Build canonical hashable StateKey fixtures for tests."""
    return (agent_xy, np.ascontiguousarray(flood_grid.astype(np.uint8)).tobytes())


@pytest.mark.parametrize("agent_cls", [QLearningAgent, SARSAAgent])
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

    agent.update(state=state, action=0, reward=2.0, next_state=next_state, done=False)

    expected = 2.0 + 0.9 * 10.0
    assert agent.q_values(state)[0] == pytest.approx(expected)


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
        done=False,
    )

    expected = 2.0 + 0.9 * 3.0
    assert agent.q_values(state)[0] == pytest.approx(expected)


def test_observation_to_state_key_accepts_numpy_arrays_and_is_hashable() -> None:
    observation = {
        "agent": np.array([2, 4], dtype=np.int32),
        "flood": np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8),
    }

    key = observation_to_state_key(observation)

    hash(key)
    assert isinstance(key, tuple)
    assert isinstance(key[0], tuple)
    assert isinstance(key[1], bytes)


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
