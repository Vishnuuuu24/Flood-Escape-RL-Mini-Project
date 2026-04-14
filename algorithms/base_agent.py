"""Shared tabular RL utilities and base agent implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np

StateKey: TypeAlias = tuple[tuple[int, int], bytes]
ObservationLike: TypeAlias = Mapping[str, Any]
LOCAL_SENSOR_RADIUS = 1


def observation_to_state_key(observation: ObservationLike) -> StateKey:
    """Convert an observation dict into a robust hashable state key.

    Expected input format is:
    ``{"agent": array_like([x, y]), "flood": array_like(grid)}``
    """
    if "agent" not in observation or "flood" not in observation:
        raise KeyError("Observation must contain 'agent' and 'flood' keys.")

    agent_array = np.asarray(observation["agent"], dtype=np.int64).reshape(-1)
    if agent_array.size < 2:
        raise ValueError("Observation['agent'] must include at least two coordinates.")

    agent_position = (int(agent_array[0]), int(agent_array[1]))
    flood_array = np.asarray(observation["flood"], dtype=np.uint8)
    if flood_array.ndim != 2:
        raise ValueError("Observation['flood'] must be a 2D grid.")

    x, y = agent_position
    height, width = flood_array.shape
    if not (0 <= x < height and 0 <= y < width):
        raise ValueError("Observation['agent'] must be inside the flood grid bounds.")

    # Use a local 3x3 flood neighborhood centered at the agent to reduce state-space.
    pad = LOCAL_SENSOR_RADIUS
    padded_flood = np.pad(flood_array, pad_width=pad, mode="constant", constant_values=0)
    local_patch = padded_flood[x : x + (2 * pad + 1), y : y + (2 * pad + 1)]
    local_bits = np.ascontiguousarray(local_patch, dtype=np.uint8).reshape(-1)
    local_sensor = np.packbits(local_bits, bitorder="little").tobytes()

    return agent_position, local_sensor


class BaseTabularAgent:
    """Base class for dictionary-backed tabular action-value RL agents."""

    def __init__(
        self,
        n_actions: int,
        *,
        gamma: float = 0.99,
        alpha: float = 0.1,
        alpha_decay: float = 1.0,
        min_alpha: float = 0.01,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
        seed: int | None = None,
    ) -> None:
        """Initialize common tabular RL parameters and state containers."""
        if n_actions <= 0:
            raise ValueError("n_actions must be positive.")

        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.alpha_decay = float(alpha_decay)
        self.min_alpha = float(min_alpha)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.min_epsilon = float(min_epsilon)
        self.rng = np.random.default_rng(seed)

        self.q_table: dict[StateKey, np.ndarray] = {}
        self.terminal_states: set[StateKey] = set()

    def _ensure_state_row(self, state: StateKey) -> np.ndarray:
        """Return Q(s, :) and initialize unseen states lazily."""
        if state in self.terminal_states:
            row = self.q_table.get(state)
            if row is None:
                row = np.zeros(self.n_actions, dtype=np.float64)
                self.q_table[state] = row
            return row

        row = self.q_table.get(state)
        if row is None:
            row = np.zeros(self.n_actions, dtype=np.float64)
            self.q_table[state] = row
        return row

    def q_values(self, state: StateKey) -> np.ndarray:
        """Return action values Q(s, :) for a given state."""
        return self._ensure_state_row(state)

    def mark_terminal_state(self, state: StateKey) -> None:
        """Track terminal states and force their Q rows to remain all zeros."""
        self.terminal_states.add(state)
        self.q_table[state] = np.zeros(self.n_actions, dtype=np.float64)

    def is_terminal_state(self, state: StateKey) -> bool:
        """Check whether a state is terminal."""
        return state in self.terminal_states

    def select_action(self, state: StateKey, *, explore: bool = True) -> int:
        """Pick an action via epsilon-greedy policy over Q(s, :)."""
        if self.is_terminal_state(state):
            return 0

        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        row = self._ensure_state_row(state)
        max_value = float(np.max(row))
        greedy_actions = np.flatnonzero(np.isclose(row, max_value))
        chosen_index = int(self.rng.integers(len(greedy_actions)))
        return int(greedy_actions[chosen_index])

    def update_q_value(self, state: StateKey, action: int, target: float) -> None:
        """Apply one-step update to Q(s, a) unless s is terminal."""
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions - 1}].")

        if self.is_terminal_state(state):
            return

        row = self._ensure_state_row(state)
        row[action] += self.alpha * (float(target) - row[action])

    def decay_epsilon(self) -> float:
        """Apply multiplicative epsilon decay with a minimum floor."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def decay_alpha(self) -> float:
        """Apply multiplicative learning-rate decay with a minimum floor."""
        self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)
        return self.alpha

    def decay_hyperparameters(self) -> tuple[float, float]:
        """Decay alpha and epsilon together, returning updated values."""
        return self.decay_alpha(), self.decay_epsilon()

    def update(
        self,
        state: StateKey,
        action: int,
        reward: float,
        next_state: StateKey,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Update value estimates from one transition.

        Implementations should treat only ``terminated=True`` as a true terminal
        transition for bootstrapping targets. ``truncated`` indicates a
        time-limit style episode cutoff and should not force a zero bootstrap
        target by itself.
        """
        raise NotImplementedError("Subclasses must implement update().")
