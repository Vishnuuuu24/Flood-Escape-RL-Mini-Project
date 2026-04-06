"""Model-based Dyna-Q agent for tabular control."""

from __future__ import annotations

import numpy as np

from algorithms.base_agent import BaseTabularAgent, StateKey


class DynaQAgent(BaseTabularAgent):
    """Dyna-Q agent combining real experience with planning updates.

    This agent performs a standard off-policy Q-learning update on each real
    transition, stores that transition in an internal model, and then executes
    additional planning backups by replaying model transitions.
    """

    def __init__(self, n_actions: int, *, planning_steps: int = 20, **kwargs: float | int | None) -> None:
        super().__init__(n_actions=n_actions, **kwargs)
        if planning_steps < 0:
            raise ValueError("planning_steps must be non-negative.")

        self.planning_steps = int(planning_steps)
        self.model: dict[tuple[StateKey, int], tuple[StateKey, float, bool]] = {}
        self._model_keys: list[tuple[StateKey, int]] = []

    def update(
        self,
        state: StateKey,
        action: int,
        reward: float,
        next_state: StateKey,
        done: bool,
    ) -> None:
        """Update from real transition and then perform planning updates."""
        self._apply_q_learning_backup(state, action, reward, next_state, done)
        self._store_transition(state, action, reward, next_state, done)
        self._run_planning_updates()

    def _apply_q_learning_backup(
        self,
        state: StateKey,
        action: int,
        reward: float,
        next_state: StateKey,
        done: bool,
    ) -> None:
        if done:
            td_target = float(reward)
            self.mark_terminal_state(next_state)
        else:
            td_target = float(reward) + self.gamma * float(np.max(self.q_values(next_state)))

        self.update_q_value(state, action, td_target)

    def _store_transition(
        self,
        state: StateKey,
        action: int,
        reward: float,
        next_state: StateKey,
        done: bool,
    ) -> None:
        model_key = (state, int(action))
        if model_key not in self.model:
            self._model_keys.append(model_key)
        self.model[model_key] = (next_state, float(reward), bool(done))

    def _run_planning_updates(self) -> None:
        if self.planning_steps == 0 or not self._model_keys:
            return

        for _ in range(self.planning_steps):
            sampled_index = int(self.rng.integers(0, len(self._model_keys)))
            sampled_state, sampled_action = self._model_keys[sampled_index]
            next_state, reward, done = self.model[(sampled_state, sampled_action)]
            self._apply_q_learning_backup(sampled_state, sampled_action, reward, next_state, done)
