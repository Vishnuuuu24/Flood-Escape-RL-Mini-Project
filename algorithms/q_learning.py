"""Off-policy Q-learning algorithm for tabular control."""

from __future__ import annotations

import numpy as np

from algorithms.base_agent import BaseTabularAgent, StateKey


class QLearningAgent(BaseTabularAgent):
    """Q-learning agent with strict off-policy max backup updates."""

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
        """Perform Q-learning update using max_a Q(s', a)."""
        if terminated:
            td_target = float(reward)
            self.mark_terminal_state(next_state)
        else:
            td_target = float(reward) + self.gamma * float(np.max(self.q_values(next_state)))

        self.update_q_value(state, action, td_target)
