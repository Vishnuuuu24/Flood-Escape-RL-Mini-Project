"""On-policy SARSA algorithm for tabular control."""

from __future__ import annotations

from algorithms.base_agent import BaseTabularAgent, StateKey


class SARSAAgent(BaseTabularAgent):
    """SARSA agent with strict on-policy temporal-difference updates."""

    def update(
        self,
        state: StateKey,
        action: int,
        reward: float,
        next_state: StateKey,
        next_action: int,
        done: bool,
    ) -> None:
        """Perform SARSA update using the actual action selected in next state."""
        if done:
            td_target = float(reward)
            self.mark_terminal_state(next_state)
        else:
            next_q = self.q_values(next_state)[next_action]
            td_target = float(reward) + self.gamma * next_q

        self.update_q_value(state, action, td_target)
