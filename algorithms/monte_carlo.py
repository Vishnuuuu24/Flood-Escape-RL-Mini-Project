"""First-visit Monte Carlo control for tabular action values."""

from __future__ import annotations

from collections.abc import Sequence

from algorithms.base_agent import BaseTabularAgent, StateKey

EpisodeStep = tuple[StateKey, int, float]


class MonteCarloControl(BaseTabularAgent):
    """First-visit MC control with backward return computation."""

    def __init__(self, n_actions: int, **kwargs: float | int | None) -> None:
        """Initialize MC control agent and episodic return trackers."""
        super().__init__(n_actions=n_actions, **kwargs)
        self.episode_memory: list[EpisodeStep] = []
        self._returns_sum: dict[tuple[StateKey, int], float] = {}
        self._returns_count: dict[tuple[StateKey, int], int] = {}

    def start_episode(self) -> None:
        """Reset episodic transition memory."""
        self.episode_memory.clear()

    def record_transition(self, state: StateKey, action: int, reward: float) -> None:
        """Append one transition tuple (s, a, r) for the current episode."""
        self.episode_memory.append((state, action, float(reward)))

    def update_from_episode(self, episode: Sequence[EpisodeStep] | None = None) -> None:
        """Run first-visit MC updates over a full episode trajectory."""
        trajectory = list(self.episode_memory if episode is None else episode)
        visited: set[tuple[StateKey, int]] = set()
        return_so_far = 0.0

        for state, action, reward in reversed(trajectory):
            return_so_far = float(reward) + self.gamma * return_so_far
            state_action = (state, action)

            if state_action in visited:
                continue
            visited.add(state_action)

            if self.is_terminal_state(state):
                continue

            self._returns_sum[state_action] = self._returns_sum.get(state_action, 0.0) + return_so_far
            self._returns_count[state_action] = self._returns_count.get(state_action, 0) + 1
            average_return = self._returns_sum[state_action] / self._returns_count[state_action]
            self.q_values(state)[action] = average_return

        if episode is None:
            self.episode_memory.clear()

    def end_episode(self) -> tuple[float, float]:
        """Update from stored trajectory and decay hyperparameters."""
        self.update_from_episode()
        return self.decay_hyperparameters()
