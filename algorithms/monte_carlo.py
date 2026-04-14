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

    def update_from_episode(
        self,
        episode: Sequence[EpisodeStep] | None = None,
        *,
        terminal_state: StateKey | None = None,
        terminated: bool = False,
    ) -> None:
        """Run first-visit MC updates over a full episode trajectory."""
        trajectory = list(self.episode_memory if episode is None else episode)
        if not trajectory:
            if terminated and terminal_state is not None:
                self.mark_terminal_state(terminal_state)
            return

        # Phase 1: Compute all returns G_t efficiently via a backward pass.
        # We store only the return for the *first* visit to each state-action pair.
        first_visit_returns: dict[tuple[StateKey, int], float] = {}
        return_so_far = 0.0

        for state, action, reward in reversed(trajectory):
            return_so_far = float(reward) + self.gamma * return_so_far
            state_action = (state, action)
            # By overwriting in a backward pass as we move towards the start of the
            # trajectory, the final value for (s, a) will be its first-visit return.
            if not self.is_terminal_state(state):
                first_visit_returns[state_action] = return_so_far

        # Phase 2: Update episodic return trackers and Q-table using first-visit data.
        for state_action, G in first_visit_returns.items():
            state, action = state_action
            self._returns_sum[state_action] = self._returns_sum.get(state_action, 0.0) + G
            self._returns_count[state_action] = self._returns_count.get(state_action, 0) + 1
            average_return = self._returns_sum[state_action] / self._returns_count[state_action]
            self.q_values(state)[action] = average_return

        if terminated and terminal_state is not None:
            self.mark_terminal_state(terminal_state)

        if episode is None:
            self.episode_memory.clear()

    def end_episode(
        self,
        *,
        terminal_state: StateKey | None = None,
        terminated: bool = False,
    ) -> tuple[float, float]:
        """Update from stored trajectory and decay hyperparameters."""
        self.update_from_episode(terminal_state=terminal_state, terminated=terminated)
        return self.decay_hyperparameters()
