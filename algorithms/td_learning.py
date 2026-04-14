"""TD(0) prediction for tabular state-value estimation."""

from __future__ import annotations

from algorithms.base_agent import StateKey


class TDPrediction:
    """Tabular TD(0) value estimator with dynamic state discovery."""

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        alpha: float = 0.1,
        alpha_decay: float = 1.0,
        min_alpha: float = 0.01,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
    ) -> None:
        """Initialize TD(0) hyperparameters and value table."""
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.alpha_decay = float(alpha_decay)
        self.min_alpha = float(min_alpha)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.min_epsilon = float(min_epsilon)

        self.v_table: dict[StateKey, float] = {}
        self.terminal_states: set[StateKey] = set()

    def value(self, state: StateKey) -> float:
        """Return V(s), lazily initializing unseen states to zero."""
        if state in self.terminal_states:
            return 0.0
        return self.v_table.setdefault(state, 0.0)

    def mark_terminal_state(self, state: StateKey) -> None:
        """Track terminal state and force its value to zero."""
        self.terminal_states.add(state)
        self.v_table[state] = 0.0

    def is_terminal_state(self, state: StateKey) -> bool:
        """Check whether a state is terminal."""
        return state in self.terminal_states

    def update(
        self,
        state: StateKey,
        reward: float,
        next_state: StateKey,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Apply one TD(0) update: V(s) <- V(s) + alpha * (target - V(s))."""
        if self.is_terminal_state(state):
            return

        if terminated:
            bootstrap = 0.0
            self.mark_terminal_state(next_state)
        else:
            bootstrap = self.value(next_state)

        td_target = float(reward) + self.gamma * bootstrap
        current_value = self.value(state)
        self.v_table[state] = current_value + self.alpha * (td_target - current_value)

    def decay_alpha(self) -> float:
        """Apply multiplicative learning-rate decay with a floor."""
        self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)
        return self.alpha

    def decay_epsilon(self) -> float:
        """Apply multiplicative exploration decay with a floor."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def decay_hyperparameters(self) -> tuple[float, float]:
        """Decay both alpha and epsilon, returning updated values."""
        return self.decay_alpha(), self.decay_epsilon()
