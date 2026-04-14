"""Gymnasium environment for a stochastic flood-escape grid world."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Position = tuple[int, int]
Observation = dict[str, np.ndarray]


class FloodEscapeEnv(gym.Env[Observation, int]):
    """Custom 6x6 GridWorld where flood spreads dynamically over time.

    State is fully observable and includes:
    - agent position
    - binary flood map

    Actions:
    - 0: Up
    - 1: Down
    - 2: Left
    - 3: Right
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        *,
        max_steps: int = 100,
        move_success_prob: float = 0.8,
        flood_spread_prob: float = 0.7,
        goal_position: Position = (5, 5),
        initial_flood_cells: tuple[Position, ...] = ((0, 5),),
    ) -> None:
        """Initialize the environment.

        Args:
            max_steps: Maximum allowed steps per episode before truncation.
            move_success_prob: Probability of applying the intended action.
            flood_spread_prob: Probability for flood to spread to each adjacent cell.
            goal_position: Goal cell coordinates.
            initial_flood_cells: Starting flooded cells.
        """
        self.grid_size = 6
        self.max_steps = max_steps
        self.move_success_prob = move_success_prob
        self.flood_spread_prob = flood_spread_prob
        self.goal_position = goal_position
        self.initial_flood_cells = initial_flood_cells

        self._validate_static_positions()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32),
                "flood": spaces.MultiBinary((self.grid_size, self.grid_size)),
            }
        )

        self._action_deltas: dict[int, Position] = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        self.agent_position: Position = (0, 0)
        self.flood_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self._episode_steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the episode and return the initial observation.

        The agent spawn position is sampled so it never overlaps the goal or
        an initially flooded cell.
        """
        super().reset(seed=seed)

        self._episode_steps = 0
        self.flood_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for cell in self.initial_flood_cells:
            self.flood_map[cell] = 1

        self.agent_position = self._sample_agent_start()

        return self._get_observation(), {"goal_position": self.goal_position}

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Advance one timestep using stochastic movement and dynamic flood spread."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Expected one of [0, 1, 2, 3].")

        self._episode_steps += 1
        self.agent_position = self._sample_next_position(action)

        reward = -0.1
        terminated = False
        truncated = False

        if self.agent_position == self.goal_position:
            reward += 10.0
            terminated = True
        elif self.flood_map[self.agent_position] == 1:
            reward -= 10.0
            terminated = True
        elif self._is_adjacent_to_flood(self.agent_position):
            reward -= 2.0

        self._spread_flood()

        # Agent can still fail if flood reaches its current cell after spreading.
        if (
            not terminated
            and self.agent_position != self.goal_position
            and self.flood_map[self.agent_position] == 1
        ):
            reward -= 10.0
            terminated = True

        if not terminated and self._episode_steps >= self.max_steps:
            truncated = True

        observation = self._get_observation()
        info = {"steps": self._episode_steps}
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Print a compact text representation of the 6x6 grid."""
        grid = np.full((self.grid_size, self.grid_size), ".", dtype="<U1")
        grid[self.flood_map == 1] = "F"
        gx, gy = self.goal_position
        grid[gx, gy] = "G"
        ax, ay = self.agent_position
        grid[ax, ay] = "A"
        print("\n".join(" ".join(row) for row in grid))

    def _sample_next_position(self, action: int) -> Position:
        """Apply intended action with 0.8 default success, else random valid adjacent move."""
        if self.np_random.random() < self.move_success_prob:
            return self._apply_action(self.agent_position, action)

        neighbors = self._get_valid_neighbors(self.agent_position)
        sampled_index = int(self.np_random.integers(0, len(neighbors)))
        return neighbors[sampled_index]

    def _apply_action(self, position: Position, action: int) -> Position:
        """Apply action and clamp coordinates so they always remain in [0, 5]."""
        dx, dy = self._action_deltas[action]
        x, y = position
        new_x = int(np.clip(x + dx, 0, self.grid_size - 1))
        new_y = int(np.clip(y + dy, 0, self.grid_size - 1))
        return new_x, new_y

    def _get_valid_neighbors(self, position: Position) -> list[Position]:
        """Return orthogonally adjacent in-bounds coordinates."""
        x, y = position
        neighbors: list[Position] = []
        for dx, dy in self._action_deltas.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                neighbors.append((nx, ny))
        return neighbors

    def _is_adjacent_to_flood(self, position: Position) -> bool:
        """Check whether any orthogonally adjacent cell is flooded."""
        return any(self.flood_map[nx, ny] == 1 for nx, ny in self._get_valid_neighbors(position))

    def _spread_flood(self) -> None:
        """Spread flood to eligible target cells with one probability draw per cell.

        The goal cell is always protected and must never become flooded.
        """
        current_flood = self.flood_map.copy()
        updated_flood = current_flood.copy()
        goal_x, goal_y = self.goal_position

        candidate_targets: set[Position] = set()
        for fx, fy in np.argwhere(current_flood == 1):
            for nx, ny in self._get_valid_neighbors((int(fx), int(fy))):
                if (nx, ny) == self.goal_position or current_flood[nx, ny] == 1:
                    continue
                candidate_targets.add((nx, ny))

        # Evaluate each target once using the previous-step flood snapshot.
        for tx, ty in sorted(candidate_targets):
            has_flooded_neighbor = any(
                current_flood[nx, ny] == 1 for nx, ny in self._get_valid_neighbors((tx, ty))
            )
            if has_flooded_neighbor and self.np_random.random() < self.flood_spread_prob:
                updated_flood[tx, ty] = 1

        updated_flood[goal_x, goal_y] = 0
        self.flood_map = updated_flood

    def _sample_agent_start(self) -> Position:
        """Sample a valid non-terminal spawn cell (not goal, not flooded)."""
        valid_positions = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) != self.goal_position and self.flood_map[x, y] == 0
        ]
        if not valid_positions:
            raise RuntimeError("No valid spawn cells available for the agent.")

        sampled_index = int(self.np_random.integers(0, len(valid_positions)))
        return valid_positions[sampled_index]

    def _validate_static_positions(self) -> None:
        """Validate static goal/flood configuration for the fixed 6x6 grid."""
        gx, gy = self.goal_position
        if not (0 <= gx < self.grid_size and 0 <= gy < self.grid_size):
            raise ValueError("goal_position must be inside the 6x6 grid.")

        for fx, fy in self.initial_flood_cells:
            if not (0 <= fx < self.grid_size and 0 <= fy < self.grid_size):
                raise ValueError("initial_flood_cells contains out-of-bounds coordinates.")
            if (fx, fy) == self.goal_position:
                raise ValueError("initial_flood_cells must not include the goal cell.")

    def _get_observation(self) -> Observation:
        """Build the current observation dictionary."""
        return {
            "agent": np.array(self.agent_position, dtype=np.int32),
            "flood": self.flood_map.copy(),
        }
