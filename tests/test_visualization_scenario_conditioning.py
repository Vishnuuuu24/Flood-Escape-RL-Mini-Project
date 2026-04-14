"""Unit tests for scenario-conditioned policy/value extraction in plots."""

from __future__ import annotations

import numpy as np
import pytest
from algorithms.base_agent import observation_to_state_key
from experiments.run_experiments import _derive_policy_from_q_table
from utils.visualization import (
    _extract_scenario_q_values,
    _extract_scenario_scalar_values,
    _resolve_scenario_flood_map,
)


def _state_key(position: tuple[int, int], flood_map: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return observation_to_state_key(
        {
            "agent": np.asarray(position, dtype=np.int64),
            "flood": flood_map,
        }
    )


def test_scalar_value_extraction_omission_uses_deterministic_base_fallback() -> None:
    grid_size = 6
    position = (0, 4)

    base_fallback_scenario = np.zeros((grid_size, grid_size), dtype=np.uint8)
    alternate_scenario = np.zeros((grid_size, grid_size), dtype=np.uint8)
    alternate_scenario[0, 5] = 1

    base_state = _state_key(position, base_fallback_scenario)
    alternate_state = _state_key(position, alternate_scenario)
    assert base_state != alternate_state

    table = {
        base_state: 10.0,
        alternate_state: -2.0,
    }

    default_scenario = _resolve_scenario_flood_map(None, grid_size)
    default_values = _extract_scenario_scalar_values(table, grid_size, default_scenario)
    assert default_values[position] == pytest.approx(10.0)

    conditioned_base = _extract_scenario_scalar_values(table, grid_size, base_fallback_scenario)
    conditioned_alt = _extract_scenario_scalar_values(table, grid_size, alternate_scenario)

    assert conditioned_base[position] == pytest.approx(10.0)
    assert conditioned_alt[position] == pytest.approx(-2.0)


def test_q_policy_derivation_omission_uses_deterministic_base_fallback() -> None:
    grid_size = 6
    position = (0, 4)

    base_fallback_scenario = np.zeros((grid_size, grid_size), dtype=np.uint8)
    alternate_scenario = np.zeros((grid_size, grid_size), dtype=np.uint8)
    alternate_scenario[0, 5] = 1

    base_state = _state_key(position, base_fallback_scenario)
    alternate_state = _state_key(position, alternate_scenario)

    q_table = {
        base_state: np.array([8.0, 0.0, 0.0, 0.0], dtype=np.float64),
        alternate_state: np.array([0.0, 9.0, 0.0, 0.0], dtype=np.float64),
    }

    default_policy = _derive_policy_from_q_table(q_table, grid_size)
    base_policy = _derive_policy_from_q_table(q_table, grid_size, scenario_flood_map=base_fallback_scenario)
    alt_policy = _derive_policy_from_q_table(q_table, grid_size, scenario_flood_map=alternate_scenario)

    assert default_policy[position] == 0
    assert base_policy[position] == 0
    assert alt_policy[position] == 1

    conditioned_q = _extract_scenario_q_values(q_table, grid_size, base_fallback_scenario)
    assert np.array_equal(conditioned_q[position], q_table[base_state])


def test_q_policy_derivation_omission_does_not_mix_for_missing_base_fallback_key() -> None:
    grid_size = 6
    position = (3, 3)

    base_fallback_scenario = np.zeros((grid_size, grid_size), dtype=np.uint8)
    alternate_scenario = np.zeros((grid_size, grid_size), dtype=np.uint8)
    alternate_scenario[2, 3] = 1

    base_state = _state_key(position, base_fallback_scenario)
    alternate_state = _state_key(position, alternate_scenario)
    assert base_state != alternate_state

    q_table = {
        alternate_state: np.array([0.0, 0.0, 9.0, 0.0], dtype=np.float64),
    }

    default_policy = _derive_policy_from_q_table(q_table, grid_size)
    base_policy = _derive_policy_from_q_table(
        q_table,
        grid_size,
        scenario_flood_map=base_fallback_scenario,
    )

    assert position not in default_policy
    assert position not in base_policy
