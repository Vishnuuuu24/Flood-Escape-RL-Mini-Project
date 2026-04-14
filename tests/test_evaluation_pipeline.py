"""Integration checks for Phase 4/5 evaluation pipeline behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from algorithms import TDPrediction, observation_to_state_key
from experiments.run_experiments import _episode_seed, _is_success, _td_behavior_action, run_all_experiments


def test_run_all_experiments_micro_training_artifacts_and_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "plots"

    metrics_by_algo, generated_plots = run_all_experiments(
        episodes=5,
        seed=123,
        smooth_window=2,
        output_dir=output_dir,
    )

    expected_algorithms = {
        "MonteCarloControl",
        "TDPrediction",
        "SARSAAgent",
        "QLearningAgent",
        "DynaQAgent",
    }
    assert set(metrics_by_algo) == expected_algorithms

    expected_files = {
        "learning_curves.png",
        "learning_montecarlocontrol.png",
        "learning_tdprediction.png",
        "learning_sarsaagent.png",
        "learning_qlearningagent.png",
        "learning_dynaqagent.png",
        "policy_grid_montecarlocontrol.png",
        "policy_grid_tdprediction.png",
        "policy_grid_sarsaagent.png",
        "policy_grid_qlearningagent.png",
        "policy_grid_dynaqagent.png",
        "environment_rollouts.png",
        "steps_comparison.png",
        "summary_metrics.png",
        "value_heatmap_montecarlocontrol.png",
        "policy_montecarlocontrol.png",
        "value_heatmap_tdprediction.png",
        "value_heatmap_sarsaagent.png",
        "policy_sarsaagent.png",
        "value_heatmap_qlearningagent.png",
        "policy_qlearningagent.png",
        "value_heatmap_dynaqagent.png",
        "policy_dynaqagent.png",
    }

    assert output_dir.exists()
    assert output_dir.is_dir()

    produced_files = {path.name for path in output_dir.glob("*.png")}
    assert expected_files.issubset(produced_files)
    assert {path.name for path in generated_plots} == expected_files

    for filename in expected_files:
        artifact = output_dir / filename
        assert artifact.exists()
        assert artifact.is_file()
        assert artifact.stat().st_size > 0

    metric_keys = ("reward_per_episode", "success_per_episode", "steps_per_episode")
    for metrics in metrics_by_algo.values():
        for key in metric_keys:
            values = np.asarray(metrics[key], dtype=np.float64)
            assert values.shape == (5,)
            assert np.all(np.isfinite(values))

    policy_report = output_dir.parent / "tables" / "policies_report.txt"
    assert policy_report.exists()
    assert policy_report.is_file()
    assert policy_report.stat().st_size > 0

    report_text = policy_report.read_text(encoding="utf-8")
    for algo_name in expected_algorithms:
        assert algo_name in report_text
    assert "Scenario-conditioned (base reset flood:" in report_text
    assert report_text.count("Scenario-conditioned (base reset flood:") >= len(expected_algorithms)


def test_policy_report_includes_phase3_sparsity_and_coverage_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "plots"

    run_all_experiments(
        episodes=4,
        seed=7,
        smooth_window=2,
        output_dir=output_dir,
    )

    policy_report = output_dir.parent / "tables" / "policies_report.txt"
    report_text = policy_report.read_text(encoding="utf-8")

    grid_size = 6
    local_sensor_bits = 9
    old_cap = (grid_size**2) * (2 ** (grid_size**2))
    new_cap = (grid_size**2) * (2**local_sensor_bits)

    assert "Phase 3 Convergence Sparsity and Coverage" in report_text
    assert "Old cap (full flood map, 36 bits):" in report_text
    assert str(old_cap) in report_text
    assert "New cap (local sensor 3x3, 9 bits):" in report_text
    assert str(new_cap) in report_text
    assert "Practical occupancy ratio (table/new cap):" in report_text
    assert "Unique states visited:" in report_text
    assert "Table rows:" in report_text

    for algo_name in (
        "MonteCarloControl",
        "TDPrediction",
        "SARSAAgent",
        "QLearningAgent",
        "DynaQAgent",
    ):
        assert f"{algo_name} sparsity" in report_text


def test_td_behavior_action_uses_position_specific_sensor_keys() -> None:
    flood = np.zeros((6, 6), dtype=np.uint8)
    flood[2, 2] = 1

    observation = {
        "agent": np.array([2, 1], dtype=np.int32),
        "flood": flood,
    }

    td_agent = TDPrediction()
    td_agent.v_table[observation_to_state_key({"agent": np.array([1, 1]), "flood": flood})] = 10.0
    td_agent.v_table[observation_to_state_key({"agent": np.array([3, 1]), "flood": flood})] = 1.0
    td_agent.v_table[observation_to_state_key({"agent": np.array([2, 0]), "flood": flood})] = 2.0
    td_agent.v_table[observation_to_state_key({"agent": np.array([2, 2]), "flood": flood})] = 3.0

    action = _td_behavior_action(
        observation,
        td_agent,
        np.random.default_rng(0),
        epsilon=0.0,
        move_success_prob=0.8,
    )

    assert action == 0


def test_success_detection_counts_goal_even_if_truncated_flag_is_set() -> None:
    goal_position = (5, 5)
    final_state = (goal_position, b"\x00")

    assert (
        _is_success(
            terminated=False,
            truncated=True,
            final_state=final_state,
            goal_position=goal_position,
        )
        == 1
    )


def test_episode_seed_is_algorithm_invariant_for_fair_comparison() -> None:
    base_seed = 42
    episode_index = 11

    assert _episode_seed(base_seed, 0, episode_index) == _episode_seed(base_seed, 1, episode_index)
    assert _episode_seed(base_seed, 1, episode_index) == _episode_seed(base_seed, 4, episode_index)

    assert _episode_seed(base_seed, 0, episode_index + 1) != _episode_seed(base_seed, 0, episode_index)
