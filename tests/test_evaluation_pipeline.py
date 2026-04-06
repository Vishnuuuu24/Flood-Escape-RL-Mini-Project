"""Integration checks for Phase 4/5 evaluation pipeline behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from experiments.run_experiments import run_all_experiments


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
    }
    assert set(metrics_by_algo) == expected_algorithms

    expected_files = {
        "learning_curves.png",
        "learning_montecarlocontrol.png",
        "learning_tdprediction.png",
        "learning_sarsaagent.png",
        "learning_qlearningagent.png",
        "steps_comparison.png",
        "summary_metrics.png",
        "value_heatmap_montecarlocontrol.png",
        "policy_montecarlocontrol.png",
        "value_heatmap_tdprediction.png",
        "value_heatmap_sarsaagent.png",
        "policy_sarsaagent.png",
        "value_heatmap_qlearningagent.png",
        "policy_qlearningagent.png",
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
