# Flood Escape Project

Phase 1 baseline scaffold for a 6x6 Flood Escape reinforcement learning project.

> 📖 **Educational Guide:** A complete theoretical and code-level breakdown of the environment, the 5 algorithms, and the engineering challenges solved in this project is available in [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md).

## Project Scope

This repository initializes the collaboration baseline for:
- A custom Gymnasium environment with dynamic flood spread and stochastic movement.
- RL algorithms to be implemented in later phases:
  - Monte Carlo Control
  - TD(0)
  - SARSA
  - Q-learning
  - Dyna-Q (model-based planning + replay)
- Shared utilities, experiments, and results directories for repeatable evaluation.

## Environment Notes

- State representation uses a local flood sensor with radius 1 (3x3 neighborhood) around the agent.
- This intentionally keeps the tabular state space manageable, but makes the task partially observable.
- Default flood spread probability is 0.5 to balance hazard pressure and learning signal on the 6x6 grid.
- You can still override `flood_spread_prob` and `move_success_prob` per experiment.

## Directory Structure

```
Flood_Escape_Project/
  PRD.md
  README.md
  .gitignore
  requirements.txt
  pyproject.toml
  env/
    __init__.py
  algorithms/
    __init__.py
  utils/
    __init__.py
  experiments/
    __init__.py
    run_experiments.py
  tests/
    __init__.py
    test_scaffold.py
  results/
    plots/
      .gitkeep
    tables/
      .gitkeep
```

## Prerequisites

- Python 3.11+
- pip

## Setup

1. Create and activate the project virtual environment (`.venv-flood-escape`).

```bash
python3 -m venv .venv-flood-escape
source .venv-flood-escape/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv-flood-escape
.venv-flood-escape\Scripts\Activate.ps1
```

2. Upgrade pip.

```bash
python -m pip install --upgrade pip
```

3. Install pinned dependencies.

```bash
pip install -r requirements.txt
```

4. Re-activate environment in new terminal sessions before running project commands.

```bash
source .venv-flood-escape/bin/activate
```

On Windows (PowerShell):

```powershell
.venv-flood-escape\Scripts\Activate.ps1
```

## How to Run the Experiments

Use these commands from the repository root (`Flood_Escape_Project/`).

macOS/Linux (activate `.venv-flood-escape`):

```bash
source .venv-flood-escape/bin/activate
```

Windows PowerShell (activate `.venv-flood-escape`):

```powershell
.venv-flood-escape\Scripts\Activate.ps1
```

Run full training for all algorithms:

```bash
python -m experiments.run_experiments --episodes 5000
```

Generated plot artifacts are written to:

```text
results/plots/
```

Notable visual outputs include:
- `policy_grid_<algorithm>.png` (terminal-style policy rendered as image)
- `policy_<algorithm>.png` (arrow policy field)
- `environment_rollouts.png` (agent trajectories over final flood states)

## Algorithms Used

- Monte Carlo Control:
Learns from complete episodes by computing returns backward from terminal states. Good baseline for episodic learning.

- TD(0) Prediction:
Learns state values incrementally at every step. More sample-efficient than pure Monte Carlo for value estimation.

- SARSA (On-policy):
Updates using the next action actually selected by the current policy. Usually safer in risky environments.

- Q-learning (Off-policy):
Updates using the maximum next-state action value, regardless of exploratory action. Often learns aggressive shortest-path behavior.

- Dyna-Q (Added Advanced Method):
Combines Q-learning with a learned transition model and planning updates. After each real step, it replays model transitions to learn faster from mistakes and successes. In this project, this is the strongest additional tabular upgrade for sample efficiency.

## Run

Run the scaffold entry point:

```bash
python -m experiments.run_experiments
```

Expected output:

```text
Flood Escape scaffold is ready. Implement algorithms in the next phase.
```

## Lint And Format

Run Ruff lint checks:

```bash
ruff check .
```

Apply Ruff formatting:

```bash
ruff format .
```

## Test

Run test suite:

```bash
pytest
```

The current tests are smoke tests that validate package imports and core dependency availability.

## Reproducibility

- Keep dependencies pinned in `requirements.txt`.
- Use fixed random seeds in future training scripts and record seed values in experiment outputs.
- Store generated plots/tables in `results/plots/` and `results/tables/` locally; only `.gitkeep` placeholders are tracked.

## Contributing

- Create a feature branch per task.
- Run lint and tests before opening a pull request:

```bash
ruff check .
pytest
```

- Keep changes focused and document any behavior changes in PR notes.

## Notes

- `results/plots/` and `results/tables/` are scaffolded with `.gitkeep` files.
- Generated outputs in these folders are gitignored by default.
- This phase focuses on project foundation only; environment dynamics and RL training logic are planned for subsequent phases.
