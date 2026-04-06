# Flood Escape Project

Phase 1 baseline scaffold for a 6x6 Flood Escape reinforcement learning project.

## Project Scope

This repository initializes the collaboration baseline for:
- A custom Gymnasium environment with dynamic flood spread and stochastic movement.
- RL algorithms to be implemented in later phases:
  - Monte Carlo Control
  - TD(0)
  - SARSA
  - Q-learning
- Shared utilities, experiments, and results directories for repeatable evaluation.

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
