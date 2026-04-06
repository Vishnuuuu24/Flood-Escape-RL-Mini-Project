📄 Product Requirements Document (PRD)

🏷️ Title

Reinforcement Learning in a Dynamic and Stochastic Flood Escape Environment: A Comparative Study of Monte Carlo and Temporal Difference Methods

⸻

1. 🎯 Objective

Design and implement a custom grid-based reinforcement learning environment where an agent must escape a dynamically flooding region under stochastic movement conditions.

The project aims to:
• Implement and compare:
• Monte Carlo Control
• TD Prediction
• SARSA (on-policy)
• Q-learning (off-policy)
• Analyze how different RL algorithms behave under:
• Dynamic hazards (flood spread)
• Stochastic transitions (movement uncertainty)

⸻

2. 🧩 Environment Overview

Grid Specification
• Size: 6 × 6 grid
• Fully observable environment
• Discrete state and action space

⸻

Entities
• Agent (A) → learns policy
• Goal (G) → terminal success state
• Flood Cells (F) → terminal failure states
• Empty Cells (.)

⸻

3. ⚙️ Core Features

⸻

🔥 Feature 1: Dynamic Flood (Spreading Hazard)

Description
• Flood starts from predefined cells
• Expands to neighboring cells over time (each step or probabilistically)

Implementation Logic
• At each timestep:
• Flood spreads to adjacent cells (up/down/left/right)
• Optional: probabilistic spread (e.g., 70% chance)

⸻

Impact on RL
• Environment becomes time-dependent
• Safe paths can become unsafe

Algorithm Behavior
• MC → High variance due to changing environment
• TD → Adapts incrementally
• SARSA → Learns safer paths (risk-aware)
• Q-learning → Chooses shortest but risky paths

⸻

🌪️ Feature 2: Stochastic Movement

Description
• Actions are not deterministic

Example:
• Intended move succeeds with 80% probability
• 20% → random adjacent move

⸻

Implementation Logic

if random.random() < 0.8:
    next_state = intended_move
else:
    next_state = random_valid_move


⸻

Impact on RL
• Introduces transition uncertainty

Algorithm Behavior
• MC → Noisy returns
• TD → Stable learning
• SARSA → Conservative (avoids risky zones)
• Q-learning → Optimistic (assumes ideal transitions)

⸻

4. 🧠 State Space Design

Representation (Chosen for performance)

👉 Reduced state representation
• Agent position: (x, y)
• Flood map (binary grid)

State = (agent_position, flood_state)

⸻

Why this choice?
• Fully observable → satisfies MDP assumption
• Keeps complexity manageable
• Works for all algorithms

⸻

5. 🎮 Action Space

Discrete actions:
• Up
• Down
• Left
• Right

⸻

6. 🎯 Reward Function (Option B)
• +10 → Reaches goal
• -10 → Enters flooded cell
• -0.1 → Each step (efficiency penalty)
• -2 → Near flood (adjacent to flood cell)

⸻

7. 🔁 Episode Definition

Episode ends when:
• Agent reaches goal ✅
• Agent enters flood ❌
• (Optional) max steps reached

⸻

8. 🏗️ System Architecture

Environment Framework

Use:
👉 Gymnasium

⸻

Custom Environment Design

Class: FloodEscapeEnv

Core Methods:

reset() → initial state
step(action) → next_state, reward, done, info
render() → visualization


⸻

Internal Components
• Grid manager
• Flood propagation module
• Transition model (stochastic movement)
• Reward calculator

⸻

9. 🤖 Algorithms to Implement

⸻

1. Monte Carlo Control
• Episodic learning
• Policy evaluation via returns
• High variance expected

⸻

2. TD Prediction (TD(0))
• Value function estimation
• Incremental updates

⸻

3. SARSA (On-policy)
• Updates based on actual action taken
• Risk-aware learning

⸻

4. Q-learning (Off-policy)
• Updates using max future reward
• Greedy optimal policy

⸻

10. 📊 Evaluation Metrics

All selected:
• 📈 Average reward vs episodes
• 🎯 Success rate (% reaching goal)
• ⏱️ Steps per episode
• 🧠 State-value tables (V(s))
• 📌 Policy visualization

⸻

11. 🧪 Experiment Design

Compare:
• MC vs TD (learning efficiency)
• SARSA vs Q-learning (risk behavior)

⸻

Key Observations to Extract:
• Convergence speed
• Policy stability
• Risk-taking vs safety
• Effect of stochasticity

⸻

12. 🖥️ Visualization Strategy

Phase 1: Console Output
• Grid printed per episode

Phase 2: Matplotlib
• Heatmaps (value function)
• Policy arrows
• Learning curves

Phase 3 (Optional): GUI
• Animated environment

⸻

13. 📂 Project Structure

project/
│
├── env/
│   └── flood_escape_env.py
│
├── algorithms/
│   ├── monte_carlo.py
│   ├── td_learning.py
│   ├── sarsa.py
│   └── q_learning.py
│
├── utils/
│   ├── visualization.py
│   └── helpers.py
│
├── experiments/
│   └── run_experiments.py
│
└── results/
    ├── plots/
    └── tables/


⸻

14. 🚀 Development Phases

Phase 1: Environment Setup
• Grid + agent + goal
• Basic transitions

⸻

Phase 2: Add Features
• Dynamic flood
• Stochastic movement

⸻

Phase 3: RL Algorithms
• Implement all four methods

⸻

Phase 4: Evaluation
• Run experiments
• Collect metrics

⸻

Phase 5: Visualization
• Graphs + policy plots

⸻

15. ⚠️ Risks & Mitigation

Risk Mitigation
Learning instability Tune rewards
State explosion Keep grid small (6×6)
Flood too aggressive Control spread probability
Sparse rewards Use shaping (-0.1, -2)


⸻

16. ✅ Success Criteria
• All 4 algorithms implemented
• Clear performance comparison
• Meaningful behavioral differences observed
• Visualizations generated

