# Flood Escape RL Project - Comprehensive Breakdown

This document serves as a complete educational guide to the Flood Escape project. It explains the theory, the environment mechanics, the workflow, the algorithms used, the engineering challenges solved, and a detailed comparison of how the agents actually behave in practice. It also includes a line-by-line code breakdown for project evaluations.

---

## 1. The Core Concept: What is "Flood Escape"?

**Flood Escape** is a custom 6x6 Gridworld environment formulated as a **Markov Decision Process (MDP)**. 

Typical academic RL projects (like OpenAI's *Frozen Lake*) feature **static** environments where hazards (holes) never move. In Flood Escape, the hazard is **dynamic and stochastic**. 

*   **Objective:** The agent must navigate from a Start cell to a Goal cell to maximize its reward.
*   **The Hazard:** A "flood" starts at specific cells and stochastically spreads to adjacent cells at every time step. If the agent touches the flood, the episode fails.
*   **The Goal:** We compare 5 classic Tabular RL algorithms to evaluate how they handle a dynamic, shifting environment compared to a static one.

---

## 2. The Environment Setup & Mechanics (`env/flood_escape_env.py`)

The environment defines the rules of the world.

### The Turn Structure (1 Step):
1. **Action Phase:** The agent chooses an action (Up, Down, Left, Right). 
2. **Movement:** The agent moves (with a slight chance of "slipping" depending on stochasticity, mimicking slipping on water).
3. **Environment Evolution:** The flood performs a "Spreading Step". It checks adjacent safe cells and expands probabilistically.

### The Rewards (The Signal):
*   **`+10.0` (Positive):** Reaching the Goal cell (Victory / `terminated=True`).
*   **`-10.0` (Negative):** Touching a flooded cell (Death / `terminated=True`).
*   **`-0.1` (Negative Penalty):** Taking a normal safe step. This tiny punishment pushes the agent to find the *shortest* safe path instead of wandering aimlessly to delay the inevitable.

---

## 3. The Big Engineering Challenges (And Our Solutions)

Applying Tabular RL to a dynamic grid presents massive mathematical hurdles. Here is how we solved them:

### A. The Tabular State Space Explosion (The Trillion-State Problem)
*   **The Problem:** Because the flood moves, the state must include the flood map. A 6x6 grid has 36 cells. The number of possible flood configurations is **$2^{36}$** (over 68 billion). Combined with 36 agent positions, the total state space is **~2.4 Trillion states**. Tabular agents treat every unique flood layout as a brand-new state, meaning they can never revisit states to learn from past mistakes.
*   **Our Solution (State Reduction):** In `algorithms/base_agent.py`, instead of feeding the agent the entire 6x6 board, we engineered a **Local 3x3 Sensor**. The state key extracts only a 3x3 window of the flood around the agent, packs it into bits, and creates a tuple: `(Agent_Position, 3x3_Sensor_Bytes)`. 
*   **Impact:** The max theoretical states dropped from 2.4 Trillion to $36 \times 2^9 = 18,432$. The agents can now easily explore and converge.

### B. Truncation vs. Termination Bias 
*   **The Problem:** Originally, hitting the maximum step limit (timeout) and dying to the flood both triggered a generic `done = True` flag. The algorithms treated both as terminal states (future value = 0.0), punishing the agent for merely running out of time on a safe cell.
*   **Our Solution:** We rewrote the update loops to strictly separate `terminated` (win/loss) from `truncated` (timeout). We only set the bootstrap future value to `0.0` if `terminated=True`.

### C. Handling Stochasticity in Dyna-Q
*   **The Problem:** Standard Dyna-Q models assume the world is deterministic. Because our flood spreads randomly, taking Action A might lead to Safety or Death. The old code just overwrote its memory with the latest outcome, causing the planning model to hallucinate incorrectly.
*   **Our Solution:** We modified the Dyna-Q dictionary to store a *list* of all observed outcomes for a `(State, Action)`. During planning, it stochastically samples from this list, directly reflecting the true probability distribution of the environment.

---

## 4. Workflow: Training & Agent Behavior

The overarching workflow for every agent is contained in `experiments/run_experiments.py`. The agent goes through thousands of episodes to figure out the map.

### The Epsilon-Greedy Policy
The agent learns using an **$\epsilon$-greedy policy**:
*   **Exploration:** With probability $\epsilon$, the agent picks a totally random direction to discover new mechanics. It often leads to death, but helps map unknown states.
*   **Exploitation:** With probability $1-\epsilon$, the agent searches its Q-table/V-table and picks the best mathematical move.
*   As the 5,000 episodes progress, $\epsilon$ slowly decays towards `0`, so the agent stops exploring and starts utilizing its learned paths.

---

## 5. The 5 Algorithms & Their Results (Comparison)

All 5 algorithms calculate paths, but because their internal math differs, their *policies* (how they behave) drastically shift. 

### A. Monte Carlo (MC) Control
*   **Workflow:** Plays a full episode from start to finish. It memorizes the sequence. When the game ends, it calculates the true discounted total reward ($G_t$) backwards and drops it into its Q-Table.
*   **Formula:** $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ G_t - Q(S_t, A_t) \right]$
*   **Behavior (Result):** It learns very sluggishly in a dynamic world because it has to wait for an episode to end. If a flood spreads halfway through, MC might unfairly blame the first few steps of the run.

### B. TD(0) Prediction
*   **Workflow:** Not a control algorithm (doesn't learn Q-values for actions). It only learns the *Value* ($V(S)$) of being in a state using a 1-step lookahead. The policy is just derived by looking at adjacent safe states and moving there.
*   **Formula:** $V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$
*   **Behavior (Result):** Fast to learn state values, but struggles slightly with pure execution because it has to calculate moves on the fly by assuming the flood won't spread into its desired neighbor cell.

### C. SARSA (On-Policy TD Control)
*   **Workflow:** Updates its Q-table every single step. It calculates the value of the next state using the *exact action* it is going to take next (`next_action`), even if that action is a random exploratory mistake.
*   **Formula:** $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$
*   **Behavior (Result: Cautious pathing):** Because SARSA factors in its own exploration mistakes, its final policy is incredibly defensive. It will take long, wide arcs around the flood to ensure it doesn't accidentally slip into the water.

### D. Q-Learning (Off-Policy TD Control)
*   **Workflow:** Updates its Q-table every step. However, it completely ignores exploratory mistakes. It tells the Q-Table, "Assume I play mathematically perfectly from now on."
*   **Formula:** $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right]$
*   **Behavior (Result: Aggressive optimal pathing):** Q-learning will skirt right on the edge of the flood if it's technically the shortest path. It assumes optimal play, making it highly effective but risky in stochastic slippery settings.

### E. Dyna-Q (Model-Based RL)
*   **Workflow:** Blends Q-Learning with memory simulation. For every real step in the world, it hallucinates $N$ "planning steps" by recalling past experiences from its dictionary and updating the Q-table without moving in the real world.
*   **Behavior (Result: Data-efficient):** Because it constantly rehearses its memories, it requires far fewer real-world episodes to find the goal. It combines the aggressive optimality of Q-learning with massive computational speed.

---

## 6. Code-Level Line-by-Line Guide for Invigilators

Use this section to point directly to the core logic in the code files.

### 1. The Environment Engine (`env/flood_escape_env.py`)
*   **`def step(self, action):`** 
    Executes one MDP transition.
*   **`self._spread_flood()`** 
    The hazard logic. Calculates adjacent candidate targets and rolls `np.random.random() < self.flood_spread_prob` to dynamically spread the hazard per target cell.
*   **Reward Assignments:** 
    Returns `10.0` for reaching `self.goal_pos`, `-10.0` for stepping on `flood_grid == 1`, and `-0.1` otherwise.

### 2. State Space Reduction (`algorithms/base_agent.py`)
*   **`def observation_to_state_key(self, obs):`**
*   **`local_patch = padded_flood[x:x+3, y:y+3]`** 
    Slices a localized 3x3 view of the 6x6 flood array around the agent.
*   **`sensor_bytes = np.packbits(...).tobytes()`** 
    Packs those 9 binary 1s/0s into deterministic, compact bytes to shrink the state explosion.

### 3. Q-Learning Core Math (`algorithms/q_learning.py`)
*   **`def update(...):`**
*   **`target = reward`** (If `terminated` is True, future value is 0).
*   **`target = reward + self.gamma * np.max(self.q_table[next_state])`** (Bootstrapping max future value).
*   **`self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])`** (The TD update).

### 4. SARSA Core Math (`algorithms/sarsa.py`)
*   **`target = reward + self.gamma * self.q_table[next_state][next_action]`** 
    Differs from Q-Learning by using the actual chosen `next_action`.

### 5. Dyna-Q Architecture (`algorithms/dyna_q.py`)
*   **`self.model[state][action].append((reward, next_state, terminated))`** 
    Saves environmental stochastic outcomes into its memory dictionary.
*   **`def _planning_step(self):`** 
    The hallucination loop. It randomly selects a past `(state, action)`, samples an outcome from memory, and applies a Q-learning update to rehearse the experience.

### 6. The Training Loop (`experiments/run_experiments.py`)
*   **`while not done:`** (Loops until `terminated` or `truncated`).
*   **`action = agent.get_action(state_key)`** (Epsilon-greedy selection).
*   **`next_obs, reward, terminated, truncated, _ = env.step(action)`** (Interact with MDP).
*   **`agent.update(state_key, action, reward, next_state_key, terminated, truncated)`** (Trigger learning).
