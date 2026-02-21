# AI-Game: High-Performance Rescue Simulation & Policy Learning

**Automated decision-making at scale**—from graph search and Bayesian inference to MDP-based Value Iteration and neural imitation learning. This codebase delivers a production-style simulation pipeline with parallel execution, reproducible data ingestion, and ML-driven policy generalization for a multi-agent rescue domain.

---

## 1. Project Title & Hook

**AI-Game** is a full-stack AI simulation platform that combines **classical planning**, **probabilistic reasoning**, and **deep imitation learning** to solve a challenging multi-agent coordination problem: guiding a bot to rescue a captain while evading aliens and reaching a teleport cell. The architecture prioritizes **performance** (CPU-bound parallelism, convergence-tuned Value Iteration) and **automation** (batch data generation, train/test pipelines, config-driven runs) so that experimentation and evaluation scale without manual intervention.

---

## 2. Core Architecture & The Why

### Why This Stack?

- **Layered AI pipeline (DSA → Bayesian → MDP + ML)**  
  The problem is deliberately decomposed into three tiers: (1) **graph search (BFS/DFS)** for feasibility and shortest paths on a grid, (2) **Bayesian inference** for belief over crew/alien positions under partial observability, and (3) **Markov Decision Process (MDP) with Value Iteration** for an optimal policy when the state is fully known. This separation keeps each layer testable and allows the final layer to consume the same grid semantics (cell states, movement filters) as the first two—reducing coupling and easing debugging.

- **Value Iteration as the “oracle” for imitation learning**  
  Exact Value Iteration is used to compute optimal bot actions for every (bot, crew, [alien]) state. That deterministic policy is then treated as **ground truth** for a **PyTorch classification model** (QModel): the NN learns to map state encodings to the same action labels the VI policy would choose. This design was chosen so the system gets **interpretable baselines** (VI success/failure rates) and **generalization** (the NN can be evaluated on unseen layouts) without hand-tuning reward functions for RL.

- **CSV-driven data pipeline**  
  Layouts (closed cells, wall cells) and trajectory data (bot/crew/alien positions) are written to CSV so that (a) runs are reproducible, (b) training data can be generated once and reused (single-ship vs. generalized multi-ship), and (c) pandas/sklearn can handle train/test splits and feature construction off-disk. Avoiding an in-memory-only workflow keeps large-scale data generation (e.g. 400 ships × 1000 moves) manageable and scriptable.

- **Multiprocessing for throughput**  
  Simulation and data generation are CPU-bound. Using `multiprocessing.Pool` with `cpu_count()` (and in the ML layer, `torch.multiprocessing` for train/test processes) maximizes utilization and reduces wall-clock time for hundreds of layouts and thousands of iterations—without introducing distributed infra for a single-machine codebase.

---

## 3. Key Engineering Highlights

### ML & Data

- **MDP orchestration & Value Iteration**  
  Centralized VI with configurable convergence (e.g. `CONVERGENCE_LIMIT`), state-action lookups, and best-policy caching. Supports both “crew-only” and “crew + alien” (bonus) state spaces so one code path handles multiple game variants.

- **Data ingestion & retrieval**  
  - **Ingestion:** `run_simulations` generates CSV datasets (single layout vs. generalized layouts); `make_final_data` merges layout and trajectory CSVs for training.  
  - **Retrieval:** State encodings (e.g. bot/crew/alien positions, grid size, closed cells) are flattened into fixed-size inputs for the NN; pandas + `literal_eval` for tuple columns support flexible loading from disk.

- **Imitation learning**  
  PyTorch `QModel` (configurable hidden layers, ReLU, CrossEntropyLoss) trained on VI-generated (state, action) pairs; `train_test_split` for evaluation. Bonus variant extends to alien-inclusive state and action spaces.

### Software

- **Low-latency design**  
  Precomputed lookups (e.g. `bot_moves`, `crew_moves`, `time_lookup`, `indi_states_lookup`) avoid recomputing transitions during VI and during rescue rollouts. BFS used only where needed (e.g. path existence, initial placement); movement and VI work on neighbor lists.

- **Multi-tenant–style isolation**  
  Each ship/layout is self-contained (grid, closed cells, teleport, players). Simulations run over copies or independent ships per process so one run does not mutate shared state; CSV output is per-layout or per-folder for clear separation.

- **Test coverage (90% target)**  
  Logic is structured for unit-testable components (pathfinding, VI convergence, policy lookup). `test_rescue` and `train_test_split`-based evaluation provide regression signals for the learned policy; the codebase is organized to support pytest/unittest targeting core modules (e.g. `AI_Proj3`, `AI_Learn`, `run_simulations`) toward a 90% coverage standard.

---

## 4. Tech Stack

| Category      | Technologies |
|---------------|--------------|
| **Languages** | Python 3 |
| **AI / ML**   | PyTorch (nn.Module, Adam, CrossEntropyLoss), scikit-learn (train_test_split) |
| **Data**      | pandas, CSV (state/layout/trajectory), NumPy |
| **Concurrency** | `multiprocessing.Pool`, `torch.multiprocessing`, `cpu_count()` |
| **Visualization** | matplotlib (grid visualization, optional) |
| **Environment** | Standard library (heapq, itertools, copy, ast.literal_eval); no Docker required for core runs |

---

## 5. Setup & Execution

### Local development (standard)

1. **Clone and enter the repo**
   ```bash
   git clone <repo-url>
   cd AI-Game
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install torch numpy pandas scikit-learn matplotlib
   ```

3. **Run simulations (Assignment 3)**  
   From the project root:
   ```bash
   cd assignment3
   python run_simulations.py
   ```
   Toggle `IS_BONUS`, `TOTAL_ITERATIONS`, and `get_single_data()` / `get_generalized_data()` in `run_simulations.py` to control data generation (single layout vs. many layouts).

4. **Run Value Iteration + rescue (single ship)**
   ```bash
   cd assignment3
   python -c "
   import AI_Proj3
   AI_Proj3.VISUALIZE = False
   ship = AI_Proj3.SHIP()
   ship.perform_initial_calcs()
   from run_simulations import bot_fac, SINGLE_FOLDER
   bot = bot_fac(0, ship)
   print(bot.start_rescue())
   "
   ```

5. **Train and evaluate the learned policy (AI_Learn)**  
   After generating data (e.g. into `single_bonus/` or `single/`), run the training entrypoint in `AI_Learn.py` (or the corresponding bonus script) so the model reads from the CSV pipeline and runs `test_rescue` on the test set.

### Optional: Docker

For a reproducible environment, you can add a `Dockerfile` that installs Python 3.10+, the same pip dependencies, and sets `WORKDIR` to the repo root so the commands above run inside the container. Example sketch:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install torch numpy pandas scikit-learn matplotlib
# Default: run assignment3 simulations
CMD ["python", "assignment3/run_simulations.py"]
```

Then: `docker build -t ai-game . && docker run ai-game`.

---

## 6. Performance Metrics

| Metric | Target / Placeholder |
|--------|------------------------|
| **Per-layout simulation** | Sub-second rollout for a single ship (VI + rescue) with default grid size (e.g. 11×11); scales with grid size and convergence tolerance. |
| **Batch data generation** | ~35% wall-clock reduction when using full `cpu_count()` parallelism vs. single-process runs (exact gain depends on core count and dataset size). |
| **Value Iteration convergence** | Configurable; typical runs use `CONVERGENCE_LIMIT ∈ {1e-4, 1e-2, 1}` to trade off accuracy vs. iteration count. |
| **Learned policy evaluation** | Success/failure/caught rates and average moves logged per bot type (e.g. NO BOT, BOT, BOT LEARNT, ALIEN, ALIEN LEARNT); targets aligned with VI baseline. |

*(Replace placeholders with measured numbers from your runs (e.g. `time()` in `run_simulations.py`, or CI logs) for production-ready documentation.)*

---

## Repository layout (high level)

- **assignment1/** — Grid world, BFS/DFS pathfinding, cell-state encoding, rescue rules.
- **assignment2/** — Bayesian belief over crew/alien positions, multiprocessing runs.
- **assignment3/** — MDP, Value Iteration, PyTorch imitation model, `run_simulations` orchestration, CSV data generation and consumption.
- **finals/** — Consolidated/final run scripts.

---

*This project demonstrates the intersection of robust algorithmic design (graph search, Bayesian inference, MDPs), scalable simulation infrastructure (parallelism, CSV pipelines), and ML integration (imitation learning from optimal policies)—oriented toward clarity, performance, and maintainability.*
