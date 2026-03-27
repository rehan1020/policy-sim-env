# 🌍 Municipal Policy Simulation — OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces/openenv/policy-sim-env)
[![Tests](https://img.shields.io/badge/tests-139%20passing-brightgreen)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-green)]()

An **OpenEnv-compatible reinforcement learning environment** where AI agents
learn to govern the simulated city of **Verdania** — proposing policy
interventions, managing a finite budget, and navigating four powerful
stakeholder lobbies across an 8-turn mayoral term.

**This environment is novel**: no public RL environment exists that models
civic decision-making with multi-district, multi-stakeholder trade-offs and
long-horizon KPI planning.

---

## 🎯 Why This Environment?

Policy-making is a genuine multi-objective optimisation problem that humans
do badly and that frontier LLMs do even worse. Every intervention:

- Helps some KPIs and hurts others (no free lunch)
- Angers some stakeholders and pleases others (political constraints)
- Has amplified or dampened effects depending on which district it targets
- Propagates across a causal graph with ±20% noise

Training an agent to govern well requires learning: **when to use pricing
vs infrastructure, which districts to prioritise, how to sequence reforms to
build political capital before pushing controversial policies.**

---

## 🏙️ The City of Verdania

Five districts with distinct demographics and policy sensitivities:

| District | Density | Income | Transit Multiplier |
|----------|---------|--------|--------------------|
| **North** | High | Low | 1.4× — high-density, low-income gains most from transit |
| **South** | Medium | High | 0.7× — wealthy, car-dependent |
| **East** | Low | Medium | 1.3× zoning — lots of undeveloped land |
| **West** | High | Medium | 1.3× bikes — progressive commuter culture |
| **Central** | Very High | Mixed | 1.5× parking, 1.4× congestion tax |

Five KPIs tracked citywide (0–100 scale, all averaged across districts):

| KPI | Direction | Weight |
|-----|-----------|--------|
| Traffic Congestion | Lower = better | 25% |
| Housing Affordability | Higher = better | 25% |
| Income Equality | Higher = better | 20% |
| Air Quality Index | Higher = better | 15% |
| Public Satisfaction | Higher = better | 15% |

---

## 🎮 Action Space

`POST /step` with JSON:

| Field | Type | Required for | Description |
|-------|------|-------------|-------------|
| `action_type` | string | always | `propose_policy` · `investigate` · `pass_turn` |
| `policy_type` | string | propose_policy | One of 10 policy types |
| `district` | string | propose_policy | One of 5 districts |
| `budget_pct` | float | propose_policy | 0.05–1.0 (fraction of annual budget) |
| `stakeholder` | string | investigate | One of 4 stakeholder groups |

### Policy Types

| Policy | Main Effect | Trade-off |
|--------|-------------|-----------|
| `expand_transit` | −12 traffic | +3 housing, +5 equality |
| `build_housing` | +15 housing | −3 air quality |
| `congestion_tax` | −18 traffic | −8 satisfaction |
| `green_spaces` | +12 air quality | −2 housing |
| `subsidise_rent` | +10 housing, +12 equality | +1 traffic |
| `zoning_reform` | +8 housing | −2 air quality |
| `bike_lanes` | −5 traffic, +7 air quality | mild |
| `emissions_tax` | +15 air quality | −5 satisfaction |
| `income_support` | +18 equality | none |
| `parking_reform` | −8 traffic | −3 satisfaction |

*All effects scaled by `budget_pct × district_multiplier × N(1.0, 0.2) noise`*

---

## 📡 Observation Space

Every `reset()` / `step()` returns a `CityObservation` with:

| Field | Description |
|-------|-------------|
| `citywide_kpis` | Average KPIs across all districts |
| `district_kpis` | Per-district breakdown |
| `composite_score` | Weighted wellbeing score (0–100) |
| `budget_left` | Remaining annual budget fraction |
| `political_capital` | 0–100; recall (episode end) at 0 |
| `stakeholder_approvals` | Per-lobby approval ratings |
| `turn` / `turns_left` | Episode progress |
| `kpi_delta` | KPI changes from last policy |
| `stakeholder_reactions` | Stance per stakeholder for last action |
| `capital_delta` | Political capital change last turn |
| `investigation_result` | Stakeholder intel (from investigate) |
| `reward` / `cumulative_reward` | Step + running reward |
| `done` / `done_reason` | Episode termination |

---

## 🤝 Stakeholders

Four lobby groups react to every policy:

| Stakeholder | Loves | Hates | Power |
|-------------|-------|-------|-------|
| Transit Workers Union | expand_transit, bike_lanes | congestion_tax, parking_reform | 1.0× |
| Property Owners Assoc. | zoning_reform, parking_reform | build_housing, subsidise_rent | **1.5×** |
| Environmental Coalition | emissions_tax, green_spaces | build_housing | 0.8× |
| Chamber of Commerce | congestion_tax, parking_reform | emissions_tax, income_support | 1.2× |

- Each **supportive** reaction: +5 × power\_level capital
- Each **opposed** reaction: −8 × power\_level capital
- If ≥2 oppose simultaneously: additional −5 coalition penalty
- **Political capital reaches 0 → agent recalled (episode ends, −0.30 reward)**

Use `investigate` (free, no budget cost) to check a stakeholder's stance before proposing.

---

## 🎮 Tasks

### Task 1 — Decongest the City 🟢 Easy

**Scenario:** Traffic crisis — north (88), central (90), west (82) are gridlocked.  
**Goal:** Reduce citywide traffic by ≥4 pts within **4 turns**, using ≤65% budget.

**Grader:**
| Reduction | Score |
|-----------|-------|
| ≥4 pts, budget ≤65% | **1.0** |
| ≥4 pts, over budget | **0.8** |
| ≥2 pts | **0.5** |
| ≥0.8 pts | **0.2** |
| < 0.8 pts | **0.0** |

**Optimal strategy:** `expand_transit` in north/west + `congestion_tax` in central.

---

### Task 2 — Equity Reform 🟡 Medium

**Scenario:** Low housing (30.8), equality (26.2), satisfaction (38.8). Traffic fine.  
**Goal:** Raise all three KPIs by ≥5 pts citywide average within **6 turns**.

**Grader:** Proportional — mean ratio of achieved gain vs target across 3 KPIs. Partial credit for partial improvement. +0.1 bonus if all three hit target and >20% budget left.

**Optimal strategy:** `income_support` → `build_housing` → `subsidise_rent` across multiple districts.

---

### Task 3 — Political Gauntlet 🔴 Hard

**Scenario:** All KPIs at crisis levels. Political capital at **50** (fragile). Stakeholders adversarial.  
**Goal:** Composite score +30 pts across **8 turns** without being recalled.

**Grader:**
```
0.50 × (composite gain / 30)    KPI improvement
0.30 × survived full term        Political survival
0.20 × (final capital / 100)    Political health
```

**Optimal strategy:** Build easy wins first (bike\_lanes, green\_spaces) to recover capital, then push bigger reforms.

---

## 💰 Reward Function

```
step_reward = -0.02                        # base step cost (efficiency pressure)
            + {-0.05 if pass_turn}         # extra penalty for doing nothing
            + Σ(KPI_weight × ΔKPI) / 100  # shaped KPI signal each turn
            - 0.03 × n_stakeholders_opposed
            + {-0.30 terminal if recalled}
```

Terminal grader score (0–1) is added to reward at episode end.  
**Key property:** the agent gets a learning signal every single turn, not just at the end.

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start new episode (`{"task_id": "...", "seed": 42}`) |
| `POST` | `/step` | Take action |
| `GET` | `/state` | Episode metadata |
| `GET` | `/tasks` | All tasks + full action schema |
| `POST` | `/grader` | Score current state for a task |
| `POST` | `/baseline` | Run rule-based baseline on all 3 tasks |
| `GET` | `/health` | Liveness check |
| `GET` | `/` | Interactive browser UI |

---

## 📊 Baseline Scores

### Rule-based greedy agent

| Task | Score | Strategy |
|------|-------|----------|
| task_1_decongest | **0.50** | expand_transit × 2 + congestion_tax + bike_lanes |
| task_2_equity | **0.48** | income_support × 2 + build_housing × 2 + subsidise_rent |
| task_3_gauntlet | **0.42** | sequenced easy-wins then structural reforms |
| **Mean** | **0.47** | |

### Do-nothing baseline (pass_turn every step)
| Task | Score |
|------|-------|
| All tasks | **0.00–0.15** |

The gap between 0.0 and 0.47 creates strong learning signal. Frontier LLMs
(GPT-4o zero-shot) typically score 0.3–0.5, leaving meaningful room to improve.

---

## 🚀 Setup & Usage

### Option A — Docker (recommended)

```bash
git clone https://huggingface.co/spaces/openenv/policy-sim-env
cd policy-sim-env

docker build -t policy-sim-env .
docker run -p 7860:7860 policy-sim-env
# Open http://localhost:7860
```

### Option B — Local Python

```bash
pip install fastapi uvicorn pydantic requests pyyaml openai
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### Running Tests

```bash
python -m unittest tests/test_core.py tests/test_environment.py tests/test_api.py -v
# Expected: 139 tests, 0 failures
```

### Running the Baseline

```bash
# Rule-based (no key needed)
python baseline.py

# LLM agent
export OPENAI_API_KEY=sk-...
export ENV_BASE_URL=http://localhost:7860
python baseline.py
```

---

## 💻 Agent Example

```python
import requests

BASE = "http://localhost:7860"

# Start episode
obs = requests.post(f"{BASE}/reset", json={"task_id": "task_1_decongest", "seed": 42}).json()
print(f"Starting composite score: {obs['composite_score']}")
print(f"Traffic (citywide): {obs['citywide_kpis']['traffic']}")

# Investigate a stakeholder before acting
result = requests.post(f"{BASE}/step", json={
    "action_type": "investigate",
    "stakeholder": "transit_union"
}).json()
print(result["observation"]["investigation_result"]["tip"])

# Propose a policy
result = requests.post(f"{BASE}/step", json={
    "action_type": "propose_policy",
    "policy_type": "expand_transit",
    "district": "north",
    "budget_pct": 0.25
}).json()
obs = result["observation"]
print(f"Traffic delta: {obs['kpi_delta']['traffic']:+.2f}")
print(f"Stakeholder reactions: {obs['stakeholder_reactions']}")
print(f"Reward: {result['reward']:.3f}")

# Check state
state = requests.get(f"{BASE}/state").json()
print(f"Turn {state['turn']}/{state['max_turns']}, budget left: {state['budget_left']:.0%}")
```

---

## 📁 Project Structure

```
policy-sim-env/
├── core/
│   ├── city_model.py        # CityState, districts, KPI classes
│   ├── policy_effects.py    # Causal impact matrix + apply_policy()
│   └── stakeholders.py      # 4 lobby groups + StakeholderEngine
├── models.py                # Pydantic: PolicyAction, CityObservation, EpisodeState
├── environment.py           # PolicyEnvironment: reset/step/state
├── tasks.py                 # 3 tasks + delta-based graders
├── app.py                   # FastAPI server (all endpoints + browser UI)
├── baseline.py              # Rule-based + LLM baseline agent
├── tests/
│   ├── test_core.py         # 49 tests: city model, policy effects, stakeholders
│   ├── test_environment.py  # 53 tests: environment loop, graders, scenarios
│   └── test_api.py          # 37 tests: endpoint contracts, baseline, serialisation
├── openenv.yaml             # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile               # HF Spaces compatible
└── README.md
```

---

## 🔬 Design Decisions

**Why SQLite-free?** The entire simulation is pure Python math — deterministic
given a seed, instant to reset, no I/O overhead. This makes graders fully
reproducible and allows thousands of episodes per second for RL training.

**Why ±20% noise?** Forces agents to learn robust policies rather than
memorising exact effect chains. The optimal policy under noise is different
from the optimal policy under certainty.

**Why political capital?** Creates a secondary survival constraint that
requires sequencing — build trust before pushing hard reforms. This is the
key mechanic that makes the hard task genuinely hard for LLMs.

**Why district multipliers?** Gives the agent a spatial reasoning sub-problem.
Targeting north for transit (1.4× effect) vs south (0.7× effect) is a learnable
skill that separates good agents from mediocre ones.

---

## 📄 License

Apache 2.0
