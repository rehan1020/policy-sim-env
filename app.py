"""
app.py
──────
FastAPI server for the Municipal Policy Simulation OpenEnv environment.

OpenEnv spec endpoints:
  POST /reset       – start a new episode
  POST /step        – take an action
  GET  /state       – current episode metadata

Required extra endpoints:
  GET  /tasks       – list tasks + action schema
  POST /grader      – score a city state externally
  POST /baseline    – run the rule-based baseline on all 3 tasks
  GET  /health      – liveness probe
  GET  /            – interactive browser UI
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from environment import PolicyEnvironment
from models import CityObservation, EpisodeState, PolicyAction
from tasks import TASKS, _clamp_score
from core.policy_effects import POLICY_TYPES, POLICY_DESCRIPTIONS
from core.city_model import DISTRICT_NAMES
from core.stakeholders import STAKEHOLDER_NAMES


# ─────────────────────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Municipal Policy Simulation — OpenEnv",
    description=(
        "An OpenEnv RL environment where agents learn to govern a simulated city "
        "by proposing policies, managing budget, and navigating stakeholder politics."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment (single-session; for multi-agent use a session-keyed dict)
_env = PolicyEnvironment(seed=int(os.environ.get("ENV_SEED", "42")))


# ─────────────────────────────────────────────────────────────────────────────
#  Request models
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action_type: str
    policy_type: Optional[str] = None
    district: Optional[str] = None
    budget_pct: Optional[float] = None
    stakeholder: Optional[str] = None
    metadata: Dict[str, Any] = {}


class GraderRequest(BaseModel):
    task_id: str
    # Optionally pass a raw city state to score; if omitted grades current env state
    citywide_kpis: Optional[Dict[str, float]] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Core OpenEnv endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=CityObservation)
def reset(req: ResetRequest = ResetRequest()):
    """Start a new episode. Optionally specify task_id and random seed."""
    try:
        obs = _env.reset(task_id=req.task_id, seed=req.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return obs


@app.post("/step")
def step(req: StepRequest):
    """
    Take one action in the current episode.
    Returns observation, reward, done flag, and info dict.
    """
    action = PolicyAction(
        action_type=req.action_type,
        policy_type=req.policy_type,
        district=req.district,
        budget_pct=req.budget_pct,
        stakeholder=req.stakeholder,
        metadata=req.metadata,
    )
    try:
        obs, reward, done, info = _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state", response_model=EpisodeState)
def state():
    """Return the current episode state (lightweight metadata)."""
    return _env.state()


# ─────────────────────────────────────────────────────────────────────────────
#  Extra required endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions and the full action schema."""
    return {
        "tasks": [
            {
                "task_id":     tid,
                "title":       task.title,
                "difficulty":  task.difficulty,
                "description": task.description,
                "max_turns":   task.max_turns,
                "hint":        task.hint,
            }
            for tid, task in TASKS.items()
        ],
        "action_schema": {
            "action_type": {
                "type": "string",
                "enum": ["propose_policy", "investigate", "pass_turn"],
                "description": (
                    "propose_policy: spend budget on a policy in a district. "
                    "investigate: query a stakeholder's stance (free). "
                    "pass_turn: skip this turn (−0.05 penalty)."
                ),
            },
            "policy_type": {
                "type": "string",
                "nullable": True,
                "enum": POLICY_TYPES,
                "required_for": "propose_policy",
                "descriptions": POLICY_DESCRIPTIONS,
            },
            "district": {
                "type": "string",
                "nullable": True,
                "enum": DISTRICT_NAMES,
                "required_for": "propose_policy",
            },
            "budget_pct": {
                "type": "number",
                "nullable": True,
                "range": [0.05, 1.0],
                "description": "Fraction of annual budget to spend.",
                "required_for": "propose_policy",
            },
            "stakeholder": {
                "type": "string",
                "nullable": True,
                "enum": STAKEHOLDER_NAMES,
                "required_for": "investigate",
            },
        },
        "total_tasks": len(TASKS),
    }


@app.post("/grader")
def grader(req: GraderRequest):
    """
    Score the current city state (or optionally pass explicit KPI values)
    for a given task. Returns score in [0, 1].
    """
    if req.task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {req.task_id}")

    # If explicit KPIs provided, build a city state; otherwise use current env
    if req.citywide_kpis:
        from core.city_model import CityState, DistrictKPIs, DISTRICT_NAMES as DN
        tmp = CityState(max_turns=TASKS[req.task_id].max_turns)
        tmp.political_capital = 70.0
        tmp.budget_left = 0.5
        for name in DN:
            dkpi = DistrictKPIs(**{k: req.citywide_kpis.get(k, 50.0) for k in
                                   ["traffic","housing","equality","air_quality","satisfaction"]})
            tmp.kpis[name] = dkpi
        score = _clamp_score(TASKS[req.task_id].grader(tmp))
    else:
        if _env._city is None:
            raise HTTPException(status_code=400, detail="No active episode — call /reset first.")
        score = _clamp_score(_env.grade_current())

    return {
        "task_id": req.task_id,
        "score": round(score, 4),
        "passed": score >= 0.7,
    }


@app.post("/baseline")
def baseline():
    """
    Run the built-in rule-based baseline agent on all 3 tasks.
    Returns per-task scores and mean score. No API key required.
    """
    results = _run_rule_based_baseline()
    mean = sum(r["score"] for r in results) / len(results)
    return {
        "agent": "rule_based_greedy",
        "results": results,
        "mean_score": round(mean, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Rule-based baseline agent (deterministic, no API key)
# ─────────────────────────────────────────────────────────────────────────────

# Pre-tuned optimal strategies for each task
_BASELINE_STRATEGIES = {
    "task_1_decongest": [
        ("propose_policy", "expand_transit",  "north",   0.25),
        ("propose_policy", "congestion_tax",  "central", 0.20),
        ("propose_policy", "expand_transit",  "west",    0.25),
        ("propose_policy", "bike_lanes",      "central", 0.15),
    ],
    "task_2_equity": [
        ("propose_policy", "income_support",  "north",   0.30),
        ("propose_policy", "build_housing",   "east",    0.25),
        ("propose_policy", "subsidise_rent",  "west",    0.20),
        ("propose_policy", "income_support",  "central", 0.15),
        ("propose_policy", "build_housing",   "north",   0.10),
    ],
    "task_3_gauntlet": [
        ("propose_policy", "bike_lanes",      "central", 0.12),
        ("propose_policy", "green_spaces",    "west",    0.12),
        ("propose_policy", "expand_transit",  "north",   0.15),
        ("propose_policy", "income_support",  "north",   0.15),
        ("propose_policy", "build_housing",   "east",    0.15),
        ("propose_policy", "expand_transit",  "west",    0.12),
        ("propose_policy", "zoning_reform",   "south",   0.12),
    ],
}


def _run_rule_based_baseline():
    results = []
    env = PolicyEnvironment(seed=42)

    for task_id, moves in _BASELINE_STRATEGIES.items():
        env.reset(task_id=task_id, seed=42)
        done = False
        final_info = {}

        for action_type, policy_type, district, budget_pct in moves:
            if done:
                break
            action = PolicyAction(
                action_type=action_type,
                policy_type=policy_type,
                district=district,
                budget_pct=budget_pct,
            )
            _, _, done, final_info = env.step(action)

        # Fill remaining turns
        while not done:
            _, _, done, final_info = env.step(PolicyAction(action_type="pass_turn"))

        score = final_info.get("grader_score", 0.0)
        results.append({
            "task_id": task_id,
            "score": round(score, 4),
            "steps": env.state().turn,
            "final_composite": env._city.composite_score() if env._city else 0,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Health + UI
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "env": "policy-sim-env",
        "version": "1.0.0",
        "tasks": list(TASKS.keys()),
    }


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=_ui_html(), status_code=200)


def _ui_html() -> str:
    tasks_info = [
        {"id": k, "title": v.title, "difficulty": v.difficulty, "max_turns": v.max_turns}
        for k, v in TASKS.items()
    ]
    diff_colors = {"easy": "#34d399", "medium": "#fbbf24", "hard": "#f87171"}

    task_badges = "".join(
        f'<div style="margin:4px 0">'
        f'<span style="background:#0d2018;color:{diff_colors[t["difficulty"]]};padding:2px 8px;'
        f'border-radius:4px;font-size:11px;font-weight:700;margin-right:6px">'
        f'{t["difficulty"].upper()}</span>'
        f'<strong style="font-size:12px">{t["id"]}</strong>'
        f'<span style="color:#64748b;font-size:11px"> — {t["title"]} ({t["max_turns"]} turns)</span>'
        f'</div>'
        for t in tasks_info
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Policy Sim — OpenEnv</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',system-ui,sans-serif;background:#070d1a;color:#e2e8f0;padding:24px}}
    h1{{font-size:20px;color:#34d399;margin-bottom:4px}}
    .sub{{font-size:12px;color:#475569;margin-bottom:20px}}
    .card{{background:#0c1829;border:1px solid #1e3a52;border-radius:8px;padding:16px;margin:12px 0}}
    h2{{font-size:13px;font-weight:700;color:#7dd3fc;margin-bottom:10px;text-transform:uppercase;letter-spacing:.5px}}
    button{{background:#0ea5e9;color:#fff;border:none;border-radius:6px;padding:7px 16px;
            cursor:pointer;font-size:12px;margin:3px;font-weight:600}}
    button:hover{{background:#0284c7}}
    button.green{{background:#059669}} button.green:hover{{background:#047857}}
    select,input{{background:#04090f;color:#e2e8f0;border:1px solid #1e3a52;border-radius:6px;
                  padding:7px;font-size:12px;margin:3px}}
    label{{font-size:11px;color:#64748b;display:block;margin-top:8px;margin-bottom:2px}}
    #out{{background:#04090f;border:1px solid #1e3a52;border-radius:6px;padding:14px;
          font-family:monospace;font-size:11px;white-space:pre-wrap;
          min-height:240px;max-height:500px;overflow-y:auto;color:#a5f3fc}}
    .row{{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-end}}
  </style>
</head>
<body>
  <h1>🌍 Municipal Policy Simulation</h1>
  <p class="sub">OpenEnv-compatible RL environment · City of Verdania · 3 tasks</p>

  <div class="card">
    <h2>Available Tasks</h2>
    {task_badges}
  </div>

  <div class="card">
    <h2>Episode Controls</h2>
    <div class="row">
      <div>
        <label>Task</label>
        <select id="taskSel">
          {"".join(f'<option value="{t["id"]}">{t["title"]}</option>' for t in tasks_info)}
        </select>
      </div>
      <div>
        <label>Seed (optional)</label>
        <input id="seed" type="number" placeholder="42" style="width:80px"/>
      </div>
    </div>
    <div style="margin-top:10px">
      <button onclick="doReset()">🔄 Reset Episode</button>
      <button onclick="doState()">📊 State</button>
      <button onclick="doTasks()">📋 Tasks</button>
      <button onclick="doHealth()">❤️ Health</button>
      <button onclick="doBaseline()" class="green">🤖 Run Baseline</button>
    </div>
  </div>

  <div class="card">
    <h2>Propose Policy</h2>
    <div class="row">
      <div>
        <label>Policy Type</label>
        <select id="polSel">
          {"".join(f'<option value="{p}">{p}</option>' for p in POLICY_TYPES)}
        </select>
      </div>
      <div>
        <label>District</label>
        <select id="distSel">
          {"".join(f'<option value="{d}">{d}</option>' for d in DISTRICT_NAMES)}
        </select>
      </div>
      <div>
        <label>Budget %</label>
        <input id="budg" type="number" value="0.25" step="0.05" min="0.05" max="1.0" style="width:80px"/>
      </div>
    </div>
    <div style="margin-top:10px">
      <button onclick="doPropose()" class="green">🏛️ Propose</button>
      <button onclick="doPass()">⏭️ Pass Turn</button>
    </div>
  </div>

  <div class="card">
    <h2>Investigate Stakeholder</h2>
    <div class="row">
      <div>
        <label>Stakeholder</label>
        <select id="shSel">
          {"".join(f'<option value="{s}">{s}</option>' for s in STAKEHOLDER_NAMES)}
        </select>
      </div>
      <button onclick="doInvestigate()" style="margin-top:18px">🔍 Investigate</button>
    </div>
  </div>

  <div class="card">
    <h2>Output</h2>
    <div id="out">← Press Reset Episode to start.</div>
  </div>

  <script>
    const out = document.getElementById('out');
    const show = o => {{ out.textContent = JSON.stringify(o, null, 2); }};
    const post = async (url, body={{}}) => (await fetch(url,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}})).json();
    const get  = async url => (await fetch(url)).json();

    async function doReset() {{
      const tid = document.getElementById('taskSel').value;
      const s = document.getElementById('seed').value;
      show(await post('/reset', {{task_id: tid, seed: s ? parseInt(s) : null}}));
    }}
    async function doState()  {{ show(await get('/state'));  }}
    async function doTasks()  {{ show(await get('/tasks'));  }}
    async function doHealth() {{ show(await get('/health')); }}
    async function doBaseline() {{ out.textContent = '⏳ Running baseline…'; show(await post('/baseline')); }}

    async function doPropose() {{
      show(await post('/step', {{
        action_type: 'propose_policy',
        policy_type: document.getElementById('polSel').value,
        district:    document.getElementById('distSel').value,
        budget_pct:  parseFloat(document.getElementById('budg').value),
      }}));
    }}
    async function doPass() {{ show(await post('/step', {{action_type: 'pass_turn'}})); }}
    async function doInvestigate() {{
      show(await post('/step', {{action_type: 'investigate', stakeholder: document.getElementById('shSel').value}}));
    }}
  </script>
</body>
</html>"""
