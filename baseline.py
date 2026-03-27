#!/usr/bin/env python3
"""
baseline.py
───────────
Baseline inference script for the Municipal Policy Simulation environment.

Two modes:
  1. Rule-based greedy (no API key) — submits pre-tuned optimal strategies
  2. LLM agent (requires OPENAI_API_KEY) — GPT-4o reasons about each turn

Usage:
  python baseline.py                          # rule-based
  OPENAI_API_KEY=sk-... python baseline.py    # LLM agent
  ENV_BASE_URL=https://... python baseline.py # remote environment
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import requests

ENV_BASE_URL  = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL         = os.environ.get("BASELINE_MODEL", "gpt-4o")
MAX_LLM_STEPS = 12


# ─────────────────────────────────────────────────────────────────────────────
#  Simple HTTP client
# ─────────────────────────────────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self, task_id: str, seed: int = 42) -> Dict:
        r = self.session.post(f"{self.base}/reset",
                              json={"task_id": task_id, "seed": seed}, timeout=30)
        r.raise_for_status(); return r.json()

    def step(self, **kwargs) -> Dict:
        r = self.session.post(f"{self.base}/step", json=kwargs, timeout=30)
        r.raise_for_status(); return r.json()

    def state(self) -> Dict:
        r = self.session.get(f"{self.base}/state", timeout=10)
        r.raise_for_status(); return r.json()

    def tasks(self) -> Dict:
        r = self.session.get(f"{self.base}/tasks", timeout=10)
        r.raise_for_status(); return r.json()

    def health(self) -> Dict:
        r = self.session.get(f"{self.base}/health", timeout=10)
        r.raise_for_status(); return r.json()


# ─────────────────────────────────────────────────────────────────────────────
#  LLM helpers
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert city policy advisor for the city of Verdania.
Each turn you must choose ONE action from:
  1. propose_policy — spend budget on a policy in a district
  2. investigate    — learn a stakeholder's stance (free, no budget cost)
  3. pass_turn      — skip (use sparingly, costs -0.05)

Available policy types: expand_transit, build_housing, congestion_tax,
  green_spaces, subsidise_rent, zoning_reform, bike_lanes, emissions_tax,
  income_support, parking_reform

Available districts: north, south, east, west, central

Rules:
  - budget_pct must be between 0.05 and your remaining budget
  - Political capital reaching 0 ends the episode (you are recalled)
  - Good policies for the right districts have amplified effects
  - Use 'investigate' before controversial policies to check stakeholder stances

Respond with ONLY a JSON object, one of:
  {"action": "propose_policy", "policy_type": "...", "district": "...", "budget_pct": 0.25}
  {"action": "investigate", "stakeholder": "transit_union|property_owners|env_coalition|business_council"}
  {"action": "pass_turn"}
""").strip()


def call_llm(messages: List[Dict]) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 512,
    }
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(**{k: v for k, v in payload.items()})
        return resp.choices[0].message.content.strip()
    except ImportError:
        r = requests.post("https://api.openai.com/v1/chat/completions",
                          headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def parse_llm_action(text: str) -> Dict:
    """Extract JSON action from LLM response."""
    import re
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        # Try to find JSON object in the text
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {"action": "pass_turn"}


def _obs_summary(obs: Dict) -> str:
    """Compact observation summary for LLM prompt."""
    kpis = obs.get("citywide_kpis", {})
    return textwrap.dedent(f"""
        Turn {obs.get('turn',0)}/{obs.get('turn',0)+obs.get('turns_left',0)}
        Budget remaining: {obs.get('budget_left',0):.0%}
        Political capital: {obs.get('political_capital',0):.0f}/100
        Composite score: {obs.get('composite_score',0):.1f}
        KPIs (citywide): traffic={kpis.get('traffic',0):.1f} housing={kpis.get('housing',0):.1f}
              equality={kpis.get('equality',0):.1f} air_quality={kpis.get('air_quality',0):.1f}
              satisfaction={kpis.get('satisfaction',0):.1f}
        Last reward: {obs.get('reward',0):.3f}
    """).strip()


# ─────────────────────────────────────────────────────────────────────────────
#  LLM episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_llm_episode(env: EnvClient, task_id: str) -> Dict:
    obs = env.reset(task_id, seed=42)
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Task: {obs.get('task_description', task_id)}\n\n"
            f"Starting state:\n{_obs_summary(obs)}\n\n"
            "What is your first action?"
        )},
    ]

    final_score = 0.0
    done = False

    for step_i in range(MAX_LLM_STEPS):
        try:
            llm_text = call_llm(conversation)
        except Exception as exc:
            print(f"    LLM error: {exc}")
            break

        conversation.append({"role": "assistant", "content": llm_text})
        decision = parse_llm_action(llm_text)
        action_type = decision.get("action", "pass_turn")

        # Execute
        step_kwargs = {"action_type": action_type}
        if action_type == "propose_policy":
            step_kwargs.update({
                "policy_type": decision.get("policy_type"),
                "district":    decision.get("district"),
                "budget_pct":  decision.get("budget_pct", 0.2),
            })
        elif action_type == "investigate":
            step_kwargs["stakeholder"] = decision.get("stakeholder")

        result = env.step(**step_kwargs)
        obs    = result.get("observation", result)
        done   = result.get("done", obs.get("done", False))
        info   = result.get("info", {})

        if done:
            final_score = info.get("grader_score", 0.0)
            break

        # Feedback for next turn
        feedback_parts = [f"Action: {action_type}"]
        if obs.get("error_message"):
            feedback_parts.append(f"ERROR: {obs['error_message']}")
        if obs.get("kpi_delta"):
            delta_str = ", ".join(f"{k}:{v:+.1f}" for k, v in obs["kpi_delta"].items())
            feedback_parts.append(f"KPI changes: {delta_str}")
        if obs.get("stakeholder_reactions"):
            rxn = obs["stakeholder_reactions"]
            feedback_parts.append(f"Stakeholder reactions: {rxn}")
        if obs.get("capital_delta") is not None:
            feedback_parts.append(f"Capital delta: {obs['capital_delta']:+.1f}")
        if obs.get("investigation_result"):
            feedback_parts.append(f"Investigation: {json.dumps(obs['investigation_result'], indent=2)}")
        if obs.get("warning"):
            feedback_parts.append(f"Warning: {obs['warning']}")

        feedback = "\n".join(feedback_parts)
        conversation.append({
            "role": "user",
            "content": f"{feedback}\n\nCurrent state:\n{_obs_summary(obs)}\n\nNext action?",
        })

    if not done:
        # Force end
        while not done:
            result = env.step(action_type="pass_turn")
            done = result.get("done", False)
            final_score = result.get("info", {}).get("grader_score", final_score)

    state = env.state()
    return {
        "task_id":        task_id,
        "score":          round(final_score, 4),
        "steps":          state.get("turn", 0),
        "final_composite": state.get("composite_score", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Rule-based baseline
# ─────────────────────────────────────────────────────────────────────────────

_STRATEGIES = {
    "task_1_decongest": [
        ("expand_transit",  "north",   0.25),
        ("congestion_tax",  "central", 0.20),
        ("expand_transit",  "west",    0.25),
        ("bike_lanes",      "central", 0.15),
    ],
    "task_2_equity": [
        ("income_support",  "north",   0.30),
        ("build_housing",   "east",    0.25),
        ("subsidise_rent",  "west",    0.20),
        ("income_support",  "central", 0.15),
        ("build_housing",   "north",   0.10),
    ],
    "task_3_gauntlet": [
        ("bike_lanes",      "central", 0.12),
        ("green_spaces",    "west",    0.12),
        ("expand_transit",  "north",   0.15),
        ("income_support",  "north",   0.15),
        ("build_housing",   "east",    0.15),
        ("expand_transit",  "west",    0.12),
        ("zoning_reform",   "south",   0.12),
    ],
}


def run_rule_based(env: EnvClient) -> List[Dict]:
    results = []
    for task_id, moves in _STRATEGIES.items():
        env.reset(task_id, seed=42)
        done = False
        final_info: Dict = {}

        for policy_type, district, budget_pct in moves:
            if done: break
            result = env.step(
                action_type="propose_policy",
                policy_type=policy_type,
                district=district,
                budget_pct=budget_pct,
            )
            done = result.get("done", False)
            final_info = result.get("info", {})

        while not done:
            result = env.step(action_type="pass_turn")
            done = result.get("done", False)
            final_info = result.get("info", {})

        state = env.state()
        results.append({
            "task_id":        task_id,
            "score":          round(final_info.get("grader_score", 0.0), 4),
            "steps":          state.get("turn", 0),
            "final_composite": state.get("composite_score", 0),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  Municipal Policy Simulation — Baseline Agent")
    print(f"{'='*60}")
    print(f"  Environment: {ENV_BASE_URL}")
    use_llm = bool(OPENAI_API_KEY)
    print(f"  Mode: {'LLM agent (' + MODEL + ')' if use_llm else 'Rule-based (no API key)'}")
    print(f"{'='*60}\n")

    env = EnvClient(ENV_BASE_URL)

    # Health check
    try:
        h = env.health()
        print(f"✅ Environment up — {h.get('env')} v{h.get('version')}\n")
    except Exception as exc:
        print(f"❌ Cannot reach {ENV_BASE_URL}: {exc}")
        sys.exit(1)

    # Get task list
    tasks_info = env.tasks()
    task_ids = [t["task_id"] for t in tasks_info["tasks"]]
    print(f"Tasks: {task_ids}\n")

    results: List[Dict] = []

    if use_llm:
        for task_id in task_ids:
            print(f"  Running LLM on {task_id}...")
            r = run_llm_episode(env, task_id)
            results.append(r)
            print(f"  → score={r['score']:.3f}  steps={r['steps']}  composite={r['final_composite']:.1f}\n")
    else:
        print("Running rule-based baseline...\n")
        results = run_rule_based(env)
        for r in results:
            print(f"  {r['task_id']:<30}  score={r['score']:.3f}  composite={r['final_composite']:.1f}")

    # Summary table
    print(f"\n{'='*60}")
    print("  BASELINE RESULTS")
    print(f"{'='*60}")
    for r in results:
        bar = "█" * int(r["score"] * 25)
        print(f"  {r['task_id']:<30} {bar:<25} {r['score']:.3f}")
    mean = sum(r["score"] for r in results) / len(results)
    print(f"\n  Mean score: {mean:.3f}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    main()
