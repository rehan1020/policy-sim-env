#!/usr/bin/env python3
"""
inference.py  ←  required by OpenEnv validator at repo root
Alias for baseline.py — runs the baseline agent against all 3 tasks.

Usage:
    python inference.py
    OPENAI_API_KEY=sk-... python inference.py
    ENV_BASE_URL=https://your-space.hf.space python inference.py
"""
import os
import sys
import json
import requests

ENV_BASE_URL  = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Baseline strategies (rule-based, no API key needed) ───────────────────────

STRATEGIES = {
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


def run(base_url: str) -> list:
    results = []

    for task_id, moves in STRATEGIES.items():
        # Reset
        r = requests.post(f"{base_url}/reset",
                          json={"task_id": task_id, "seed": 42}, timeout=30)
        r.raise_for_status()

        done = False
        final_info = {}

        # Execute strategy
        for policy_type, district, budget_pct in moves:
            if done:
                break
            r = requests.post(f"{base_url}/step", json={
                "action_type": "propose_policy",
                "policy_type": policy_type,
                "district":    district,
                "budget_pct":  budget_pct,
            }, timeout=30)
            r.raise_for_status()
            result = r.json()
            done = result.get("done", False)
            final_info = result.get("info", {})

        # Fill remaining turns
        while not done:
            r = requests.post(f"{base_url}/step",
                              json={"action_type": "pass_turn"}, timeout=30)
            r.raise_for_status()
            result = r.json()
            done = result.get("done", False)
            final_info = result.get("info", {})

        score = final_info.get("grader_score", 0.0)
        results.append({"task_id": task_id, "score": round(score, 4)})
        print(f"  {task_id}: {score:.4f}")

    return results


if __name__ == "__main__":
    print(f"\n{'='*50}")
    print("  OpenEnv Inference Script — Policy Sim")
    print(f"  Environment: {ENV_BASE_URL}")
    print(f"{'='*50}\n")

    results = run(ENV_BASE_URL)

    mean = sum(r["score"] for r in results) / len(results)
    print(f"\n  Mean score: {mean:.4f}")
    print(json.dumps({"results": results, "mean_score": mean}, indent=2))
