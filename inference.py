import os
import json
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

try:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "hf_dummy_token")
except Exception:
    client = None

TASKS = ["task_1_decongest", "task_2_equity", "task_3_gauntlet"]

SYSTEM_PROMPT = """You are a city policy agent for Verdania. Choose the best action given the current city state.
Respond ONLY with a valid JSON action object.
Available action_types: propose_policy, investigate, pass_turn.
Policy types: expand_transit, build_housing, congestion_tax, green_spaces,
subsidise_rent, zoning_reform, bike_lanes, emissions_tax, income_support, parking_reform.
Districts: north, south, east, west, central.
budget_pct: float between 0.05 and 1.0."""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    if lines and lines[0].strip().lower() == "json":
        lines = lines[1:]
    return "\n".join(lines).strip()


def get_llm_action(observation: dict) -> dict:
    user_msg = (
        f"Current city state: {json.dumps(observation, indent=2)}\n"
        "Choose your action:"
    )

    try:
        if client is None:
            return {"action_type": "pass_turn"}
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        text = (resp.choices[0].message.content or "").strip()
        return json.loads(_strip_code_fences(text))
    except Exception:
        return {"action_type": "pass_turn"}


def run_task(task_id: str) -> None:
    print(f"[START] task={task_id}", flush=True)

    obs = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "seed": 42},
        timeout=30,
    ).json()

    step_num = 0
    done = False

    while not done:
        action = get_llm_action(obs)
        result = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=30).json()
        step_num += 1

        reward = float(result.get("reward", 0.0))
        obs = result.get("observation", {})
        done = bool(obs.get("done", False))

        print(f"[STEP] step={step_num} reward={reward:.4f}", flush=True)

    grader = requests.post(
        f"{ENV_BASE_URL}/grader",
        json={"task_id": task_id},
        timeout=30,
    ).json()
    score = float(grader.get("score", 0.0))

    print(f"[END] task={task_id} score={score:.4f} steps={step_num}", flush=True)


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
