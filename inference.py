import os
import json
import time
import requests as req
from openai import OpenAI

# -- Environment variables (hackathon-injected) ------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Validator injects API_KEY for the LiteLLM proxy; fall back to HF_TOKEN locally
API_KEY = os.environ.get("API_KEY") or HF_TOKEN

# Policy simulation environment URL (same container)
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

# -- OpenAI client (routes ALL LLM calls through the hackathon's LiteLLM proxy)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# -- Tasks to run ------------------------------------------------------------
TASKS = ["task_1_decongest", "task_2_equity", "task_3_gauntlet"]

# -- System prompt -----------------------------------------------------------
SYSTEM_PROMPT = """You are a city policy agent governing Verdania.
Choose the best action for the current city state.
Respond ONLY with a valid JSON object — no markdown, no explanation.

Action formats:
  {"action_type": "propose_policy", "policy_type": "<type>", "district": "<district>", "budget_pct": <0.05-1.0>}
  {"action_type": "investigate", "stakeholder": "<stakeholder>"}
  {"action_type": "pass_turn"}

Policy types: expand_transit, build_housing, congestion_tax, green_spaces,
  subsidise_rent, zoning_reform, bike_lanes, emissions_tax, income_support, parking_reform
Districts: north, south, east, west, central
Stakeholders: transit_union, property_owners, environmental_coalition, chamber_of_commerce
"""


def wait_for_server(url: str, retries: int = 30, delay: int = 2) -> None:
    """Wait until the environment server is healthy."""
    for _ in range(retries):
        try:
            r = req.get(f"{url}/health", timeout=3)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(delay)
    raise RuntimeError(f"Environment server at {url} never became healthy after {retries} retries")


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if "```" not in text:
        return text
    parts = text.split("```")
    if len(parts) >= 2:
        text = parts[1]
    if text.startswith("json"):
        text = text[4:]
    return text.strip()


def get_action(observation: dict) -> dict:
    """Ask the LLM (via LiteLLM proxy) for the next action."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Current city state:\n{json.dumps(observation, indent=2)}\n\n"
                    "What is your next action?"
                )},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception:
        return {"action_type": "pass_turn"}


def run_task(task_id: str) -> None:
    """Run one full task episode and emit structured stdout logs."""
    print(f"[START] task={task_id}", flush=True)

    obs = req.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "seed": 42},
        timeout=30,
    ).json()

    step_num = 0
    done = False

    while not done:
        action = get_action(obs)
        result = req.post(
            f"{ENV_BASE_URL}/step",
            json=action,
            timeout=30,
        ).json()
        step_num += 1
        reward = result.get("reward", 0.0)
        obs = result.get("observation", {})
        done = obs.get("done", False)
        print(f"[STEP] step={step_num} reward={reward:.4f}", flush=True)

        # Safety: cap at 10 steps to avoid infinite loops
        if step_num >= 10:
            break

    grader = req.post(
        f"{ENV_BASE_URL}/grader",
        json={"task_id": task_id},
        timeout=30,
    ).json()
    score = grader.get("score", 0.0)
    print(f"[END] task={task_id} score={score:.4f} steps={step_num}", flush=True)


if __name__ == "__main__":
    wait_for_server(ENV_BASE_URL)
    for task in TASKS:
        run_task(task)
