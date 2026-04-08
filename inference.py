import os
import json
import time
import requests as req
from openai import OpenAI

# -- Env vars -- hackathon validator injects these at evaluation time --
API_BASE_URL = os.getenv("API_BASE_URL",
                         "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME",
                       "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")          # no default -- required
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional

# Validator injects API_KEY for LiteLLM proxy.
# Locally / on HF Space we fall back to HF_TOKEN.
API_KEY = os.environ.get("API_KEY") or HF_TOKEN

# Decide whether to use LLM or rule-based fallback
USE_LLM = bool(API_KEY)

# -- Policy simulation environment -----------------------------------
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

# -- OpenAI-compatible client (HF Inference API or LiteLLM proxy) ----
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY or "placeholder",   # placeholder stops OpenAI() crashing
)

# -- Tasks ------------------------------------------------------------
TASKS = ["task_1_decongest", "task_2_equity", "task_3_gauntlet"]

# -- System prompt ----------------------------------------------------
SYSTEM_PROMPT = """You are a city policy agent governing Verdania.
Choose the best action for the current city state.
Respond ONLY with a valid JSON object -- no markdown, no explanation.

Action formats:
  {"action_type": "propose_policy", "policy_type": "<type>", "district": "<district>", "budget_pct": <0.05-1.0>}
  {"action_type": "investigate", "stakeholder": "<stakeholder>"}
  {"action_type": "pass_turn"}

Policy types: expand_transit, build_housing, congestion_tax, green_spaces,
  subsidise_rent, zoning_reform, bike_lanes, emissions_tax, income_support,
  parking_reform
Districts: north, south, east, west, central
Stakeholders: transit_union, property_owners, environmental_coalition,
  chamber_of_commerce

Strategy hints:
  task_1_decongest  -> use expand_transit (north/west) + congestion_tax (central)
  task_2_equity     -> use income_support + build_housing + subsidise_rent
  task_3_gauntlet   -> start with bike_lanes/green_spaces to build political
                      capital, then push bigger reforms
"""

# -- Rule-based fallback actions per task ----------------------------
FALLBACK_ACTIONS = {
    "task_1_decongest": [
        {"action_type": "propose_policy", "policy_type": "expand_transit",
         "district": "north", "budget_pct": 0.25},
        {"action_type": "propose_policy", "policy_type": "expand_transit",
         "district": "west", "budget_pct": 0.25},
        {"action_type": "propose_policy", "policy_type": "congestion_tax",
         "district": "central", "budget_pct": 0.25},
        {"action_type": "propose_policy", "policy_type": "bike_lanes",
         "district": "west", "budget_pct": 0.15},
    ],
    "task_2_equity": [
        {"action_type": "propose_policy", "policy_type": "income_support",
         "district": "north", "budget_pct": 0.25},
        {"action_type": "propose_policy", "policy_type": "income_support",
         "district": "east", "budget_pct": 0.25},
        {"action_type": "propose_policy", "policy_type": "build_housing",
         "district": "east", "budget_pct": 0.25},
        {"action_type": "propose_policy", "policy_type": "subsidise_rent",
         "district": "north", "budget_pct": 0.20},
        {"action_type": "propose_policy", "policy_type": "build_housing",
         "district": "north", "budget_pct": 0.20},
        {"action_type": "propose_policy", "policy_type": "subsidise_rent",
         "district": "east", "budget_pct": 0.15},
    ],
    "task_3_gauntlet": [
        {"action_type": "propose_policy", "policy_type": "bike_lanes",
         "district": "west", "budget_pct": 0.15},
        {"action_type": "propose_policy", "policy_type": "green_spaces",
         "district": "central", "budget_pct": 0.15},
        {"action_type": "propose_policy", "policy_type": "expand_transit",
         "district": "north", "budget_pct": 0.25},
        {"action_type": "propose_policy", "policy_type": "income_support",
         "district": "north", "budget_pct": 0.25},
        {"action_type": "propose_policy", "policy_type": "subsidise_rent",
         "district": "east", "budget_pct": 0.20},
        {"action_type": "propose_policy", "policy_type": "build_housing",
         "district": "east", "budget_pct": 0.20},
        {"action_type": "propose_policy", "policy_type": "emissions_tax",
         "district": "central", "budget_pct": 0.20},
        {"action_type": "propose_policy", "policy_type": "zoning_reform",
         "district": "east", "budget_pct": 0.15},
    ],
}


def wait_for_server(url: str, retries: int = 30, delay: int = 2) -> None:
    """Wait until the environment server is healthy before doing anything."""
    print(f"Waiting for environment server at {url} ...", flush=True)
    for i in range(retries):
        try:
            r = req.get(f"{url}/health", timeout=3)
            if r.status_code == 200:
                print("Environment server is ready.", flush=True)
                return
        except Exception:
            pass
        time.sleep(delay)
    raise RuntimeError(
        f"Environment server at {url} never became healthy "
        f"after {retries * delay} seconds"
    )


def get_llm_action(observation: dict) -> dict:
    """
    Ask the LLM (via HF Inference API or LiteLLM proxy) for the next action.
    Returns a valid action dict. Falls back to pass_turn on any error.
    """
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Current city state:\n"
                    f"{json.dumps(observation, indent=2)}\n\n"
                    f"What is your next action? Reply with JSON only."
                )},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()

        # Strip markdown code fences if model wraps in ```json ... ```
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break

        return json.loads(text.strip())

    except Exception as e:
        print(f"LLM call failed: {e} -- using pass_turn fallback", flush=True)
        return {"action_type": "pass_turn"}


def get_action(observation: dict, task_id: str, step_num: int) -> dict:
    """
    Return the next action.
    If USE_LLM is True  -> ask the LLM via the proxy.
    If USE_LLM is False -> use deterministic rule-based fallback.
    """
    if USE_LLM:
        return get_llm_action(observation)

    # Rule-based fallback -- cycle through the preset action list
    fallback = FALLBACK_ACTIONS.get(task_id, [])
    if fallback:
        index = (step_num - 1) % len(fallback)
        return fallback[index]

    return {"action_type": "pass_turn"}


def run_task(task_id: str) -> None:
    """
    Run one full task episode and emit the required structured stdout logs.

    Output format (validator requires this exactly):
        [START] task=<task_id>
        [STEP] step=<N> reward=<float>
        [END] task=<task_id> score=<float> steps=<N>
    """
    print(f"[START] task={task_id}", flush=True)

    # Reset the environment
    try:
        reset_resp = req.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id, "seed": 42},
            timeout=30,
        )
        obs = reset_resp.json()
    except Exception as e:
        print(f"Reset failed for {task_id}: {e}", flush=True)
        print(f"[STEP] step=1 reward=0.0000", flush=True)
        print(f"[END] task={task_id} score=0.0000 steps=1", flush=True)
        return

    step_num = 0
    done = False
    MAX_STEPS = 10  # safety cap -- tasks have at most 8 turns

    while not done and step_num < MAX_STEPS:
        action = get_action(obs, task_id, step_num + 1)

        try:
            step_resp = req.post(
                f"{ENV_BASE_URL}/step",
                json=action,
                timeout=30,
            )
            result = step_resp.json()
        except Exception as e:
            print(f"Step failed: {e} -- stopping episode", flush=True)
            break

        step_num += 1
        reward = result.get("reward", 0.0)
        obs = result.get("observation", {})
        done = obs.get("done", False)

        print(f"[STEP] step={step_num} reward={reward:.4f}", flush=True)

    # Get the grader score for this episode
    try:
        grader_resp = req.post(
            f"{ENV_BASE_URL}/grader",
            json={"task_id": task_id},
            timeout=30,
        )
        score = grader_resp.json().get("score", 0.0)
    except Exception as e:
        print(f"Grader failed: {e}", flush=True)
        score = 0.0

    print(f"[END] task={task_id} score={score:.4f} steps={step_num}",
          flush=True)


if __name__ == "__main__":
    wait_for_server(ENV_BASE_URL)
    for task in TASKS:
        run_task(task)
