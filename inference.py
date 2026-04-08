import os
import json
import time
import requests as req
from openai import OpenAI

DEPRECATED_HF_HOST = "api-inference.huggingface.co"
ROUTER_HF_HOST = "router.huggingface.co"

# Mandatory environment configuration
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# LiteLLM key can be injected as API_KEY by validator; HF_TOKEN remains mandatory in docs.
API_KEY = os.getenv("API_KEY") or HF_TOKEN
USE_LLM = bool(API_KEY)

# Environment runtime configuration
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = os.getenv("OPENENV_BENCHMARK", "policy-sim-env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))


def _normalize_api_base_url(url: str) -> str:
    if DEPRECATED_HF_HOST in url:
        return url.replace(DEPRECATED_HF_HOST, ROUTER_HF_HOST)
    return url

# OpenAI-compatible client for all LLM calls
API_BASE_URL = _normalize_api_base_url(API_BASE_URL)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY or "placeholder",
)

TASKS = ["task_1_decongest", "task_2_equity", "task_3_gauntlet"]

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

# Deterministic fallback actions for no-key local testing
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


def wait_for_server(url: str, retries: int = 30, delay: int = 2) -> bool:
    for _ in range(retries):
        try:
            r = req.get(f"{url}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False


def _normalize_action(action: dict) -> dict:
    if not isinstance(action, dict):
        return {"action_type": "pass_turn"}

    action_type = action.get("action_type", "pass_turn")
    if action_type not in {"propose_policy", "investigate", "pass_turn"}:
        return {"action_type": "pass_turn"}

    if action_type == "propose_policy":
        policy_type = action.get("policy_type")
        district = action.get("district")
        budget_pct = action.get("budget_pct", 0.2)
        try:
            budget_pct = float(budget_pct)
        except Exception:
            budget_pct = 0.2
        budget_pct = max(0.05, min(1.0, budget_pct))
        if not policy_type or not district:
            return {"action_type": "pass_turn"}
        return {
            "action_type": "propose_policy",
            "policy_type": policy_type,
            "district": district,
            "budget_pct": budget_pct,
        }

    if action_type == "investigate":
        stakeholder = action.get("stakeholder")
        if not stakeholder:
            return {"action_type": "pass_turn"}
        return {"action_type": "investigate", "stakeholder": stakeholder}

    return {"action_type": "pass_turn"}


def _llm_response_text(active_client: OpenAI, observation: dict) -> str:
    resp = active_client.chat.completions.create(
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
    return (resp.choices[0].message.content or "").strip()


def get_llm_action(observation: dict) -> dict:
    try:
        text = _llm_response_text(client, observation)

    except Exception as exc:
        message = str(exc)
        should_retry_with_router = (
            DEPRECATED_HF_HOST in API_BASE_URL
            and ("Error code: 410" in message or "no longer supported" in message)
        )
        if not should_retry_with_router:
            return {"action_type": "pass_turn"}

        retry_base_url = API_BASE_URL.replace(DEPRECATED_HF_HOST, ROUTER_HF_HOST)
        retry_client = OpenAI(base_url=retry_base_url, api_key=API_KEY or "placeholder")
        try:
            text = _llm_response_text(retry_client, observation)
        except Exception:
            return {"action_type": "pass_turn"}

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    try:
        return _normalize_action(json.loads(text.strip()))
    except Exception:
        return {"action_type": "pass_turn"}


def get_action(observation: dict, task_id: str, step_num: int) -> dict:
    if USE_LLM:
        return get_llm_action(observation)

    fallback = FALLBACK_ACTIONS.get(task_id, [])
    if fallback:
        index = (step_num - 1) % len(fallback)
        return _normalize_action(fallback[index])

    return {"action_type": "pass_turn"}


def _format_action(action: dict) -> str:
    return json.dumps(action, separators=(",", ":"), ensure_ascii=True)


def _clamp_unit(value: float) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    # Keep native unit values unchanged; otherwise normalize shaped rewards
    # (typically around [-1, 1]) into [0, 1] for validator-facing logs.
    if 0.0 <= value <= 1.0:
        return value
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


def _log_start(task_id: str) -> None:
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def _log_step(step_num: int, action_str: str, reward: float, done: bool, error: str | None) -> None:
    done_val = str(bool(done)).lower()
    error_val = error if error else "null"
    reward_val = _clamp_unit(reward)
    print(
        f"[STEP] step={step_num} action={action_str} reward={reward_val:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{_clamp_unit(r):.2f}" for r in rewards)
    print(
        f"[END] success={str(bool(success)).lower()} steps={steps} score={_clamp_unit(score):.3f} rewards={rewards_str}",
        flush=True,
    )


def run_task(task_id: str) -> None:
    _log_start(task_id)

    step_num = 0
    done = False
    score = 0.0
    rewards: list[float] = []
    success = False
    last_error: str | None = None
    obs: dict = {}

    try:
        try:
            reset_resp = req.post(
                f"{ENV_BASE_URL}/reset",
                json={"task_id": task_id, "seed": 42},
                timeout=30,
            )
            reset_resp.raise_for_status()
            obs = reset_resp.json()
        except Exception as exc:
            last_error = str(exc)
            done = True

        while not done and step_num < MAX_STEPS:
            action = get_action(obs, task_id, step_num + 1)
            action_str = _format_action(action)
            reward = 0.0

            try:
                step_resp = req.post(
                    f"{ENV_BASE_URL}/step",
                    json=action,
                    timeout=30,
                )
                step_resp.raise_for_status()
                result = step_resp.json()
                reward = float(result.get("reward", 0.0) or 0.0)
                obs = result.get("observation", {})
                done = bool(result.get("done", obs.get("done", False)))
                err_field = obs.get("error_message")
                last_error = err_field if err_field else None
            except Exception as exc:
                done = True
                last_error = str(exc)

            step_num += 1
            rewards.append(reward)
            _log_step(step_num, action_str, reward, done, last_error)

        try:
            grader_resp = req.post(
                f"{ENV_BASE_URL}/grader",
                json={"task_id": task_id},
                timeout=30,
            )
            grader_resp.raise_for_status()
            score = float(grader_resp.json().get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0

        score = min(max(score, 0.0), 1.0)
        success = last_error is None and 0.0 <= score <= 1.0

    finally:
        _log_end(success, step_num, score, rewards)

if __name__ == "__main__":
    wait_for_server(ENV_BASE_URL)
    for task in TASKS:
        run_task(task)
