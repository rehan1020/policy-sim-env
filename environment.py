"""
environment.py
──────────────
PolicyEnvironment: the core RL environment.

Implements the OpenEnv interface:
  reset(task_id)  → CityObservation
  step(action)    → (CityObservation, reward, done, info)
  state()         → EpisodeState
"""
from __future__ import annotations

import random
import uuid
from typing import Any, Dict, Optional, Tuple

from core.city_model import (
    CityState,
    KPI_NAMES,
    KPI_WEIGHTS,
    KPI_INVERT,
    DISTRICT_NAMES,
)
from core.policy_effects import (
    POLICY_TYPES,
    apply_policy,
)
from core.stakeholders import (
    StakeholderEngine,
    STAKEHOLDERS,
    STAKEHOLDER_NAMES,
)
from models import CityObservation, EpisodeState, PolicyAction
from tasks import TASKS, Task


# ─────────────────────────────────────────────────────────────────────────────
#  Reward shaping constants
# ─────────────────────────────────────────────────────────────────────────────

STEP_PENALTY          = -0.02   # base cost per turn
PASS_TURN_PENALTY     = -0.05   # extra cost for doing nothing
RECALL_PENALTY        = -0.30   # terminal penalty if recalled
INVESTIGATION_REWARD  =  0.0    # neutral (information gathering)
BUDGET_OVERRUN_SCALE  =  0.10   # -0.10 per 10% over budget (not used; budget caps)

# KPI delta → reward scaling: total weighted delta / 100
KPI_REWARD_SCALE      = 1.0 / 100.0
SCORE_EPSILON         = 0.0001


def _strict_open_unit(score: float, eps: float = SCORE_EPSILON) -> float:
    """Clamp score into the strict open interval (0, 1)."""
    try:
        score = float(score)
    except Exception:
        score = 0.0
    if score <= 0.0:
        return eps
    if score >= 1.0:
        return 1.0 - eps
    return score


# ─────────────────────────────────────────────────────────────────────────────
#  PolicyEnvironment
# ─────────────────────────────────────────────────────────────────────────────

class PolicyEnvironment:

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._rng = random.Random(seed)
        self._task: Optional[Task] = None
        self._city: Optional[CityState] = None
        self._stakeholders: Optional[StakeholderEngine] = None
        self._episode_id: str = str(uuid.uuid4())
        self._done: bool = True
        self._done_reason: Optional[str] = None
        self._cumulative_reward: float = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    #  OpenEnv interface
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> CityObservation:
        """Start a new episode."""
        if task_id and task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id!r}. Available: {list(TASKS.keys())}")
        if task_id is None:
            task_id = list(TASKS.keys())[0]

        if seed is not None:
            self._rng = random.Random(seed)

        self._task = TASKS[task_id]
        self._city = self._task.make_city()
        self._stakeholders = self._task.make_stakeholders()
        self._episode_id = str(uuid.uuid4())
        self._done = False
        self._done_reason = None
        self._cumulative_reward = 0.0

        return self._build_obs(
            reward=0.0,
            done=False,
            info={"event": "reset"},
        )

    def step(self, action: PolicyAction) -> Tuple[CityObservation, float, bool, Dict[str, Any]]:
        """Take one action. Returns (observation, reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode is over — call reset() first.")
        assert self._task and self._city and self._stakeholders

        reward = STEP_PENALTY
        info: Dict[str, Any] = {}
        kpi_delta: Optional[Dict[str, float]] = None
        reactions: Optional[Dict[str, str]] = None
        capital_delta: Optional[float] = None
        warning: Optional[str] = None
        error: Optional[str] = None
        investigation_result: Optional[Dict] = None
        last_policy = None
        last_district = None
        last_budget_spent = None

        # ── Action dispatch ───────────────────────────────────────────────────

        if action.action_type == "investigate":
            result = self._do_investigate(action)
            if "error" in result:
                error = result["error"]
                reward += -0.05
            else:
                investigation_result = result
                reward += INVESTIGATION_REWARD

        elif action.action_type == "pass_turn":
            reward += PASS_TURN_PENALTY
            info["event"] = "passed_turn"

        elif action.action_type == "propose_policy":
            result = self._do_propose(action)
            if "error" in result:
                error = result["error"]
                reward += -0.05
            else:
                kpi_delta       = result["kpi_delta"]
                reactions       = result["reactions"]
                capital_delta   = result["capital_delta"]
                warning         = result.get("warning")
                last_policy     = result["policy_type"]
                last_district   = result["district"]
                last_budget_spent = result["budget_spent"]
                reward          += result["kpi_reward"]
                reward          += result["political_penalty"]
                info["kpi_delta"] = kpi_delta

        else:
            error = (
                f"Unknown action_type: {action.action_type!r}. "
                "Use 'propose_policy', 'investigate', or 'pass_turn'."
            )
            reward += -0.05

        # ── Advance turn & check terminal ─────────────────────────────────────
        self._city.turn += 1
        self._stakeholders.end_of_turn_recovery()

        if self._stakeholders.is_recalled():
            self._done = True
            self._done_reason = "recalled"
            reward += RECALL_PENALTY
            info["event"] = "recalled"
        elif self._city.is_terminal():
            self._done = True
            self._done_reason = "max_turns"
            grader_score = _strict_open_unit(self._task.grader(self._city))
            info["grader_score"] = grader_score
            info["event"] = "episode_end"

        self._cumulative_reward += reward

        obs = self._build_obs(
            reward=reward,
            done=self._done,
            info=info,
            kpi_delta=kpi_delta,
            reactions=reactions,
            capital_delta=capital_delta,
            warning=warning,
            error=error,
            investigation_result=investigation_result,
            last_action_type=action.action_type,
            last_policy=last_policy,
            last_district=last_district,
            last_budget_spent=last_budget_spent,
        )
        return obs, round(reward, 4), self._done, info

    def state(self) -> EpisodeState:
        """Return lightweight episode metadata."""
        if self._city is None:
            return EpisodeState(
                episode_id=self._episode_id,
                task_id="none",
            )
        turns_left = max(0, self._task.max_turns - self._city.turn)
        return EpisodeState(
            episode_id=self._episode_id,
            task_id=self._task.task_id,
            turn=self._city.turn,
            max_turns=self._task.max_turns,
            turns_left=turns_left,
            cumulative_reward=round(self._cumulative_reward, 4),
            composite_score=self._city.composite_score(),
            political_capital=round(self._stakeholders.political_capital, 1),
            budget_left=round(self._city.budget_left, 3),
            done=self._done,
            done_reason=self._done_reason,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Action handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _do_investigate(self, action: PolicyAction) -> Dict:
        sname = (action.stakeholder or "").strip()
        if not sname:
            return {"error": "investigate requires 'stakeholder' field."}
        if sname not in STAKEHOLDERS:
            return {"error": f"Unknown stakeholder: {sname!r}. Options: {STAKEHOLDER_NAMES}"}

        sh = STAKEHOLDERS[sname]
        approval = self._stakeholders.approval_ratings[sname]

        return {
            "stakeholder": sname,
            "description": sh.description,
            "current_approval": round(approval, 1),
            "loved_policies": sh.loved_policies,
            "hated_policies": sh.hated_policies,
            "power_level": sh.power_level,
            "tip": (
                f"This stakeholder supports: {sh.loved_policies}. "
                f"Avoid: {sh.hated_policies}."
            ),
        }

    def _do_propose(self, action: PolicyAction) -> Dict:
        # Validate fields
        pt = (action.policy_type or "").strip()
        dist = (action.district or "").strip()
        bpct = action.budget_pct

        if not pt:
            return {"error": "propose_policy requires 'policy_type'."}
        if not dist:
            return {"error": "propose_policy requires 'district'."}
        if bpct is None:
            return {"error": "propose_policy requires 'budget_pct'."}
        if pt not in POLICY_TYPES:
            return {"error": f"Unknown policy_type: {pt!r}. Options: {POLICY_TYPES}"}
        if dist not in DISTRICT_NAMES:
            return {"error": f"Unknown district: {dist!r}. Options: {DISTRICT_NAMES}"}
        if not (0 < bpct <= 1.0):
            return {"error": f"budget_pct must be in (0, 1]. Got {bpct}"}

        # Compute effects (on target district)
        city_before = self._city.citywide_kpis()
        deltas, warn = apply_policy(pt, dist, bpct, self._city, self._rng)

        # Apply deltas to the target district
        actual_budget = min(bpct, self._city.budget_left)
        for kpi, delta in deltas.items():
            self._city.kpis[dist].delta(kpi, delta)
        self._city.budget_left = max(0.0, self._city.budget_left - actual_budget)

        # Citywide KPI delta
        city_after = self._city.citywide_kpis()
        citywide_delta = {k: round(city_after[k] - city_before[k], 3) for k in KPI_NAMES}

        # Stakeholder reaction + capital
        sh_result = self._stakeholders.process_policy(pt)

        # KPI component of reward
        kpi_reward = 0.0
        for kpi in KPI_NAMES:
            d = citywide_delta[kpi]
            if KPI_INVERT[kpi]:
                d = -d   # inverted: going down is good
            kpi_reward += KPI_WEIGHTS[kpi] * d * KPI_REWARD_SCALE

        # Political penalty per opposed stakeholder
        political_penalty = -0.03 * sh_result["opposed_count"]

        # Record in history
        self._city.history.append({
            "turn": self._city.turn,
            "policy_type": pt,
            "district": dist,
            "budget_pct": actual_budget,
            "kpi_delta": citywide_delta,
        })

        return {
            "policy_type": pt,
            "district": dist,
            "budget_spent": round(actual_budget, 3),
            "kpi_delta": citywide_delta,
            "reactions": sh_result["reactions"],
            "capital_delta": sh_result["capital_delta"],
            "kpi_reward": round(kpi_reward, 4),
            "political_penalty": round(political_penalty, 4),
            "warning": warn or None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Observation builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_obs(
        self,
        reward: float,
        done: bool,
        info: Dict,
        kpi_delta: Optional[Dict] = None,
        reactions: Optional[Dict] = None,
        capital_delta: Optional[float] = None,
        warning: Optional[str] = None,
        error: Optional[str] = None,
        investigation_result: Optional[Dict] = None,
        last_action_type: Optional[str] = None,
        last_policy: Optional[str] = None,
        last_district: Optional[str] = None,
        last_budget_spent: Optional[float] = None,
    ) -> CityObservation:
        turns_left = max(0, self._task.max_turns - self._city.turn)
        return CityObservation(
            task_id=self._task.task_id,
            task_description=self._task.description,
            citywide_kpis=self._city.citywide_kpis(),
            district_kpis={
                name: dkpi.as_dict()
                for name, dkpi in self._city.kpis.items()
            },
            composite_score=self._city.composite_score(),
            budget_left=round(self._city.budget_left, 3),
            political_capital=round(self._stakeholders.political_capital, 1),
            stakeholder_approvals=self._stakeholders.snapshot()["approval_ratings"],
            turn=self._city.turn,
            turns_left=turns_left,
            last_action_type=last_action_type,
            last_policy_type=last_policy,
            last_district=last_district,
            last_budget_spent=last_budget_spent,
            kpi_delta=kpi_delta,
            stakeholder_reactions=reactions,
            capital_delta=capital_delta,
            warning=warning,
            error_message=error,
            investigation_result=investigation_result,
            reward=round(reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            done=done,
            done_reason=self._done_reason,
            info=info,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Convenience
    # ─────────────────────────────────────────────────────────────────────────

    def grade_current(self) -> float:
        """Grade the current city state for the active task."""
        if self._task is None or self._city is None:
            raise RuntimeError("No active episode.")
        return _strict_open_unit(self._task.grader(self._city))
