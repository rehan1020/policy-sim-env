"""
models.py
─────────
Typed Pydantic models implementing the OpenEnv spec:
  PolicyAction      ← what the agent sends
  CityObservation   ← what the agent receives
  EpisodeState      ← lightweight metadata (GET /state)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
#  ACTION
# ─────────────────────────────────────────────────────────────────────────────

class PolicyAction(BaseModel):
    """
    An action the agent can take each turn.

    action_type choices:
      "propose_policy"  – spend budget on a policy in a district (main action)
      "investigate"     – query a stakeholder's current stance without spending
      "pass_turn"       – skip this turn (small penalty; sometimes strategically useful)
    """
    action_type: str = Field(
        ...,
        description="One of: 'propose_policy', 'investigate', 'pass_turn'",
    )
    policy_type: Optional[str] = Field(
        default=None,
        description="Policy to apply — required for propose_policy. "
                    "One of: expand_transit, build_housing, congestion_tax, "
                    "green_spaces, subsidise_rent, zoning_reform, bike_lanes, "
                    "emissions_tax, income_support, parking_reform",
    )
    district: Optional[str] = Field(
        default=None,
        description="Target district — required for propose_policy. "
                    "One of: north, south, east, west, central",
    )
    budget_pct: Optional[float] = Field(
        default=None,
        description="Fraction of annual budget to spend (0.05–1.0). Required for propose_policy.",
    )
    stakeholder: Optional[str] = Field(
        default=None,
        description="Stakeholder to investigate — required for investigate action. "
                    "One of: transit_union, property_owners, env_coalition, business_council",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
#  OBSERVATION
# ─────────────────────────────────────────────────────────────────────────────

class CityObservation(BaseModel):
    """What the agent observes after every reset() / step()."""

    # ── Task context ──────────────────────────────────────────────────────────
    task_id: str = Field(..., description="Active task identifier")
    task_description: str = Field(..., description="Natural-language goal")

    # ── City state ────────────────────────────────────────────────────────────
    citywide_kpis: Dict[str, float] = Field(
        ..., description="Average KPIs across all districts (0–100 each)"
    )
    district_kpis: Dict[str, Dict[str, float]] = Field(
        ..., description="Per-district KPI values"
    )
    composite_score: float = Field(..., description="Weighted wellbeing score (0–100)")

    # ── Resources ─────────────────────────────────────────────────────────────
    budget_left: float = Field(..., description="Fraction of annual budget remaining (0–1)")
    political_capital: float = Field(..., description="Political capital remaining (0–100)")

    # ── Stakeholders ──────────────────────────────────────────────────────────
    stakeholder_approvals: Dict[str, float] = Field(
        ..., description="Per-stakeholder approval rating (0–100)"
    )

    # ── Turn info ─────────────────────────────────────────────────────────────
    turn: int = Field(..., description="Current turn (1-indexed for readability)")
    turns_left: int = Field(..., description="Turns remaining")

    # ── Feedback from last action ─────────────────────────────────────────────
    last_action_type: Optional[str] = Field(default=None)
    last_policy_type: Optional[str] = Field(default=None)
    last_district: Optional[str] = Field(default=None)
    last_budget_spent: Optional[float] = Field(default=None)
    kpi_delta: Optional[Dict[str, float]] = Field(
        default=None, description="KPI changes caused by last policy (citywide avg)"
    )
    stakeholder_reactions: Optional[Dict[str, str]] = Field(
        default=None, description="Stance per stakeholder for last policy"
    )
    capital_delta: Optional[float] = Field(default=None)
    warning: Optional[str] = Field(default=None, description="Non-fatal issue with last action")
    error_message: Optional[str] = Field(default=None, description="Fatal action error")

    # ── Investigate result ────────────────────────────────────────────────────
    investigation_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Stakeholder stance prediction (from investigate action)",
    )

    # ── Episode signals ───────────────────────────────────────────────────────
    reward: float = Field(default=0.0)
    cumulative_reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    done_reason: Optional[str] = Field(default=None)
    info: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
#  EPISODE STATE  (lightweight, for GET /state)
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeState(BaseModel):
    episode_id: str
    task_id: str
    turn: int = 0
    max_turns: int = 8
    turns_left: int = 8
    cumulative_reward: float = 0.0
    composite_score: float = 0.0
    political_capital: float = 70.0
    budget_left: float = 1.0
    done: bool = False
    done_reason: Optional[str] = None
