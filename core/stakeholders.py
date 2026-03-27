"""
core/stakeholders.py
─────────────────────
Four lobby groups that react to every policy.
Their collective approval drives the political capital resource.

Each stakeholder has:
  - policies they love (stance = "supportive",  capital_delta = +5)
  - policies they hate (stance = "opposed",     capital_delta = -8)
  - everything else   (stance = "neutral",      capital_delta =  0)

If ≥2 stakeholders oppose a policy → additional "coalition penalty" of -5.
If political capital reaches 0 → the agent is recalled (episode ends).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  Stakeholder definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stakeholder:
    name: str
    description: str
    loved_policies: List[str]     # → "supportive" stance, +capital
    hated_policies: List[str]     # → "opposed" stance, -capital
    power_level: float            # 0.5–2.0; multiplies capital_delta
    # Passively recovers capital if they haven't been upset recently
    recovery_per_turn: float = 2.0

    # capital deltas per stance (before power scaling)
    SUPPORT_DELTA: float = +5.0
    OPPOSE_DELTA:  float = -8.0

    def react(self, policy_type: str) -> Tuple[str, float]:
        """
        Returns (stance, capital_delta).
        stance: "supportive" | "neutral" | "opposed"
        """
        if policy_type in self.loved_policies:
            return "supportive", self.SUPPORT_DELTA * self.power_level
        elif policy_type in self.hated_policies:
            return "opposed", self.OPPOSE_DELTA * self.power_level
        else:
            return "neutral", 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  The four stakeholders
# ─────────────────────────────────────────────────────────────────────────────

STAKEHOLDERS: Dict[str, Stakeholder] = {
    "transit_union": Stakeholder(
        name="Transit Workers Union",
        description="Represents bus and rail workers. Supports public transit investment.",
        loved_policies=["expand_transit", "bike_lanes"],
        hated_policies=["parking_reform", "congestion_tax"],
        power_level=1.0,
        recovery_per_turn=2.0,
    ),
    "property_owners": Stakeholder(
        name="Property Owners Association",
        description="Represents landlords and developers. Opposes anything that caps rents.",
        loved_policies=["zoning_reform", "parking_reform"],
        hated_policies=["build_housing", "subsidise_rent", "income_support"],
        power_level=1.5,   # well-funded, high political influence
        recovery_per_turn=1.5,
    ),
    "env_coalition": Stakeholder(
        name="Environmental Coalition",
        description="Green advocacy group. Pushes hard on air quality and green space.",
        loved_policies=["emissions_tax", "green_spaces", "bike_lanes"],
        hated_policies=["build_housing", "expand_transit"],  # oppose land use
        power_level=0.8,
        recovery_per_turn=3.0,   # forgiving — recovers quickly
    ),
    "business_council": Stakeholder(
        name="Chamber of Commerce",
        description="Represents local businesses. Wants congestion down, taxes low.",
        loved_policies=["congestion_tax", "parking_reform", "expand_transit"],
        hated_policies=["emissions_tax", "income_support", "subsidise_rent"],
        power_level=1.2,
        recovery_per_turn=1.0,
    ),
}

STAKEHOLDER_NAMES: List[str] = list(STAKEHOLDERS.keys())

# Extra penalty when ≥2 stakeholders are simultaneously opposed
COALITION_OPPOSITION_PENALTY: float = -5.0


# ─────────────────────────────────────────────────────────────────────────────
#  StakeholderEngine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StakeholderEngine:
    """
    Manages stakeholder reactions and political capital over an episode.

    political_capital: 0–100. Agent recalled at 0.
    approval_ratings:  per-stakeholder current approval (0–100).
    """
    political_capital: float = 70.0
    approval_ratings: Dict[str, float] = field(default_factory=dict)
    _turns_since_upset: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for name in STAKEHOLDER_NAMES:
            if name not in self.approval_ratings:
                self.approval_ratings[name] = 60.0
            if name not in self._turns_since_upset:
                self._turns_since_upset[name] = 0

    # ── React to a policy ────────────────────────────────────────────────────

    def process_policy(self, policy_type: str) -> Dict:
        """
        React to a proposed policy.
        Returns a reactions dict and the net capital delta.
        """
        reactions = {}
        total_capital_delta = 0.0
        opposed_count = 0

        for sname, stakeholder in STAKEHOLDERS.items():
            stance, delta = stakeholder.react(policy_type)
            reactions[sname] = stance
            total_capital_delta += delta

            # Update per-stakeholder approval
            if stance == "supportive":
                self.approval_ratings[sname] = min(100.0, self.approval_ratings[sname] + 6)
                self._turns_since_upset[sname] += 1
            elif stance == "opposed":
                self.approval_ratings[sname] = max(0.0, self.approval_ratings[sname] - 10)
                self._turns_since_upset[sname] = 0
                opposed_count += 1
            else:
                self._turns_since_upset[sname] += 1

        # Coalition opposition penalty
        if opposed_count >= 2:
            total_capital_delta += COALITION_OPPOSITION_PENALTY

        # Apply to political capital
        old_capital = self.political_capital
        self.political_capital = max(0.0, min(100.0, self.political_capital + total_capital_delta))
        actual_delta = self.political_capital - old_capital

        return {
            "reactions": reactions,
            "capital_delta": round(actual_delta, 1),
            "new_capital": round(self.political_capital, 1),
            "opposed_count": opposed_count,
            "coalition_penalty_applied": opposed_count >= 2,
        }

    # ── End-of-turn recovery ──────────────────────────────────────────────────

    def end_of_turn_recovery(self) -> float:
        """
        Each stakeholder passively recovers approval at end of turn.
        Political capital also recovers slightly if nobody is upset.
        Returns capital recovered.
        """
        all_happy = True
        for sname, stakeholder in STAKEHOLDERS.items():
            turns = self._turns_since_upset[sname]
            if turns > 0:
                rec = min(stakeholder.recovery_per_turn * turns, 5.0)
                self.approval_ratings[sname] = min(
                    100.0, self.approval_ratings[sname] + rec
                )
            else:
                all_happy = False

        recovery = 3.0 if all_happy else 1.0
        old = self.political_capital
        self.political_capital = min(100.0, self.political_capital + recovery)
        return round(self.political_capital - old, 1)

    # ── State snapshot ────────────────────────────────────────────────────────

    def snapshot(self) -> Dict:
        return {
            "political_capital": round(self.political_capital, 1),
            "approval_ratings": {k: round(v, 1) for k, v in self.approval_ratings.items()},
        }

    def copy(self) -> "StakeholderEngine":
        return StakeholderEngine(
            political_capital=self.political_capital,
            approval_ratings=dict(self.approval_ratings),
            _turns_since_upset=dict(self._turns_since_upset),
        )

    def is_recalled(self) -> bool:
        return self.political_capital <= 0
