"""
core/city_model.py
──────────────────
Defines the city of Verdania: its 5 districts, 5 KPIs, and the CityState
dataclass that holds the full simulation state at any point in time.

Nothing in this file is random — all randomness lives in policy_effects.py.
This makes CityState fully serialisable and unit-testable.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  KPI names & weights (used by reward function)
# ─────────────────────────────────────────────────────────────────────────────

KPI_NAMES: List[str] = [
    "traffic",        # congestion level  — lower is better
    "housing",        # affordability     — higher is better
    "equality",       # income equality   — higher is better
    "air_quality",    # AQI score         — higher is better
    "satisfaction",   # public approval   — higher is better
]

# For reward computation: how much each KPI delta contributes
KPI_WEIGHTS: Dict[str, float] = {
    "traffic":     0.25,
    "housing":     0.25,
    "equality":    0.20,
    "air_quality": 0.15,
    "satisfaction":0.15,
}

# All KPIs are kept in [0, 100].
# For "traffic" a HIGH value = BAD (more congestion).
# For all others a HIGH value = GOOD.
# The reward function inverts traffic so Δ is always "higher = better".
KPI_INVERT: Dict[str, bool] = {
    "traffic":      True,   # improvement = going DOWN
    "housing":      False,
    "equality":     False,
    "air_quality":  False,
    "satisfaction": False,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Districts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class District:
    name: str
    population_density: str          # "low" | "medium" | "high" | "very_high"
    income_level: str                # "low" | "medium" | "high" | "mixed"
    # Per-policy type multipliers: amplify or dampen policy effect in this district
    policy_multipliers: Dict[str, float] = field(default_factory=dict)

    def multiplier_for(self, policy_type: str) -> float:
        return self.policy_multipliers.get(policy_type, 1.0)


DISTRICTS: Dict[str, District] = {
    "north": District(
        name="north",
        population_density="high",
        income_level="low",
        policy_multipliers={
            "expand_transit":  1.4,   # dense + low income → huge transit benefit
            "subsidise_rent":  1.5,
            "income_support":  1.4,
            "build_housing":   1.2,
            "congestion_tax":  0.7,   # lower income hit harder by tax
            "emissions_tax":   0.8,
            "green_spaces":    0.9,
            "bike_lanes":      1.1,
            "zoning_reform":   1.0,
            "parking_reform":  0.9,
        },
    ),
    "south": District(
        name="south",
        population_density="medium",
        income_level="high",
        policy_multipliers={
            "expand_transit":  0.7,   # high income drives, low transit uptake
            "subsidise_rent":  0.5,
            "income_support":  0.6,
            "build_housing":   0.8,
            "congestion_tax":  1.3,   # high income can afford it, deters driving
            "emissions_tax":   1.2,
            "green_spaces":    1.3,   # wealthier area values parks
            "bike_lanes":      0.9,
            "zoning_reform":   0.8,
            "parking_reform":  1.2,
        },
    ),
    "east": District(
        name="east",
        population_density="low",
        income_level="medium",
        policy_multipliers={
            "expand_transit":  0.9,
            "subsidise_rent":  1.0,
            "income_support":  1.1,
            "build_housing":   1.3,   # low density → lots of space to build
            "congestion_tax":  1.0,
            "emissions_tax":   1.1,
            "green_spaces":    1.1,
            "bike_lanes":      0.8,   # too spread out for bikes
            "zoning_reform":   1.4,   # easy to rezone low-density land
            "parking_reform":  0.7,
        },
    ),
    "west": District(
        name="west",
        population_density="high",
        income_level="medium",
        policy_multipliers={
            "expand_transit":  1.2,
            "subsidise_rent":  1.1,
            "income_support":  1.0,
            "build_housing":   1.0,
            "congestion_tax":  1.1,
            "emissions_tax":   1.0,
            "green_spaces":    1.0,
            "bike_lanes":      1.3,   # dense + medium income → bike culture
            "zoning_reform":   0.9,
            "parking_reform":  1.1,
        },
    ),
    "central": District(
        name="central",
        population_density="very_high",
        income_level="mixed",
        policy_multipliers={
            "expand_transit":  1.3,
            "subsidise_rent":  1.3,
            "income_support":  1.2,
            "build_housing":   0.6,   # very dense — hard to build more
            "congestion_tax":  1.4,   # biggest congestion impact
            "emissions_tax":   1.1,
            "green_spaces":    0.7,   # almost no space left
            "bike_lanes":      1.4,
            "zoning_reform":   0.7,
            "parking_reform":  1.5,   # parking is at a premium
        },
    ),
}

DISTRICT_NAMES: List[str] = list(DISTRICTS.keys())


# ─────────────────────────────────────────────────────────────────────────────
#  CityState
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DistrictKPIs:
    """KPI values for a single district (0–100 each)."""
    traffic:      float = 50.0
    housing:      float = 50.0
    equality:     float = 50.0
    air_quality:  float = 50.0
    satisfaction: float = 50.0

    def get(self, kpi: str) -> float:
        return getattr(self, kpi)

    def set(self, kpi: str, value: float) -> None:
        setattr(self, kpi, max(0.0, min(100.0, float(value))))

    def delta(self, kpi: str, amount: float) -> float:
        """Apply delta, clamp, return actual change made."""
        old = self.get(kpi)
        self.set(kpi, old + amount)
        return self.get(kpi) - old

    def as_dict(self) -> Dict[str, float]:
        return {k: round(getattr(self, k), 2) for k in KPI_NAMES}

    def copy(self) -> "DistrictKPIs":
        return DistrictKPIs(**self.as_dict())


@dataclass
class CityState:
    """
    Complete city state at one point in time.

    kpis        – per-district KPI values
    budget_left – fraction of annual budget remaining (0.0–1.0)
    political_capital – 0–100; agent recalled if it hits 0
    turn        – current turn number (0-indexed)
    max_turns   – total turns in this episode
    history     – list of (policy_type, district, budget_pct, kpi_delta_citywide)
    """
    kpis: Dict[str, DistrictKPIs] = field(default_factory=dict)
    budget_left: float = 1.0
    political_capital: float = 70.0
    turn: int = 0
    max_turns: int = 8
    history: List[dict] = field(default_factory=list)

    # ── convenience ─────────────────────────────────────────────────────────

    @classmethod
    def default(cls, max_turns: int = 8) -> "CityState":
        """Create a default mid-range city state."""
        state = cls(max_turns=max_turns)
        for name in DISTRICT_NAMES:
            state.kpis[name] = DistrictKPIs()
        return state

    def citywide_kpis(self) -> Dict[str, float]:
        """Average KPI values across all districts."""
        totals = {k: 0.0 for k in KPI_NAMES}
        n = len(self.kpis)
        for dkpi in self.kpis.values():
            for k in KPI_NAMES:
                totals[k] += dkpi.get(k)
        return {k: round(v / n, 2) for k, v in totals.items()}

    def composite_score(self) -> float:
        """
        Weighted composite wellbeing score (0–100).
        Traffic is inverted so higher always = better.
        """
        cw = self.citywide_kpis()
        score = 0.0
        for kpi, weight in KPI_WEIGHTS.items():
            val = cw[kpi]
            if KPI_INVERT[kpi]:
                val = 100.0 - val  # invert traffic
            score += weight * val
        return round(score, 2)

    def copy(self) -> "CityState":
        new = CityState(
            budget_left=self.budget_left,
            political_capital=self.political_capital,
            turn=self.turn,
            max_turns=self.max_turns,
            history=copy.deepcopy(self.history),
        )
        new.kpis = {name: dkpi.copy() for name, dkpi in self.kpis.items()}
        return new

    def is_terminal(self) -> bool:
        return self.turn >= self.max_turns or self.political_capital <= 0

    def summary(self) -> Dict:
        return {
            "turn": self.turn,
            "max_turns": self.max_turns,
            "budget_left": round(self.budget_left, 3),
            "political_capital": round(self.political_capital, 1),
            "citywide_kpis": self.citywide_kpis(),
            "composite_score": self.composite_score(),
            "done": self.is_terminal(),
        }
