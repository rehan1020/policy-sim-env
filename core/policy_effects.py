"""
core/policy_effects.py
──────────────────────
The causal impact matrix and apply_policy() function.

All randomness is here, seeded and reproducible.
apply_policy() returns a KPI delta dict — it does NOT mutate CityState directly.
The caller (environment.py) decides what to do with the deltas.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from core.city_model import (
    KPI_NAMES,
    DISTRICTS,
    CityState,
    DistrictKPIs,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Available policy types
# ─────────────────────────────────────────────────────────────────────────────

POLICY_TYPES: List[str] = [
    "expand_transit",
    "build_housing",
    "congestion_tax",
    "green_spaces",
    "subsidise_rent",
    "zoning_reform",
    "bike_lanes",
    "emissions_tax",
    "income_support",
    "parking_reform",
]

POLICY_DESCRIPTIONS: Dict[str, str] = {
    "expand_transit":  "Fund new bus/rail routes — reduces traffic, mild housing & equality gains",
    "build_housing":   "Approve new housing developments — improves affordability, mild air cost",
    "congestion_tax":  "Charge drivers to enter downtown — strong traffic cut, public backlash",
    "green_spaces":    "Convert lots to parks/trees — improves air & satisfaction, costly on housing",
    "subsidise_rent":  "Direct rent subsidies for low-income tenants — strong equality & housing",
    "zoning_reform":   "Rezone areas for mixed use — moderate housing, enables future growth",
    "bike_lanes":      "Build protected cycling infrastructure — traffic & air improvement",
    "emissions_tax":   "Tax high-emission vehicles/businesses — strong air quality, satisfaction dip",
    "income_support":  "Direct income transfers to lowest earners — strong equality & satisfaction",
    "parking_reform":  "Reduce parking minimums — traffic & housing improvement, mixed reaction",
}

# ─────────────────────────────────────────────────────────────────────────────
#  Causal impact matrix
#  Format: policy_type → {kpi: base_effect_per_unit_budget}
#
#  Effects are at budget_pct = 1.0, district multiplier = 1.0.
#  Negative traffic = improvement (less congestion).
#  All values will be scaled by actual budget_pct and district multiplier.
# ─────────────────────────────────────────────────────────────────────────────

IMPACT_MATRIX: Dict[str, Dict[str, float]] = {
    #                      traffic  housing  equality  air_quality  satisfaction
    "expand_transit":  {"traffic": -12, "housing":  +3, "equality":  +5, "air_quality":  +4, "satisfaction":  +6},
    "build_housing":   {"traffic":  -2, "housing": +15, "equality":  +8, "air_quality":  -3, "satisfaction":  +4},
    "congestion_tax":  {"traffic": -18, "housing":  +1, "equality":  -4, "air_quality":  +8, "satisfaction":  -8},
    "green_spaces":    {"traffic":  -1, "housing":  -2, "equality":  +2, "air_quality": +12, "satisfaction":  +9},
    "subsidise_rent":  {"traffic":  +1, "housing": +10, "equality": +12, "air_quality":   0, "satisfaction":  +7},
    "zoning_reform":   {"traffic":  -3, "housing":  +8, "equality":  +6, "air_quality":  -2, "satisfaction":  +2},
    "bike_lanes":      {"traffic":  -5, "housing":   0, "equality":  +3, "air_quality":  +7, "satisfaction":  +5},
    "emissions_tax":   {"traffic":  -4, "housing":   0, "equality":  -3, "air_quality": +15, "satisfaction":  -5},
    "income_support":  {"traffic":   0, "housing":  +2, "equality": +18, "air_quality":   0, "satisfaction":  +8},
    "parking_reform":  {"traffic":  -8, "housing":  +4, "equality":  +1, "air_quality":  +3, "satisfaction":  -3},
}

# ── Minimum budget fraction needed to get any meaningful effect ───────────────
MIN_BUDGET_FOR_EFFECT: float = 0.05

# ── Noise standard deviation as a fraction of the base effect ────────────────
NOISE_STD: float = 0.20   # ±20% noise on each KPI delta


# ─────────────────────────────────────────────────────────────────────────────
#  apply_policy()
# ─────────────────────────────────────────────────────────────────────────────

def apply_policy(
    policy_type: str,
    district: str,
    budget_pct: float,
    city_state: CityState,
    rng: Optional[random.Random] = None,
) -> Tuple[Dict[str, float], str]:
    """
    Compute the KPI deltas that result from applying a policy to a district.

    Returns:
        (kpi_deltas, warning_message)
        kpi_deltas — dict[kpi_name → signed delta] for the target district
        warning_message — empty string if OK, otherwise a human-readable warning

    Does NOT mutate city_state. The caller applies the deltas.
    """
    if rng is None:
        rng = random.Random()

    # ── Validation ────────────────────────────────────────────────────────────
    warnings = []
    if policy_type not in IMPACT_MATRIX:
        return {k: 0.0 for k in KPI_NAMES}, f"Unknown policy_type: {policy_type}"
    if district not in DISTRICTS:
        return {k: 0.0 for k in KPI_NAMES}, f"Unknown district: {district}"
    if budget_pct < 0 or budget_pct > 1.0:
        return {k: 0.0 for k in KPI_NAMES}, f"budget_pct must be in [0, 1], got {budget_pct}"
    if budget_pct > city_state.budget_left + 1e-9:
        warnings.append(
            f"Requested {budget_pct:.0%} but only {city_state.budget_left:.0%} budget left. "
            "Effect scaled down."
        )
        budget_pct = city_state.budget_left

    if budget_pct < MIN_BUDGET_FOR_EFFECT:
        return {k: 0.0 for k in KPI_NAMES}, "budget_pct too small — no meaningful effect"

    # ── Compute deltas ────────────────────────────────────────────────────────
    base_effects = IMPACT_MATRIX[policy_type]
    district_obj = DISTRICTS[district]
    dist_mult = district_obj.multiplier_for(policy_type)

    deltas: Dict[str, float] = {}
    for kpi in KPI_NAMES:
        base = base_effects.get(kpi, 0.0)
        noise = rng.gauss(1.0, NOISE_STD)         # multiplicative noise
        noise = max(0.5, min(1.5, noise))          # clamp noise to ±50%
        raw_delta = base * budget_pct * dist_mult * noise
        deltas[kpi] = round(raw_delta, 3)

    warning = "; ".join(warnings)
    return deltas, warning


def policy_summary() -> List[Dict]:
    """Return a list of dicts describing all available policies."""
    return [
        {
            "policy_type": pt,
            "description": POLICY_DESCRIPTIONS[pt],
            "base_effects": IMPACT_MATRIX[pt],
        }
        for pt in POLICY_TYPES
    ]
