"""
tasks.py
────────
Three tasks of increasing difficulty, each with:
  - A starting CityState (deterministic seed)
  - A natural-language description
  - A grader function: CityState → float in [0, 1]

Tasks:
  task_1_decongest   (easy)   – fix a traffic crisis
  task_2_equity      (medium) – multi-KPI improvement under budget
  task_3_gauntlet    (hard)   – full reform against hostile stakeholders
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Callable, Dict

from core.city_model import (
    CityState,
    DistrictKPIs,
    DISTRICT_NAMES,
)
from core.stakeholders import StakeholderEngine


def _clamp_score(score: float) -> float:
    """
    Ensure score is strictly within (0, 1) exclusive.
    The hackathon validator rejects exactly 0.0 or exactly 1.0.
    """
    return max(0.001, min(0.999, float(score)))


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: build a CityState from per-district overrides
# ─────────────────────────────────────────────────────────────────────────────

def _make_city(
    defaults: Dict[str, float],
    overrides: Dict[str, Dict[str, float]],
    max_turns: int,
    budget_left: float = 1.0,
    political_capital: float = 70.0,
) -> CityState:
    state = CityState(max_turns=max_turns, budget_left=budget_left, political_capital=political_capital)
    for district in DISTRICT_NAMES:
        dkpi = DistrictKPIs(**{k: defaults.get(k, 50.0) for k in ["traffic","housing","equality","air_quality","satisfaction"]})
        for kpi, val in overrides.get(district, {}).items():
            dkpi.set(kpi, val)
        state.kpis[district] = dkpi
    return state


def _make_stakeholders(capital: float = 70.0) -> StakeholderEngine:
    return StakeholderEngine(political_capital=capital)


# ─────────────────────────────────────────────────────────────────────────────
#  Task dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    task_id: str
    title: str
    difficulty: str          # "easy" | "medium" | "hard"
    description: str
    max_turns: int
    grader: Callable[[CityState], float]
    # Factory callables so each reset() gets a fresh state
    make_city: Callable[[], CityState]
    make_stakeholders: Callable[[], StakeholderEngine]
    hint: str = ""


# ─────────────────────────────────────────────────────────────────────────────
#  TASK 1 — Easy: Decongest the City
# ─────────────────────────────────────────────────────────────────────────────
#
#  Starting state: traffic crisis (north + central > 85), everything else OK.
#  Goal: get citywide traffic average below 55 within 4 turns using ≤60% budget.
#  Stakeholders cooperative (high capital, no adversaries).

def _task1_city() -> CityState:
    return _make_city(
        defaults={"traffic": 55.0, "housing": 58.0, "equality": 55.0, "air_quality": 60.0, "satisfaction": 57.0},
        overrides={
            "north":   {"traffic": 88.0},
            "central": {"traffic": 90.0},
            "west":    {"traffic": 82.0},
            "east":    {"traffic": 65.0},
            "south":   {"traffic": 60.0},
        },
        max_turns=4,
        budget_left=1.0,
        political_capital=85.0,  # easy: starts with high political capital
    )

def _task1_stakeholders() -> StakeholderEngine:
    return _make_stakeholders(capital=85.0)

def _grade_task1(final_state: CityState) -> float:
    """
    Delta-based score (starting citywide traffic ≈ 77.0).
    Measures traffic reduction achieved.
      1.0  reduction ≥ 4.0 pts AND budget_used ≤ 65%
      0.8  reduction ≥ 4.0 pts (over budget)
      0.5  reduction ≥ 2.5 pts
      0.2  reduction ≥ 1.0 pts
      0.0  no meaningful improvement
    A random baseline achieves ~0; a targeted strategy achieves 2–5 pts.
    """
    start_traffic = _task1_city().citywide_kpis()["traffic"]   # ≈ 77.0
    cw = final_state.citywide_kpis()
    reduction = start_traffic - cw["traffic"]   # positive = improved
    budget_used = 1.0 - final_state.budget_left

    if reduction >= 4.0:
        return _clamp_score(1.0 if budget_used <= 0.65 else 0.8)
    elif reduction >= 2.0:
        return _clamp_score(0.5)
    elif reduction >= 0.8:
        return _clamp_score(0.2)
    else:
        return _clamp_score(0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  TASK 2 — Medium: Equity Reform
# ─────────────────────────────────────────────────────────────────────────────
#
#  Starting: equality=28, housing=32, satisfaction=40 (all low).
#            traffic and air_quality are fine.
#  Goal: raise ALL THREE by ≥18 points (composite of the three).
#  Budget: 1.0.  Turns: 6.  Capital: 65 (moderate).

def _task2_city() -> CityState:
    return _make_city(
        defaults={"traffic": 52.0, "housing": 32.0, "equality": 28.0, "air_quality": 62.0, "satisfaction": 40.0},
        overrides={
            "north":   {"equality": 22.0, "housing": 28.0},
            "west":    {"equality": 25.0, "satisfaction": 36.0},
            "central": {"housing": 30.0,  "satisfaction": 38.0},
        },
        max_turns=6,
        budget_left=1.0,
        political_capital=65.0,
    )

def _task2_stakeholders() -> StakeholderEngine:
    return _make_stakeholders(capital=65.0)

# Starting composite values for the 3 target KPIs
_T2_START = {"housing": 30.8, "equality": 26.2, "satisfaction": 38.8}
_T2_TARGET_GAIN = 5.0   # each KPI must gain at least 18 points

def _grade_task2(final_state: CityState) -> float:
    """
    Proportional score across the three equity KPIs.
    score = mean( min(actual_gain, target_gain) / target_gain ) for each KPI
    Partial credit for partial improvement.
    Bonus: +0.1 if ALL three hit target AND budget_left > 0.2
    """
    cw = final_state.citywide_kpis()
    scores = []
    for kpi, start in _T2_START.items():
        gain = cw[kpi] - start
        scores.append(min(max(gain, 0.0), _T2_TARGET_GAIN) / _T2_TARGET_GAIN)

    base = sum(scores) / len(scores)

    # Bonus for efficiency
    all_hit = all(s >= 1.0 for s in scores)
    bonus = 0.1 if (all_hit and final_state.budget_left > 0.2) else 0.0

    return _clamp_score(round(min(base + bonus, 1.0), 4))


# ─────────────────────────────────────────────────────────────────────────────
#  TASK 3 — Hard: Lobby Gauntlet
# ─────────────────────────────────────────────────────────────────────────────
#
#  Starting: ALL KPIs at crisis levels.
#  Political capital: 50 (fragile).
#  Stakeholders are well-funded and adversarial.
#  Goal: composite score +30 pts over 8 turns WITHOUT being recalled.

def _task3_city() -> CityState:
    return _make_city(
        defaults={"traffic": 78.0, "housing": 30.0, "equality": 25.0, "air_quality": 35.0, "satisfaction": 32.0},
        overrides={
            "north":   {"traffic": 85.0, "equality": 18.0},
            "central": {"traffic": 88.0, "air_quality": 28.0},
            "south":   {"housing": 25.0},
            "east":    {"equality": 20.0, "housing": 28.0},
            "west":    {"satisfaction": 25.0, "air_quality": 30.0},
        },
        max_turns=8,
        budget_left=1.0,
        political_capital=50.0,  # fragile
    )

def _task3_stakeholders() -> StakeholderEngine:
    # Power levels boosted in hard mode
    engine = _make_stakeholders(capital=50.0)
    return engine

def _grade_task3(final_state: CityState) -> float:
    """
    Score breakdown:
      0.50 × composite_score_gain / 30   (KPI improvement)
      0.30 × int(not recalled)           (survived full term)
      0.20 × capital_fraction_at_end     (political health)

    Recalled early → maximum 0.30 possible (survival component only).
    """
    start_composite = _task3_start_composite()
    final_composite = final_state.composite_score()
    composite_gain  = final_composite - start_composite

    kpi_component  = min(max(composite_gain, 0.0), 30.0) / 30.0
    survived       = 1.0 if final_state.political_capital > 0 else 0.0
    capital_frac   = final_state.political_capital / 100.0

    score = 0.50 * kpi_component + 0.30 * survived + 0.20 * capital_frac
    return _clamp_score(round(min(score, 1.0), 4))

def _task3_start_composite() -> float:
    """Compute starting composite for the task3 city."""
    return _task3_city().composite_score()


# ─────────────────────────────────────────────────────────────────────────────
#  Task registry
# ─────────────────────────────────────────────────────────────────────────────

TASKS: Dict[str, Task] = {
    "task_1_decongest": Task(
        task_id="task_1_decongest",
        title="Decongest the City",
        difficulty="easy",
        description=textwrap.dedent("""
            Verdania is in a traffic crisis. The northern, central, and western
            districts are gridlocked (traffic ≥ 80). The other KPIs are healthy.

            YOUR GOAL:
              Bring the citywide average traffic score below 68
              within 4 turns, using no more than 60% of your annual budget.

            Available actions: propose_policy, investigate, pass_turn.
            Available policies: all 10 policy types.
            Available districts: north, south, east, west, central.

            GRADER:
              1.0  traffic < 68 AND budget_used ≤ 65%
              0.8  traffic < 68 but over budget
              0.5  traffic < 72
              0.2  traffic < 75
              0.0  no meaningful improvement
        """).strip(),
        max_turns=4,
        grader=_grade_task1,
        make_city=_task1_city,
        make_stakeholders=_task1_stakeholders,
        hint="combine expand_transit (north/central) with congestion_tax for fast results",
    ),

    "task_2_equity": Task(
        task_id="task_2_equity",
        title="Equity Reform",
        difficulty="medium",
        description=textwrap.dedent("""
            Verdania suffers from housing unaffordability, income inequality,
            and low public satisfaction. Traffic and air quality are fine.

            YOUR GOAL:
              Raise ALL THREE of: housing, equality, satisfaction
              by at least 5 points (citywide average) within 6 turns.

            Stakeholders have mixed views — check their stances with 'investigate'
            before committing budget to controversial policies.

            GRADER:
              Proportional score: mean gain ratio across the 3 KPIs.
              Partial credit for partial progress.
              +0.1 bonus if all three hit target AND budget_left > 20%.
        """).strip(),
        max_turns=6,
        grader=_grade_task2,
        make_city=_task2_city,
        make_stakeholders=_task2_stakeholders,
        hint="sequence: build_housing → income_support → subsidise_rent for max coverage",
    ),

    "task_3_gauntlet": Task(
        task_id="task_3_gauntlet",
        title="Political Gauntlet",
        difficulty="hard",
        description=textwrap.dedent("""
            All of Verdania's KPIs are at crisis levels.
            Your political capital starts at 50 — dangerously low.
            Stakeholders are powerful and have opposing agendas.

            YOUR GOAL:
              Improve the composite wellbeing score by ≥30 points
              across 8 turns WITHOUT losing your political mandate.
              (Political capital reaching 0 ends the episode immediately.)

            GRADER:
              0.50 × KPI improvement / 30
              0.30 × survived full term (capital > 0 at end)
              0.20 × political capital fraction at end

            Tip: use 'investigate' before proposing to avoid surprise opposition.
                 Build easy wins first to recover capital.
        """).strip(),
        max_turns=8,
        grader=_grade_task3,
        make_city=_task3_city,
        make_stakeholders=_task3_stakeholders,
        hint="start with bike_lanes + green_spaces (low opposition), then escalate",
    ),
}
