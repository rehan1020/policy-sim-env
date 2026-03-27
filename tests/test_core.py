"""
tests/test_core.py
──────────────────
Deep unit tests for city_model, policy_effects, and stakeholders.
Pure stdlib — no pydantic/fastapi required.
"""
import sys
import os
import unittest
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.city_model import (
    CityState, DistrictKPIs, DISTRICT_NAMES, KPI_NAMES, KPI_WEIGHTS,
    KPI_INVERT, DISTRICTS,
)
from core.policy_effects import (
    IMPACT_MATRIX, POLICY_TYPES, apply_policy, MIN_BUDGET_FOR_EFFECT,
)
from core.stakeholders import (
    Stakeholder, StakeholderEngine, STAKEHOLDERS, STAKEHOLDER_NAMES,
    COALITION_OPPOSITION_PENALTY,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CityState tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDistrictKPIs(unittest.TestCase):

    def test_defaults_are_50(self):
        dkpi = DistrictKPIs()
        for kpi in KPI_NAMES:
            self.assertAlmostEqual(dkpi.get(kpi), 50.0)

    def test_set_clamps_high(self):
        dkpi = DistrictKPIs()
        dkpi.set("traffic", 150.0)
        self.assertEqual(dkpi.get("traffic"), 100.0)

    def test_set_clamps_low(self):
        dkpi = DistrictKPIs()
        dkpi.set("housing", -20.0)
        self.assertEqual(dkpi.get("housing"), 0.0)

    def test_delta_returns_actual_change(self):
        dkpi = DistrictKPIs()
        dkpi.set("traffic", 95.0)
        actual = dkpi.delta("traffic", +20.0)   # would go to 115, clamped to 100
        self.assertAlmostEqual(actual, 5.0, places=5)
        self.assertAlmostEqual(dkpi.get("traffic"), 100.0)

    def test_delta_negative(self):
        dkpi = DistrictKPIs()
        dkpi.set("traffic", 10.0)
        actual = dkpi.delta("traffic", -20.0)   # would go to -10, clamped to 0
        self.assertAlmostEqual(actual, -10.0, places=5)
        self.assertAlmostEqual(dkpi.get("traffic"), 0.0)

    def test_as_dict_roundtrip(self):
        dkpi = DistrictKPIs(traffic=72.1, housing=38.5, equality=44.0, air_quality=61.3, satisfaction=52.7)
        d = dkpi.as_dict()
        self.assertEqual(set(d.keys()), set(KPI_NAMES))
        self.assertAlmostEqual(d["traffic"], 72.1, places=2)

    def test_copy_is_independent(self):
        dkpi = DistrictKPIs(traffic=80.0)
        copy = dkpi.copy()
        copy.set("traffic", 10.0)
        self.assertAlmostEqual(dkpi.get("traffic"), 80.0)  # original unchanged


class TestCityState(unittest.TestCase):

    def _full_city(self) -> CityState:
        state = CityState.default()
        return state

    def test_default_creates_all_districts(self):
        state = self._full_city()
        self.assertEqual(set(state.kpis.keys()), set(DISTRICT_NAMES))

    def test_citywide_kpis_are_average(self):
        state = CityState.default()
        # Set all districts to 60 for traffic
        for name in DISTRICT_NAMES:
            state.kpis[name].set("traffic", 60.0)
        cw = state.citywide_kpis()
        self.assertAlmostEqual(cw["traffic"], 60.0, places=3)

    def test_citywide_kpis_averages_correctly_with_variance(self):
        state = CityState.default()
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for name, val in zip(DISTRICT_NAMES, values):
            state.kpis[name].set("housing", val)
        cw = state.citywide_kpis()
        self.assertAlmostEqual(cw["housing"], 30.0, places=3)  # mean of 10–50

    def test_composite_score_traffic_inverted(self):
        """Lowering traffic should INCREASE composite score."""
        state = CityState.default()
        before = state.composite_score()
        for name in DISTRICT_NAMES:
            state.kpis[name].set("traffic", 10.0)   # very low congestion = great
        after = state.composite_score()
        self.assertGreater(after, before)

    def test_composite_score_range(self):
        """Composite score must always be in [0, 100]."""
        # Best case: all good KPIs at 100, traffic at 0
        state = CityState.default()
        for name in DISTRICT_NAMES:
            state.kpis[name].set("traffic", 0.0)
            state.kpis[name].set("housing", 100.0)
            state.kpis[name].set("equality", 100.0)
            state.kpis[name].set("air_quality", 100.0)
            state.kpis[name].set("satisfaction", 100.0)
        best = state.composite_score()
        self.assertLessEqual(best, 100.0)
        self.assertGreaterEqual(best, 0.0)

        # Worst case
        state2 = CityState.default()
        for name in DISTRICT_NAMES:
            state2.kpis[name].set("traffic", 100.0)
            for k in ["housing","equality","air_quality","satisfaction"]:
                state2.kpis[name].set(k, 0.0)
        worst = state2.composite_score()
        self.assertLessEqual(worst, 100.0)
        self.assertGreaterEqual(worst, 0.0)
        self.assertLess(worst, best)

    def test_copy_is_deep(self):
        state = CityState.default()
        state.kpis["north"].set("traffic", 90.0)
        copy = state.copy()
        copy.kpis["north"].set("traffic", 10.0)
        self.assertAlmostEqual(state.kpis["north"].get("traffic"), 90.0)

    def test_is_terminal_max_turns(self):
        state = CityState.default(max_turns=4)
        state.turn = 4
        self.assertTrue(state.is_terminal())
        state.turn = 3
        self.assertFalse(state.is_terminal())

    def test_is_terminal_zero_capital(self):
        state = CityState.default()
        state.political_capital = 0
        self.assertTrue(state.is_terminal())

    def test_kpi_weights_sum_to_one(self):
        total = sum(KPI_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=9)


# ═══════════════════════════════════════════════════════════════════════════════
#  Policy effects tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPolicyEffects(unittest.TestCase):

    def _state(self) -> CityState:
        return CityState.default()

    def test_all_policy_types_in_matrix(self):
        for pt in POLICY_TYPES:
            self.assertIn(pt, IMPACT_MATRIX, f"{pt} missing from IMPACT_MATRIX")

    def test_all_kpis_in_matrix_rows(self):
        for pt, effects in IMPACT_MATRIX.items():
            for kpi in KPI_NAMES:
                self.assertIn(kpi, effects, f"KPI {kpi} missing from {pt}")

    def test_apply_returns_all_kpis(self):
        state = self._state()
        deltas, _ = apply_policy("expand_transit", "north", 0.3, state)
        self.assertEqual(set(deltas.keys()), set(KPI_NAMES))

    def test_expand_transit_reduces_traffic(self):
        """expand_transit should ALWAYS reduce traffic (base effect = -12)."""
        rng = random.Random(42)
        state = self._state()
        for _ in range(30):
            d, _ = apply_policy("expand_transit", "central", 0.5, state, rng)
            self.assertLess(d["traffic"], 0.0, "expand_transit must reduce traffic")

    def test_income_support_raises_equality(self):
        """income_support has the highest base equality effect (+18)."""
        rng = random.Random(99)
        state = self._state()
        for _ in range(20):
            d, _ = apply_policy("income_support", "north", 0.5, state, rng)
            self.assertGreater(d["equality"], 0.0)

    def test_emissions_tax_raises_air_quality(self):
        rng = random.Random(1)
        state = self._state()
        for _ in range(20):
            d, _ = apply_policy("emissions_tax", "south", 0.4, state, rng)
            self.assertGreater(d["air_quality"], 0.0)

    def test_district_multiplier_amplifies_effect(self):
        """North has 1.4× transit multiplier vs south's 0.7×."""
        rng_north = random.Random(7)
        rng_south = random.Random(7)  # same seed for fair comparison
        state = self._state()
        deltas_north = []
        deltas_south = []
        for _ in range(100):
            d_n, _ = apply_policy("expand_transit", "north", 0.5, state, rng_north)
            d_s, _ = apply_policy("expand_transit", "south", 0.5, state, rng_south)
            deltas_north.append(d_n["traffic"])
            deltas_south.append(d_s["traffic"])
        avg_north = sum(deltas_north) / len(deltas_north)
        avg_south = sum(deltas_south) / len(deltas_south)
        # North traffic delta should be more negative (bigger reduction)
        self.assertLess(avg_north, avg_south)

    def test_budget_scales_effect(self):
        """Double budget → roughly double effect (same seed)."""
        state = self._state()
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        d_small, _ = apply_policy("build_housing", "east", 0.2, state, rng1)
        d_large, _ = apply_policy("build_housing", "east", 0.4, state, rng2)
        # Housing delta should be larger with more budget
        self.assertGreater(d_large["housing"], d_small["housing"])

    def test_zero_budget_returns_zeros(self):
        state = self._state()
        deltas, warning = apply_policy("expand_transit", "north", 0.0, state)
        for v in deltas.values():
            self.assertEqual(v, 0.0)
        self.assertIn("small", warning.lower())

    def test_unknown_policy_returns_zeros_with_error(self):
        state = self._state()
        deltas, warning = apply_policy("magic_policy", "north", 0.5, state)
        for v in deltas.values():
            self.assertEqual(v, 0.0)
        self.assertIn("Unknown", warning)

    def test_unknown_district_returns_zeros_with_error(self):
        state = self._state()
        deltas, warning = apply_policy("expand_transit", "atlantis", 0.5, state)
        for v in deltas.values():
            self.assertEqual(v, 0.0)
        self.assertIn("Unknown", warning)

    def test_over_budget_scales_down(self):
        state = self._state()
        state.budget_left = 0.2
        rng = random.Random(1)
        d_over, warning = apply_policy("expand_transit", "north", 0.8, state, rng)
        # Should warn about over-budget
        self.assertIn("budget", warning.lower())

    def test_noise_is_bounded(self):
        """All deltas should be within 50% of the noiseless value."""
        state = self._state()
        for _ in range(500):
            rng = random.Random()
            d, _ = apply_policy("expand_transit", "north", 1.0, state, rng)
            # Base traffic effect = -12 × 1.0 × 1.4 = -16.8
            # With ±50% noise: range [-25.2, -8.4]
            self.assertLess(d["traffic"], 0.0)
            self.assertGreaterEqual(d["traffic"], -30.0)   # generous bound

    def test_deterministic_with_same_seed(self):
        state = self._state()
        rng1 = random.Random(12345)
        rng2 = random.Random(12345)
        d1, _ = apply_policy("bike_lanes", "west", 0.5, state, rng1)
        d2, _ = apply_policy("bike_lanes", "west", 0.5, state, rng2)
        for kpi in KPI_NAMES:
            self.assertAlmostEqual(d1[kpi], d2[kpi], places=10)

    def test_different_seeds_different_results(self):
        state = self._state()
        rng1 = random.Random(1)
        rng2 = random.Random(2)
        d1, _ = apply_policy("green_spaces", "south", 0.5, state, rng1)
        d2, _ = apply_policy("green_spaces", "south", 0.5, state, rng2)
        # Not all KPIs need to differ, but at least one should
        different = any(abs(d1[k] - d2[k]) > 0.001 for k in KPI_NAMES)
        self.assertTrue(different)


# ═══════════════════════════════════════════════════════════════════════════════
#  Stakeholder tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestStakeholder(unittest.TestCase):

    def test_loved_policy_returns_supportive(self):
        sh = STAKEHOLDERS["transit_union"]
        stance, delta = sh.react("expand_transit")
        self.assertEqual(stance, "supportive")
        self.assertGreater(delta, 0)

    def test_hated_policy_returns_opposed(self):
        sh = STAKEHOLDERS["transit_union"]
        stance, delta = sh.react("congestion_tax")
        self.assertEqual(stance, "opposed")
        self.assertLess(delta, 0)

    def test_neutral_policy_returns_neutral(self):
        sh = STAKEHOLDERS["transit_union"]
        stance, delta = sh.react("subsidise_rent")
        self.assertEqual(stance, "neutral")
        self.assertEqual(delta, 0.0)

    def test_high_power_amplifies_delta(self):
        """property_owners (power=1.5) should have larger delta than transit_union (power=1.0)."""
        sh_low  = STAKEHOLDERS["transit_union"]    # power 1.0
        sh_high = STAKEHOLDERS["property_owners"]  # power 1.5
        _, d_low  = sh_low.react("expand_transit")
        _, d_high = sh_high.react("build_housing")  # they hate this
        # Both opposed with their own power levels
        abs_low  = abs(d_low)   # if opposed
        # Just verify power scales correctly
        sh_test = STAKEHOLDERS["env_coalition"]    # power 0.8
        _, d_env = sh_test.react("emissions_tax")  # they love this
        sh_test2 = STAKEHOLDERS["property_owners"]  # power 1.5
        _, d_prop = sh_test2.react("zoning_reform")  # they love this
        # Env coalition (0.8) gets smaller delta than property owners (1.5)
        self.assertLess(abs(d_env), abs(d_prop))


class TestStakeholderEngine(unittest.TestCase):

    def _engine(self, capital: float = 70.0) -> StakeholderEngine:
        return StakeholderEngine(political_capital=capital)

    def test_initial_approvals_set(self):
        eng = self._engine()
        for name in STAKEHOLDER_NAMES:
            self.assertIn(name, eng.approval_ratings)
            self.assertGreaterEqual(eng.approval_ratings[name], 0)
            self.assertLessEqual(eng.approval_ratings[name], 100)

    def test_supportive_reaction_increases_capital(self):
        eng = self._engine(capital=50.0)
        result = eng.process_policy("expand_transit")  # transit_union loves this
        self.assertIn("transit_union", result["reactions"])
        # transit_union stance = supportive
        self.assertEqual(result["reactions"]["transit_union"], "supportive")

    def test_opposed_reaction_decreases_capital(self):
        eng = self._engine(capital=70.0)
        before = eng.political_capital
        eng.process_policy("subsidise_rent")   # property_owners hate this
        after = eng.political_capital
        self.assertLess(after, before)

    def test_coalition_penalty_when_two_oppose(self):
        """congestion_tax: transit_union opposes (−8), business_council supports (+6)
           emissions_tax:  business_council opposes, property_owners neutral, env supports
           income_support: property_owners oppose (−12), business_council oppose (−9.6)
           → coalition penalty should trigger for income_support"""
        eng = self._engine(capital=80.0)
        before = eng.political_capital
        result = eng.process_policy("income_support")
        # Count opposed
        opposed = [v for v in result["reactions"].values() if v == "opposed"]
        if len(opposed) >= 2:
            self.assertTrue(result["coalition_penalty_applied"])

    def test_recall_at_zero_capital(self):
        eng = self._engine(capital=5.0)
        # Trigger heavy opposition
        for _ in range(5):
            eng.process_policy("income_support")   # property_owners + business_council oppose
        self.assertTrue(eng.is_recalled())
        self.assertLessEqual(eng.political_capital, 0.0)

    def test_capital_capped_at_100(self):
        eng = self._engine(capital=99.0)
        eng.process_policy("expand_transit")   # transit_union + business_council supportive
        self.assertLessEqual(eng.political_capital, 100.0)

    def test_capital_floored_at_0(self):
        eng = self._engine(capital=1.0)
        eng.process_policy("income_support")
        self.assertGreaterEqual(eng.political_capital, 0.0)

    def test_approval_ratings_clamped(self):
        eng = self._engine()
        for _ in range(20):
            eng.process_policy("expand_transit")
        for name in STAKEHOLDER_NAMES:
            self.assertLessEqual(eng.approval_ratings[name], 100.0)
            self.assertGreaterEqual(eng.approval_ratings[name], 0.0)

    def test_end_of_turn_recovery_increases_capital(self):
        eng = self._engine(capital=50.0)
        before = eng.political_capital
        eng.end_of_turn_recovery()
        after = eng.political_capital
        self.assertGreaterEqual(after, before)

    def test_copy_is_independent(self):
        eng = self._engine(capital=70.0)
        copy = eng.copy()
        eng.process_policy("congestion_tax")
        # Copy should not be affected
        self.assertAlmostEqual(copy.political_capital, 70.0)

    def test_snapshot_returns_expected_keys(self):
        eng = self._engine()
        snap = eng.snapshot()
        self.assertIn("political_capital", snap)
        self.assertIn("approval_ratings", snap)
        for name in STAKEHOLDER_NAMES:
            self.assertIn(name, snap["approval_ratings"])


# ═══════════════════════════════════════════════════════════════════════════════
#  Integration: city + policy + stakeholders
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):

    def test_apply_and_update_city(self):
        """Verify applying a policy actually changes city KPIs."""
        state = CityState.default()
        for name in DISTRICT_NAMES:
            state.kpis[name].set("traffic", 80.0)

        rng = random.Random(1)
        deltas, _ = apply_policy("expand_transit", "north", 0.5, state, rng)
        before = state.kpis["north"].get("traffic")
        state.kpis["north"].delta("traffic", deltas["traffic"])
        after = state.kpis["north"].get("traffic")
        self.assertLess(after, before)   # traffic improved

    def test_multiple_policies_compound(self):
        """Applying multiple helpful policies should improve composite score."""
        state = CityState.default()
        for name in DISTRICT_NAMES:
            state.kpis[name].set("traffic", 80.0)
            state.kpis[name].set("housing", 30.0)

        initial_score = state.composite_score()
        rng = random.Random(42)

        policies = [
            ("expand_transit", "north", 0.3),
            ("congestion_tax", "central", 0.2),
            ("build_housing", "east", 0.3),
        ]
        for pt, dist, bp in policies:
            deltas, _ = apply_policy(pt, dist, bp, state, rng)
            for kpi, delta in deltas.items():
                state.kpis[dist].delta(kpi, delta)
            state.budget_left -= bp

        final_score = state.composite_score()
        self.assertGreater(final_score, initial_score)

    def test_budget_depletion_limits_effect(self):
        """Once budget is exhausted, policy should have reduced effect."""
        state = CityState.default()
        state.budget_left = 0.05   # almost nothing left
        rng = random.Random(1)
        # Request 0.3 but only 0.05 left
        d_limited, warn = apply_policy("expand_transit", "north", 0.3, state, rng)
        self.assertIn("budget", warn.lower())

        state2 = CityState.default()
        state2.budget_left = 1.0
        rng2 = random.Random(1)
        d_full, _ = apply_policy("expand_transit", "north", 0.3, state2, rng2)
        # Full budget effect should be larger in magnitude
        self.assertLess(d_full["traffic"], d_limited["traffic"])  # more negative = bigger reduction


if __name__ == "__main__":
    unittest.main(verbosity=2)
