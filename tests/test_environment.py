"""
tests/test_environment.py
──────────────────────────
Deep tests for PolicyEnvironment, task graders, and the full episode loop.
Uses a pydantic stub so it runs without installing pydantic.
"""
import sys
import os
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Stub pydantic so models.py imports work ──────────────────────────────────
pydantic_mod = types.ModuleType("pydantic")

class _BaseModel:
    def __init__(self, **kw):
        for k, v in kw.items():
            object.__setattr__(self, k, v)
    def __setattr__(self, k, v):
        object.__setattr__(self, k, v)
    def model_dump(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

def _Field(default=None, **kw):
    return default

pydantic_mod.BaseModel = _BaseModel
pydantic_mod.Field     = _Field
sys.modules["pydantic"] = pydantic_mod
# ─────────────────────────────────────────────────────────────────────────────

from core.city_model import CityState, DISTRICT_NAMES, KPI_NAMES
from tasks import TASKS, _grade_task1, _grade_task2, _grade_task3, _task1_city, _task2_city, _task3_city
from environment import PolicyEnvironment
from models import PolicyAction


# ═══════════════════════════════════════════════════════════════════════════════
#  Task grader unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTask1Grader(unittest.TestCase):
    """Grader is delta-based: measures traffic reduction from the task-1 starting value (~77)."""

    def _city_reduced(self, reduction: float, budget_used: float = 0.5) -> CityState:
        """Build a city with traffic reduced by `reduction` pts from task1 start."""
        state = _task1_city()
        start_traffic = state.citywide_kpis()["traffic"]   # ~77
        target = start_traffic - reduction
        for name in DISTRICT_NAMES:
            state.kpis[name].set("traffic", target)
        state.budget_left = 1.0 - budget_used
        return state

    def test_perfect_score_big_reduction_on_budget(self):
        state = self._city_reduced(reduction=5.0, budget_used=0.60)
        score = _grade_task1(state)
        self.assertEqual(score, 1.0)

    def test_score_0_8_big_reduction_over_budget(self):
        state = self._city_reduced(reduction=5.0, budget_used=0.80)
        score = _grade_task1(state)
        self.assertEqual(score, 0.8)

    def test_score_0_5_moderate_reduction(self):
        state = self._city_reduced(reduction=2.5, budget_used=0.5)
        score = _grade_task1(state)
        self.assertEqual(score, 0.5)

    def test_score_0_2_small_reduction(self):
        state = self._city_reduced(reduction=1.0, budget_used=0.4)
        score = _grade_task1(state)
        self.assertEqual(score, 0.2)

    def test_score_0_for_no_improvement(self):
        state = _task1_city()   # no reduction applied
        score = _grade_task1(state)
        self.assertEqual(score, 0.0)

    def test_score_in_range(self):
        for reduction in [0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
            state = self._city_reduced(reduction=float(reduction))
            score = _grade_task1(state)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestTask2Grader(unittest.TestCase):

    def test_zero_for_no_improvement(self):
        state = _task2_city()   # starting values
        score = _grade_task2(state)
        self.assertAlmostEqual(score, 0.0, places=3)

    def test_partial_for_partial_improvement(self):
        state = _task2_city()
        # Apply 2.5 points gain to each district per KPI.
        # citywide gain = 2.5, target = 5.0 -> ratio = 0.5 -> score ~ 0.5
        for name in DISTRICT_NAMES:
            state.kpis[name].set("housing",      state.kpis[name].get("housing")      + 2.5)
            state.kpis[name].set("equality",     state.kpis[name].get("equality")     + 2.5)
            state.kpis[name].set("satisfaction", state.kpis[name].get("satisfaction") + 2.5)
        score = _grade_task2(state)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.8)

    def test_perfect_score_with_bonus(self):
        state = _task2_city()
        for name in DISTRICT_NAMES:
            state.kpis[name].set("housing",      state.kpis[name].get("housing")      + 25.0)
            state.kpis[name].set("equality",     state.kpis[name].get("equality")     + 25.0)
            state.kpis[name].set("satisfaction", state.kpis[name].get("satisfaction") + 25.0)
        state.budget_left = 0.25
        score = _grade_task2(state)
        self.assertAlmostEqual(score, 1.0, delta=0.05)

    def test_improving_only_one_kpi_gives_partial(self):
        state = _task2_city()
        for name in DISTRICT_NAMES:
            state.kpis[name].set("housing", state.kpis[name].get("housing") + 20.0)
        score = _grade_task2(state)
        # 1/3 of KPIs fully improved → ~0.33
        self.assertAlmostEqual(score, 0.33, delta=0.05)

    def test_score_always_in_range(self):
        import random
        rng = random.Random(42)
        for _ in range(20):
            state = _task2_city()
            for name in DISTRICT_NAMES:
                for kpi in ["housing", "equality", "satisfaction"]:
                    state.kpis[name].set(kpi, rng.uniform(0, 100))
            score = _grade_task2(state)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestTask3Grader(unittest.TestCase):

    def test_recalled_agent_max_score(self):
        """Agent that gets recalled scores ≤ 0.30 (no KPI improvement assumed)."""
        state = _task3_city()
        state.political_capital = 0.0   # recalled
        score = _grade_task3(state)
        # Without KPI improvement, kpi_component≈0, survived=0, capital=0 → score≈0
        self.assertAlmostEqual(score, 0.0, delta=0.05)

    def test_survived_but_no_kpi_gain(self):
        state = _task3_city()
        state.political_capital = 60.0  # survived
        score = _grade_task3(state)
        # survived=1 → 0.30, capital_frac=0.6 → 0.12, kpi=0 → 0.0
        self.assertAlmostEqual(score, 0.42, delta=0.05)

    def test_full_improvement_gives_near_one(self):
        state = _task3_city()
        # Dramatically improve all KPIs
        for name in DISTRICT_NAMES:
            state.kpis[name].set("traffic", 30.0)
            state.kpis[name].set("housing", 70.0)
            state.kpis[name].set("equality", 70.0)
            state.kpis[name].set("air_quality", 80.0)
            state.kpis[name].set("satisfaction", 75.0)
        state.political_capital = 80.0
        score = _grade_task3(state)
        self.assertGreater(score, 0.8)

    def test_score_always_in_range(self):
        import random
        rng = random.Random(99)
        for _ in range(20):
            state = _task3_city()
            for name in DISTRICT_NAMES:
                for kpi in KPI_NAMES:
                    state.kpis[name].set(kpi, rng.uniform(0, 100))
            state.political_capital = rng.uniform(0, 100)
            score = _grade_task3(state)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  PolicyEnvironment tests
# ═══════════════════════════════════════════════════════════════════════════════

def _action(**kw) -> PolicyAction:
    defaults = dict(
        action_type="pass_turn", policy_type=None, district=None,
        budget_pct=None, stakeholder=None, metadata={}
    )
    defaults.update(kw)
    return PolicyAction(**defaults)


class TestEnvironmentReset(unittest.TestCase):

    def test_reset_returns_observation(self):
        env = PolicyEnvironment(seed=42)
        obs = env.reset("task_1_decongest")
        self.assertEqual(obs.task_id, "task_1_decongest")
        self.assertFalse(obs.done)
        self.assertEqual(obs.turn, 0)
        self.assertAlmostEqual(obs.budget_left, 1.0)

    def test_reset_all_tasks(self):
        env = PolicyEnvironment(seed=1)
        for tid in TASKS:
            obs = env.reset(tid)
            self.assertEqual(obs.task_id, tid)
            self.assertFalse(obs.done)

    def test_reset_unknown_task_raises(self):
        env = PolicyEnvironment()
        with self.assertRaises(ValueError):
            env.reset("task_999_nonexistent")

    def test_reset_clears_previous_episode(self):
        env = PolicyEnvironment(seed=5)
        env.reset("task_1_decongest")
        # Take a step
        obs1, _, _, _ = env.step(_action(
            action_type="propose_policy",
            policy_type="expand_transit",
            district="north",
            budget_pct=0.3,
        ))
        self.assertEqual(obs1.turn, 1)
        # Reset again — should be back to turn 0
        obs2 = env.reset("task_1_decongest")
        self.assertEqual(obs2.turn, 0)
        self.assertAlmostEqual(obs2.budget_left, 1.0)
        self.assertFalse(obs2.done)

    def test_reset_with_seed_is_deterministic(self):
        env1 = PolicyEnvironment(seed=42)
        env2 = PolicyEnvironment(seed=42)
        obs1 = env1.reset("task_1_decongest")
        obs2 = env2.reset("task_1_decongest")
        self.assertAlmostEqual(obs1.composite_score, obs2.composite_score)
        self.assertEqual(obs1.citywide_kpis, obs2.citywide_kpis)

    def test_cumulative_reward_resets_to_zero(self):
        env = PolicyEnvironment(seed=1)
        env.reset("task_1_decongest")
        env.step(_action(action_type="pass_turn"))
        env.step(_action(action_type="pass_turn"))
        obs_restart = env.reset("task_1_decongest")
        self.assertEqual(obs_restart.cumulative_reward, 0.0)


class TestEnvironmentStep(unittest.TestCase):

    def _fresh_env(self, task="task_1_decongest", seed=42):
        env = PolicyEnvironment(seed=seed)
        env.reset(task)
        return env

    def test_propose_policy_changes_kpis(self):
        env = self._fresh_env()
        obs_before = env.reset("task_1_decongest")
        traffic_before = obs_before.citywide_kpis["traffic"]
        obs, reward, done, info = env.step(_action(
            action_type="propose_policy",
            policy_type="expand_transit",
            district="north",
            budget_pct=0.5,
        ))
        traffic_after = obs.citywide_kpis["traffic"]
        self.assertLess(traffic_after, traffic_before)

    def test_propose_policy_reduces_budget(self):
        env = self._fresh_env()
        obs, _, _, _ = env.step(_action(
            action_type="propose_policy",
            policy_type="expand_transit",
            district="north",
            budget_pct=0.3,
        ))
        self.assertAlmostEqual(obs.budget_left, 0.7, delta=0.01)

    def test_step_increments_turn(self):
        env = self._fresh_env()
        for i in range(3):
            obs, _, _, _ = env.step(_action(action_type="pass_turn"))
            self.assertEqual(obs.turn, i + 1)

    def test_done_after_max_turns(self):
        env = self._fresh_env("task_1_decongest")
        done = False
        for _ in range(4):   # task 1 has 4 turns
            obs, reward, done, info = env.step(_action(action_type="pass_turn"))
        self.assertTrue(done)
        self.assertTrue(obs.done)

    def test_grader_score_in_info_at_end(self):
        env = self._fresh_env("task_1_decongest")
        info = {}
        for _ in range(4):
            _, _, _, info = env.step(_action(action_type="pass_turn"))
        self.assertIn("grader_score", info)
        score = info["grader_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_step_after_done_raises(self):
        env = self._fresh_env("task_1_decongest")
        for _ in range(4):
            env.step(_action(action_type="pass_turn"))
        with self.assertRaises(RuntimeError):
            env.step(_action(action_type="pass_turn"))

    def test_invalid_action_type_returns_error(self):
        env = self._fresh_env()
        obs, reward, done, info = env.step(_action(action_type="teleport"))
        self.assertIsNotNone(obs.error_message)
        self.assertIn("Unknown", obs.error_message)
        self.assertLess(reward, 0)

    def test_propose_without_policy_type_returns_error(self):
        env = self._fresh_env()
        obs, reward, done, info = env.step(_action(
            action_type="propose_policy",
            district="north",
            budget_pct=0.3,
        ))
        self.assertIsNotNone(obs.error_message)
        self.assertLess(reward, 0)

    def test_propose_without_district_returns_error(self):
        env = self._fresh_env()
        obs, reward, done, info = env.step(_action(
            action_type="propose_policy",
            policy_type="expand_transit",
            budget_pct=0.3,
        ))
        self.assertIsNotNone(obs.error_message)

    def test_propose_without_budget_returns_error(self):
        env = self._fresh_env()
        obs, reward, done, info = env.step(_action(
            action_type="propose_policy",
            policy_type="expand_transit",
            district="north",
        ))
        self.assertIsNotNone(obs.error_message)

    def test_propose_unknown_policy_returns_error(self):
        env = self._fresh_env()
        obs, _, _, _ = env.step(_action(
            action_type="propose_policy",
            policy_type="magic_beam",
            district="north",
            budget_pct=0.3,
        ))
        self.assertIsNotNone(obs.error_message)

    def test_propose_unknown_district_returns_error(self):
        env = self._fresh_env()
        obs, _, _, _ = env.step(_action(
            action_type="propose_policy",
            policy_type="expand_transit",
            district="narnia",
            budget_pct=0.3,
        ))
        self.assertIsNotNone(obs.error_message)

    def test_investigate_returns_info(self):
        env = self._fresh_env()
        obs, reward, done, info = env.step(_action(
            action_type="investigate",
            stakeholder="transit_union",
        ))
        self.assertIsNotNone(obs.investigation_result)
        self.assertEqual(obs.investigation_result["stakeholder"], "transit_union")
        self.assertIn("loved_policies", obs.investigation_result)
        self.assertIn("hated_policies", obs.investigation_result)
        self.assertFalse(done)

    def test_investigate_unknown_stakeholder_returns_error(self):
        env = self._fresh_env()
        obs, reward, _, _ = env.step(_action(
            action_type="investigate",
            stakeholder="big_pharma",
        ))
        self.assertIsNotNone(obs.error_message)
        self.assertLess(reward, 0)

    def test_pass_turn_has_penalty(self):
        env = self._fresh_env()
        obs, reward, done, info = env.step(_action(action_type="pass_turn"))
        # STEP_PENALTY (-0.02) + PASS_PENALTY (-0.05) = -0.07
        self.assertAlmostEqual(reward, -0.07, places=4)

    def test_reward_is_always_finite(self):
        import math
        env = self._fresh_env()
        for _ in range(4):
            _, reward, _, _ = env.step(_action(action_type="pass_turn"))
            self.assertTrue(math.isfinite(reward))

    def test_cumulative_reward_accumulates(self):
        env = self._fresh_env()
        total = 0.0
        for _ in range(3):
            obs, reward, _, _ = env.step(_action(action_type="pass_turn"))
            total += reward
        self.assertAlmostEqual(obs.cumulative_reward, total, places=4)

    def test_kpi_delta_returned_on_propose(self):
        env = self._fresh_env()
        obs, _, _, _ = env.step(_action(
            action_type="propose_policy",
            policy_type="expand_transit",
            district="north",
            budget_pct=0.3,
        ))
        self.assertIsNotNone(obs.kpi_delta)
        self.assertEqual(set(obs.kpi_delta.keys()), set(KPI_NAMES))

    def test_stakeholder_reactions_returned_on_propose(self):
        env = self._fresh_env()
        obs, _, _, _ = env.step(_action(
            action_type="propose_policy",
            policy_type="expand_transit",
            district="north",
            budget_pct=0.3,
        ))
        self.assertIsNotNone(obs.stakeholder_reactions)
        from core.stakeholders import STAKEHOLDER_NAMES
        for name in STAKEHOLDER_NAMES:
            self.assertIn(name, obs.stakeholder_reactions)
            self.assertIn(obs.stakeholder_reactions[name], ["supportive", "neutral", "opposed"])

    def test_district_kpis_in_observation(self):
        env = self._fresh_env()
        obs, _, _, _ = env.step(_action(action_type="pass_turn"))
        for dist in DISTRICT_NAMES:
            self.assertIn(dist, obs.district_kpis)
            for kpi in KPI_NAMES:
                self.assertIn(kpi, obs.district_kpis[dist])

    def test_recall_triggers_terminal(self):
        """Force a recall by repeatedly proposing policies that anger stakeholders."""
        env = PolicyEnvironment(seed=42)
        env.reset("task_3_gauntlet")   # starts with capital=50 (fragile)
        done = False
        max_steps = 20
        for _ in range(max_steps):
            obs, _, done, info = env.step(_action(
                action_type="propose_policy",
                policy_type="income_support",   # property_owners + business_council oppose
                district="north",
                budget_pct=0.1,
            ))
            if done:
                break
        self.assertTrue(done)
        self.assertIn(obs.done_reason, ["recalled", "max_turns"])


class TestEnvironmentState(unittest.TestCase):

    def test_state_reflects_current_episode(self):
        env = PolicyEnvironment(seed=1)
        env.reset("task_2_equity")
        env.step(_action(action_type="pass_turn"))
        env.step(_action(action_type="pass_turn"))
        state = env.state()
        self.assertEqual(state.task_id, "task_2_equity")
        self.assertEqual(state.turn, 2)
        self.assertFalse(state.done)

    def test_state_shows_done_after_episode_ends(self):
        env = PolicyEnvironment(seed=1)
        env.reset("task_1_decongest")
        for _ in range(4):
            env.step(_action(action_type="pass_turn"))
        state = env.state()
        self.assertTrue(state.done)

    def test_state_turns_left_decrements(self):
        env = PolicyEnvironment(seed=1)
        env.reset("task_1_decongest")
        for i in range(3):
            env.step(_action(action_type="pass_turn"))
        state = env.state()
        self.assertEqual(state.turns_left, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Full episode scenarios
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullEpisodeScenarios(unittest.TestCase):

    def test_task1_good_strategy_scores_higher_than_bad(self):
        """
        Good strategy (expand_transit + congestion_tax) should score higher
        than bad strategy (green_spaces, which doesn't target traffic).
        """
        def run_episode(policies):
            env = PolicyEnvironment(seed=42)
            env.reset("task_1_decongest")
            final_info = {}
            for pt, dist, bp in policies:
                _, _, done, final_info = env.step(_action(
                    action_type="propose_policy",
                    policy_type=pt, district=dist, budget_pct=bp,
                ))
                if done:
                    break
            # Fill remaining turns
            while not done:
                _, _, done, final_info = env.step(_action(action_type="pass_turn"))
            return final_info.get("grader_score", 0.0)

        good_score = run_episode([
            ("expand_transit",  "north",   0.25),
            ("congestion_tax",  "central", 0.20),
            ("expand_transit",  "west",    0.20),
        ])
        bad_score = run_episode([
            ("green_spaces", "south", 0.1),
            ("green_spaces", "east",  0.1),
        ])
        self.assertGreater(good_score, bad_score)

    def test_task2_single_policy_insufficient(self):
        """No single policy can solve all 3 equity KPIs in task 2."""
        env = PolicyEnvironment(seed=42)
        env.reset("task_2_equity")
        for _ in range(6):
            env.step(_action(
                action_type="propose_policy",
                policy_type="income_support",  # only fixes equality
                district="north",
                budget_pct=0.15,
            ))
        state = env.state()
        score_single = TASKS["task_2_equity"].grader(env._city)
        # Score should be < 0.7 since housing isn't sufficiently improved
        self.assertLess(score_single, 0.7)

    def test_task2_diverse_strategy_scores_well(self):
        """Using housing + income + rent policies together should score well."""
        env = PolicyEnvironment(seed=42)
        env.reset("task_2_equity")
        strategy = [
            ("build_housing",  "east",  0.20),
            ("income_support", "north", 0.20),
            ("subsidise_rent", "west",  0.20),
            ("build_housing",  "south", 0.15),
            ("income_support", "central", 0.10),
        ]
        done = False
        for pt, dist, bp in strategy:
            _, _, done, _ = env.step(_action(
                action_type="propose_policy",
                policy_type=pt, district=dist, budget_pct=bp,
            ))
            if done: break
        while not done:
            _, _, done, _ = env.step(_action(action_type="pass_turn"))

        score = TASKS["task_2_equity"].grader(env._city)
        self.assertGreater(score, 0.25)

    def test_investigate_does_not_consume_budget(self):
        env = PolicyEnvironment(seed=5)
        obs_before = env.reset("task_3_gauntlet")
        budget_before = obs_before.budget_left
        for sh in ["transit_union", "property_owners", "env_coalition", "business_council"]:
            env.step(_action(action_type="investigate", stakeholder=sh))
        state = env.state()
        self.assertAlmostEqual(state.budget_left, budget_before, delta=0.001)

    def test_grader_score_is_deterministic(self):
        """Same seed + same actions → same grader score."""
        def run(seed):
            env = PolicyEnvironment(seed=seed)
            env.reset("task_1_decongest")
            actions = [
                ("expand_transit", "north", 0.3),
                ("congestion_tax", "central", 0.2),
            ]
            for pt, dist, bp in actions:
                env.step(_action(action_type="propose_policy", policy_type=pt, district=dist, budget_pct=bp))
            done = False
            final = {}
            while not done:
                _, _, done, final = env.step(_action(action_type="pass_turn"))
            return final.get("grader_score", 0.0)

        s1 = run(42)
        s2 = run(42)
        self.assertAlmostEqual(s1, s2, places=10)

    def test_all_tasks_complete_without_error(self):
        """All 3 tasks should complete cleanly with only pass_turn actions."""
        env = PolicyEnvironment(seed=0)
        for tid, task in TASKS.items():
            env.reset(tid)
            done = False
            for _ in range(task.max_turns + 2):
                if done:
                    break
                _, _, done, _ = env.step(_action(action_type="pass_turn"))
            self.assertTrue(done, f"Task {tid} never reached done state")
            state = env.state()
            self.assertTrue(state.done)

    def test_observation_has_all_required_fields(self):
        """Every observation must have the full field set."""
        required = [
            "task_id", "task_description", "citywide_kpis", "district_kpis",
            "composite_score", "budget_left", "political_capital",
            "stakeholder_approvals", "turn", "turns_left",
            "reward", "cumulative_reward", "done",
        ]
        env = PolicyEnvironment(seed=1)
        obs = env.reset("task_1_decongest")
        for field in required:
            self.assertTrue(hasattr(obs, field), f"Missing field: {field}")
            self.assertIsNotNone(getattr(obs, field), f"Field is None: {field}")

    def test_composite_score_improves_with_good_policy(self):
        env = PolicyEnvironment(seed=42)
        obs = env.reset("task_3_gauntlet")
        score_before = obs.composite_score
        # Apply strong policies
        for pt, dist, bp in [
            ("expand_transit", "north",   0.25),
            ("build_housing",  "east",    0.25),
            ("income_support", "north",   0.20),
            ("bike_lanes",     "central", 0.15),
        ]:
            obs, _, done, _ = env.step(_action(
                action_type="propose_policy",
                policy_type=pt, district=dist, budget_pct=bp,
            ))
            if done:
                break
        self.assertGreater(obs.composite_score, score_before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
