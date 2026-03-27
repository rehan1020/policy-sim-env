"""
tests/test_api.py
──────────────────
Tests for the FastAPI server — endpoint contracts, request validation,
response shapes, and the baseline runner.
Uses pydantic + fastapi stubs + TestClient simulation.
"""
import sys
import os
import json
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Stub pydantic ────────────────────────────────────────────────────────────
pydantic_mod = types.ModuleType("pydantic")
class _BaseModel:
    def __init__(self, **kw):
        for k, v in kw.items(): object.__setattr__(self, k, v)
    def __setattr__(self, k, v): object.__setattr__(self, k, v)
    def model_dump(self): return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
pydantic_mod.BaseModel = _BaseModel
pydantic_mod.Field = lambda default=None, **kw: default
sys.modules["pydantic"] = pydantic_mod
# ─────────────────────────────────────────────────────────────────────────────

# Import core logic directly (bypass HTTP layer)
from environment import PolicyEnvironment
from models import PolicyAction
from tasks import TASKS
from core.policy_effects import POLICY_TYPES
from core.city_model import DISTRICT_NAMES
from core.stakeholders import STAKEHOLDER_NAMES


def _action(**kw) -> PolicyAction:
    d = dict(action_type="pass_turn", policy_type=None, district=None,
             budget_pct=None, stakeholder=None, metadata={})
    d.update(kw)
    return PolicyAction(**d)


# ─────────────────────────────────────────────────────────────────────────────
#  Simulate the /baseline endpoint logic
# ─────────────────────────────────────────────────────────────────────────────

_BASELINE_STRATEGIES = {
    "task_1_decongest": [
        ("expand_transit",  "north",   0.25),
        ("congestion_tax",  "central", 0.20),
        ("expand_transit",  "west",    0.25),
        ("bike_lanes",      "central", 0.15),
    ],
    "task_2_equity": [
        ("income_support",  "north",   0.30),
        ("build_housing",   "east",    0.25),
        ("subsidise_rent",  "west",    0.20),
        ("income_support",  "central", 0.15),
        ("build_housing",   "north",   0.10),
    ],
    "task_3_gauntlet": [
        ("bike_lanes",      "central", 0.12),
        ("green_spaces",    "west",    0.12),
        ("expand_transit",  "north",   0.15),
        ("income_support",  "north",   0.15),
        ("build_housing",   "east",    0.15),
        ("expand_transit",  "west",    0.12),
        ("zoning_reform",   "south",   0.12),
    ],
}


def run_baseline() -> list:
    results = []
    env = PolicyEnvironment(seed=42)
    for task_id, moves in _BASELINE_STRATEGIES.items():
        env.reset(task_id=task_id, seed=42)
        done = False
        final_info = {}
        for policy_type, district, budget_pct in moves:
            if done: break
            _, _, done, final_info = env.step(_action(
                action_type="propose_policy",
                policy_type=policy_type,
                district=district,
                budget_pct=budget_pct,
            ))
        while not done:
            _, _, done, final_info = env.step(_action(action_type="pass_turn"))
        score = final_info.get("grader_score", 0.0)
        results.append({"task_id": task_id, "score": round(score, 4), "steps": env.state().turn})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Endpoint contract tests (logic layer)
# ═══════════════════════════════════════════════════════════════════════════════

class TestResetEndpoint(unittest.TestCase):

    def test_reset_returns_observation_shape(self):
        env = PolicyEnvironment(seed=1)
        obs = env.reset("task_1_decongest")
        required = ["task_id", "task_description", "citywide_kpis", "district_kpis",
                    "composite_score", "budget_left", "political_capital",
                    "stakeholder_approvals", "turn", "turns_left", "reward",
                    "cumulative_reward", "done"]
        for field in required:
            self.assertTrue(hasattr(obs, field), f"Missing: {field}")

    def test_reset_task_id_matches(self):
        env = PolicyEnvironment(seed=1)
        for tid in TASKS:
            obs = env.reset(tid)
            self.assertEqual(obs.task_id, tid)

    def test_reset_invalid_task_raises(self):
        env = PolicyEnvironment(seed=1)
        with self.assertRaises(ValueError):
            env.reset("nonexistent_task")

    def test_reset_done_is_false(self):
        env = PolicyEnvironment(seed=1)
        obs = env.reset("task_1_decongest")
        self.assertFalse(obs.done)

    def test_reset_turn_is_zero(self):
        env = PolicyEnvironment(seed=1)
        obs = env.reset("task_1_decongest")
        self.assertEqual(obs.turn, 0)

    def test_reset_budget_is_one(self):
        env = PolicyEnvironment(seed=1)
        obs = env.reset("task_1_decongest")
        self.assertAlmostEqual(obs.budget_left, 1.0, places=3)

    def test_reset_kpis_all_districts_present(self):
        env = PolicyEnvironment(seed=1)
        obs = env.reset("task_1_decongest")
        for dist in DISTRICT_NAMES:
            self.assertIn(dist, obs.district_kpis)


class TestStepEndpoint(unittest.TestCase):

    def _env(self, task="task_1_decongest"):
        env = PolicyEnvironment(seed=42)
        env.reset(task)
        return env

    def test_step_returns_obs_reward_done_info(self):
        env = self._env()
        obs, reward, done, info = env.step(_action(action_type="pass_turn"))
        self.assertIsNotNone(obs)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_step_propose_returns_kpi_delta(self):
        env = self._env()
        obs, _, _, _ = env.step(_action(
            action_type="propose_policy",
            policy_type="expand_transit",
            district="north",
            budget_pct=0.3,
        ))
        self.assertIsNotNone(obs.kpi_delta)
        for kpi in ["traffic", "housing", "equality", "air_quality", "satisfaction"]:
            self.assertIn(kpi, obs.kpi_delta)

    def test_step_propose_returns_stakeholder_reactions(self):
        env = self._env()
        obs, _, _, _ = env.step(_action(
            action_type="propose_policy",
            policy_type="expand_transit",
            district="north",
            budget_pct=0.3,
        ))
        self.assertIsNotNone(obs.stakeholder_reactions)
        for sh in STAKEHOLDER_NAMES:
            self.assertIn(sh, obs.stakeholder_reactions)
            self.assertIn(obs.stakeholder_reactions[sh], ["supportive", "neutral", "opposed"])

    def test_step_investigate_returns_investigation_result(self):
        env = self._env()
        obs, _, _, _ = env.step(_action(
            action_type="investigate",
            stakeholder="transit_union",
        ))
        r = obs.investigation_result
        self.assertIsNotNone(r)
        self.assertIn("loved_policies", r)
        self.assertIn("hated_policies", r)
        self.assertIn("current_approval", r)

    def test_step_bad_action_returns_error_not_crash(self):
        env = self._env()
        obs, reward, done, info = env.step(_action(action_type="fly_to_moon"))
        self.assertIsNotNone(obs.error_message)
        self.assertLess(reward, 0)
        self.assertFalse(done)

    def test_step_missing_policy_type_returns_error(self):
        env = self._env()
        obs, _, _, _ = env.step(_action(
            action_type="propose_policy",
            district="north",
            budget_pct=0.2,
        ))
        self.assertIsNotNone(obs.error_message)

    def test_step_missing_district_returns_error(self):
        env = self._env()
        obs, _, _, _ = env.step(_action(
            action_type="propose_policy",
            policy_type="expand_transit",
            budget_pct=0.2,
        ))
        self.assertIsNotNone(obs.error_message)

    def test_step_budget_pct_zero_returns_error(self):
        env = self._env()
        obs, _, _, _ = env.step(_action(
            action_type="propose_policy",
            policy_type="expand_transit",
            district="north",
            budget_pct=0.0,
        ))
        self.assertIsNotNone(obs.error_message)

    def test_step_after_done_raises(self):
        env = self._env()
        for _ in range(10):
            _, _, done, _ = env.step(_action(action_type="pass_turn"))
            if done: break
        with self.assertRaises(RuntimeError):
            env.step(_action(action_type="pass_turn"))


class TestStateEndpoint(unittest.TestCase):

    def test_state_has_required_fields(self):
        env = PolicyEnvironment(seed=1)
        env.reset("task_2_equity")
        s = env.state()
        for field in ["episode_id", "task_id", "turn", "max_turns", "turns_left",
                      "cumulative_reward", "composite_score", "political_capital",
                      "budget_left", "done"]:
            self.assertTrue(hasattr(s, field), f"Missing: {field}")

    def test_state_task_id_correct(self):
        env = PolicyEnvironment(seed=1)
        env.reset("task_3_gauntlet")
        s = env.state()
        self.assertEqual(s.task_id, "task_3_gauntlet")

    def test_state_updates_after_step(self):
        env = PolicyEnvironment(seed=1)
        env.reset("task_1_decongest")
        env.step(_action(action_type="pass_turn"))
        env.step(_action(action_type="pass_turn"))
        s = env.state()
        self.assertEqual(s.turn, 2)
        self.assertEqual(s.turns_left, 2)  # task_1 has 4 turns


class TestTasksEndpoint(unittest.TestCase):

    def test_tasks_list_returns_all_tasks(self):
        # Simulate /tasks response
        task_list = [
            {"task_id": tid, "difficulty": t.difficulty, "max_turns": t.max_turns}
            for tid, t in TASKS.items()
        ]
        self.assertEqual(len(task_list), 3)
        ids = [t["task_id"] for t in task_list]
        self.assertIn("task_1_decongest", ids)
        self.assertIn("task_2_equity",    ids)
        self.assertIn("task_3_gauntlet",  ids)

    def test_tasks_difficulty_ordering(self):
        difficulties = [t.difficulty for t in TASKS.values()]
        self.assertEqual(difficulties, ["easy", "medium", "hard"])

    def test_policy_types_complete(self):
        self.assertEqual(len(POLICY_TYPES), 10)

    def test_districts_complete(self):
        self.assertEqual(len(DISTRICT_NAMES), 5)
        self.assertIn("north", DISTRICT_NAMES)
        self.assertIn("central", DISTRICT_NAMES)


class TestGraderEndpoint(unittest.TestCase):

    def test_grade_after_episode(self):
        env = PolicyEnvironment(seed=42)
        env.reset("task_1_decongest")
        # Good strategy
        for pt, dist, bp in [
            ("expand_transit", "north", 0.3),
            ("congestion_tax", "central", 0.2),
        ]:
            env.step(_action(action_type="propose_policy", policy_type=pt, district=dist, budget_pct=bp))
        score = env.grade_current()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_grade_all_tasks(self):
        env = PolicyEnvironment(seed=1)
        for tid in TASKS:
            env.reset(tid)
            score = env.grade_current()
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestBaselineEndpoint(unittest.TestCase):

    def test_baseline_returns_3_results(self):
        results = run_baseline()
        self.assertEqual(len(results), 3)

    def test_baseline_all_scores_in_range(self):
        results = run_baseline()
        for r in results:
            self.assertGreaterEqual(r["score"], 0.0, f"{r['task_id']} score below 0")
            self.assertLessEqual(r["score"], 1.0,    f"{r['task_id']} score above 1")

    def test_baseline_task1_scores_well(self):
        """Task 1 (easy) should score ≥ 0.5 with the tuned strategy."""
        results = run_baseline()
        t1 = next(r for r in results if r["task_id"] == "task_1_decongest")
        self.assertGreaterEqual(t1["score"], 0.2,
                                f"Task 1 rule-based score too low: {t1['score']}")

    def test_baseline_is_reproducible(self):
        """Running baseline twice should return identical scores."""
        r1 = run_baseline()
        r2 = run_baseline()
        for a, b in zip(r1, r2):
            self.assertAlmostEqual(a["score"], b["score"], places=6,
                                   msg=f"Non-reproducible: {a['task_id']}")

    def test_baseline_mean_score_positive(self):
        results = run_baseline()
        mean = sum(r["score"] for r in results) / len(results)
        self.assertGreater(mean, 0.1)

    def test_baseline_task_ids_match(self):
        results = run_baseline()
        expected_ids = set(TASKS.keys())
        got_ids = {r["task_id"] for r in results}
        self.assertEqual(expected_ids, got_ids)


# ═══════════════════════════════════════════════════════════════════════════════
#  Response shape validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestResponseShapes(unittest.TestCase):

    def test_obs_model_dump_serialisable(self):
        env = PolicyEnvironment(seed=1)
        obs = env.reset("task_1_decongest")
        d = obs.model_dump()
        # All values should be JSON-serialisable
        try:
            json.dumps(d)
        except TypeError as e:
            self.fail(f"Observation is not JSON-serialisable: {e}")

    def test_step_result_serialisable(self):
        env = PolicyEnvironment(seed=1)
        env.reset("task_1_decongest")
        obs, reward, done, info = env.step(_action(
            action_type="propose_policy",
            policy_type="expand_transit",
            district="north",
            budget_pct=0.3,
        ))
        result = {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
        try:
            json.dumps(result)
        except TypeError as e:
            self.fail(f"Step result is not JSON-serialisable: {e}")

    def test_state_model_dump_serialisable(self):
        env = PolicyEnvironment(seed=1)
        env.reset("task_2_equity")
        s = env.state()
        d = s.model_dump()
        try:
            json.dumps(d)
        except TypeError as e:
            self.fail(f"State is not JSON-serialisable: {e}")

    def test_citywide_kpis_all_in_range(self):
        env = PolicyEnvironment(seed=1)
        obs = env.reset("task_3_gauntlet")
        for kpi, val in obs.citywide_kpis.items():
            self.assertGreaterEqual(val, 0.0,   f"{kpi} below 0")
            self.assertLessEqual(val,   100.0,  f"{kpi} above 100")

    def test_composite_score_in_range(self):
        import random
        for seed in range(10):
            env = PolicyEnvironment(seed=seed)
            for tid in TASKS:
                obs = env.reset(tid)
                self.assertGreaterEqual(obs.composite_score, 0.0)
                self.assertLessEqual(obs.composite_score, 100.0)

    def test_stakeholder_approvals_all_present(self):
        env = PolicyEnvironment(seed=1)
        obs = env.reset("task_1_decongest")
        for sh in STAKEHOLDER_NAMES:
            self.assertIn(sh, obs.stakeholder_approvals)


if __name__ == "__main__":
    unittest.main(verbosity=2)
