"""Synthetic metric-calibration tests: no real data and no scientific seed runs."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from drosolearn import metric_calibration as calibration
from drosolearn.feedback_transition import measure_stream


REPO = Path(__file__).resolve().parents[1]


def synthetic_config(trials=8, seeds=(3, 4)):
    cfg = json.loads((REPO / "configs/metric_calibration_v1.json").read_text(encoding="utf8"))
    cfg["scope"] = "integration_test"
    cfg["seeds"] = list(seeds)
    cfg["trials"] = trials
    midpoint = trials // 2
    cfg["report_segments"] = [
        {"id": "phase1", "start_trial": 1, "end_trial": midpoint},
        {"id": "phase2", "start_trial": midpoint + 1, "end_trial": trials},
    ]
    cfg["schedules"] = [
        {"id": "fixed", "state0_end_trial": trials},
        {"id": "reversal", "state0_end_trial": midpoint},
    ]
    cfg["rng"]["array_length"] = trials
    return cfg


def item(cfg, category, identifier):
    return next(entry for entry in cfg[category] if entry["id"] == identifier)


def simulate(cfg, policy="iid_biased", environment="probabilistic", schedule="fixed", seed=3):
    assert seed not in range(9001, 9011), "Scientific seeds must never run in these tests"
    return calibration.simulate(cfg, seed, item(cfg, "schedules", schedule),
                                item(cfg, "policies", policy), item(cfg, "environments", environment))


def test_synthetic_and_registered_config_are_valid_without_generating_actions(monkeypatch):
    cfg = synthetic_config()
    original = deepcopy(cfg)
    monkeypatch.setattr(calibration, "simulate", lambda *a, **k: pytest.fail("Validation generated trials"))
    calibration.validate_config(cfg)
    calibration.validate_config(json.loads((REPO / "configs/metric_calibration_v1.json").read_text(encoding="utf8")))
    assert cfg == original


@pytest.mark.parametrize("mutation", [
    lambda cfg: cfg.update(seeds=[3, 3]),
    lambda cfg: cfg.update(trials=7),
    lambda cfg: cfg["rng"].update(array_length=9),
    lambda cfg: cfg["policies"][1].update(p1=float("nan")),
    lambda cfg: cfg["policies"][1].update(p1=1.1),
    lambda cfg: cfg["schedules"][1].update(state0_end_trial=3),
    lambda cfg: cfg.update(learning=True),
    lambda cfg: cfg.update(policy_reset_at_reversal=True),
    lambda cfg: cfg.update(monte_carlo_inference=True),
    lambda cfg: cfg.update(seeds=[9001]),
])
def test_invalid_or_scope_changing_config_is_rejected(mutation):
    cfg = synthetic_config()
    mutation(cfg)
    with pytest.raises(ValueError):
        calibration.validate_config(cfg)


@pytest.mark.parametrize("policy,expected", [
    ({"id": "iid_fair", "kind": "iid", "p1": .5}, [1, 0, 0, 0, 0, 1]),
    ({"id": "iid_biased", "kind": "iid", "p1": .8}, [1, 1, 1, 0, 0, 1]),
    ({"id": "sticky", "kind": "sticky", "initial_p1": .5, "p_stay": .9}, [1, 1, 1, 1, 0, 0]),
])
def test_policy_actions_follow_fixed_thresholds_without_reading_feedback(policy, expected):
    uniforms = np.array([.1, .5, .799, .8, .9, .2])
    before = uniforms.copy()
    policy_before = deepcopy(policy)
    actual = calibration.policy_actions(policy, uniforms)
    assert list(actual) == expected
    np.testing.assert_array_equal(uniforms, before)
    assert policy == policy_before


@pytest.mark.parametrize("policy", ["iid_fair", "iid_biased", "sticky"])
def test_action_path_is_shared_across_every_environment_and_schedule_and_matches_registered_rng(policy):
    cfg = synthetic_config(16)
    original = deepcopy(cfg)
    actions = []
    for environment in cfg["environments"]:
        for schedule in cfg["schedules"]:
            rows = simulate(cfg, policy, environment["id"], schedule["id"])
            actions.append([row["choice"] for row in rows])
            np.testing.assert_array_equal([row["action_uniform"] for row in rows],
                                          np.random.default_rng(410003).random(16))
            np.testing.assert_array_equal([row["reward_uniform"] for row in rows],
                                          np.random.default_rng(420003).random(16))
    assert all(path == actions[0] for path in actions)
    assert cfg == original


@pytest.mark.parametrize("schedule,expected_reasons", [
    ("fixed", {"phase_boundary": 1}),
    ("reversal", {"phase_boundary": 1, "state_boundary": 1}),
])
def test_canonical_rows_keep_original_positions_and_exclude_one_shared_boundary(schedule, expected_reasons):
    cfg = synthetic_config()
    rows = simulate(cfg, schedule=schedule)
    assert [row["trial"] for row in rows] == list(range(1, 9))
    assert [row["source_position"] for row in rows] == list(range(1, 9))
    assert [row["phase"] for row in rows] == ["phase1"] * 4 + ["phase2"] * 4
    assert all(type(row["choice"]) is int and type(row["feedback"]) is int for row in rows)
    assert all(row["actual"] is True and row["training"] is True for row in rows)
    summary, ledger = measure_stream(rows)
    assert summary["candidates"] == 7 and summary["eligible"] == 6 and summary["excluded"] == 1
    assert summary["reason_counts"] == expected_reasons
    assert ledger[3]["trial"] == 4 and ledger[3]["next_trial"] == 5 and ledger[3]["eligible"] == 0
    assert all(entry["eligible"] for i, entry in enumerate(ledger) if i != 3)
    phases = [measure_stream(rows[:4])[0], measure_stream(rows[4:])[0]]
    for feedback in ("0", "1"):
        for field in ("denominator", "stay", "switch"):
            assert summary["by_feedback"][feedback][field] == sum(
                phase["by_feedback"][feedback][field] for phase in phases)


@pytest.mark.parametrize("schedule", ["fixed", "reversal"])
def test_actual_feedback_is_generated_from_chosen_option_and_registered_environment(schedule):
    cfg = synthetic_config(16)
    env = item(cfg, "environments", "probabilistic")
    rows = simulate(cfg, environment=env["id"], schedule=schedule)
    q0 = env["reward1_given_choice_state0"]
    for index, row in enumerate(rows):
        probabilities = q0[::-1] if schedule == "reversal" and index >= 8 else q0
        assert row["feedback"] == int(row["reward_uniform"] < probabilities[row["choice"]])


def test_sticky_policy_does_not_reset_on_reversal_or_phase_boundary():
    cfg = synthetic_config(16)
    rows = simulate(cfg, policy="sticky", schedule="reversal")
    expected = int(rows[0]["action_uniform"] < .5)
    assert rows[0]["choice"] == expected
    for row in rows[1:]:
        expected = expected if row["action_uniform"] < .9 else 1 - expected
        assert row["choice"] == expected


def analytic_cells(value):
    return value["by_feedback"]


@pytest.mark.parametrize("policy,environment,expected", [
    ("iid_fair", "probabilistic", (.5, .5)),
    ("iid_fair", "deterministic", (.5, .5)),
    ("iid_fair", "uninformative", (.5, .5)),
    ("iid_biased", "probabilistic", (13 / 17, .5)),
    ("iid_biased", "deterministic", (.8, .2)),
    ("iid_biased", "uninformative", (.68, .68)),
    ("iid_biased", "swapped", (.5, 13 / 17)),
    ("sticky", "probabilistic", (.9, .9)),
    ("sticky", "deterministic", (.9, .9)),
    ("sticky", "uninformative", (.9, .9)),
    ("sticky", "swapped", (.9, .9)),
])
def test_analytic_counterexamples_hold_without_fitting_realized_actions(policy, environment, expected):
    cfg = synthetic_config()
    rows = simulate(cfg, policy, environment)
    value = calibration.analytic(rows, item(cfg, "policies", policy), item(cfg, "environments", environment))
    cells = analytic_cells(value)
    assert [cells[str(f)]["ratio_of_expected_counts"] for f in (0, 1)] == pytest.approx(expected, abs=1e-12)
    assert sum(cells[str(f)]["expected_denominator"] for f in (0, 1)) == pytest.approx(6)
    for cell in cells.values():
        assert cell["expected_stay"] + cell["expected_switch"] == pytest.approx(cell["expected_denominator"])
    changed = deepcopy(rows)
    for row in changed:
        row.update(choice=0, feedback=0)
    assert calibration.analytic(changed, item(cfg, "policies", policy),
                                item(cfg, "environments", environment)) == value


def test_reversal_phase_counterexamples_cancel_in_whole_stream_exact_expectations():
    cfg = synthetic_config()
    rows = simulate(cfg, schedule="reversal")
    policy, env = item(cfg, "policies", "iid_biased"), item(cfg, "environments", "probabilistic")
    whole = analytic_cells(calibration.analytic(rows, policy, env))
    first = analytic_cells(calibration.analytic(rows[:4], policy, env))
    last = analytic_cells(calibration.analytic(rows[4:], policy, env))
    difference = lambda cells: cells["1"]["ratio_of_expected_counts"] - cells["0"]["ratio_of_expected_counts"]
    assert difference(first) == pytest.approx(-9 / 34, abs=1e-12)
    assert difference(last) == pytest.approx(9 / 34, abs=1e-12)
    assert abs(difference(first)) >= .2
    assert difference(whole) == pytest.approx(0, abs=1e-12)
    assert whole["0"]["ratio_of_expected_counts"] == pytest.approx(.68)
    assert whole["1"]["ratio_of_expected_counts"] == pytest.approx(.68)
    for feedback in ("0", "1"):
        for field in ("expected_denominator", "expected_stay", "expected_switch"):
            assert whole[feedback][field] == pytest.approx(first[feedback][field] + last[feedback][field])


def test_expected_counts_match_precomputed_biased_probabilistic_joint_distribution():
    cfg = synthetic_config()
    cells = analytic_cells(calibration.analytic(simulate(cfg), cfg["policies"][1], cfg["environments"][0]))
    # Six eligible pairs. For example P(a=1,f=1,next=1)=.8*.2*.8.
    # These are expected counts, so fractional values are intentional.
    assert cells["0"] == pytest.approx({"expected_denominator": 4.08, "expected_stay": 3.12,
                                        "expected_switch": .96, "ratio_of_expected_counts": 13 / 17})
    assert cells["1"] == pytest.approx({"expected_denominator": 1.92, "expected_stay": .96,
                                        "expected_switch": .96, "ratio_of_expected_counts": .5})


def test_analytic_empty_eligible_set_stays_undefined_instead_of_becoming_zero():
    cfg = synthetic_config()
    rows = simulate(cfg)[:2]
    rows[1]["phase"] = "phase2"
    policy = item(cfg, "policies", "iid_fair")
    env = item(cfg, "environments", "probabilistic")
    cells = analytic_cells(calibration.analytic(rows, policy, env))
    for cell in cells.values():
        assert cell["expected_denominator"] == cell["expected_stay"] == cell["expected_switch"] == 0
        assert cell["ratio_of_expected_counts"] is None


def test_empirical_empty_feedback_cell_is_reported_as_undefined_with_zero_count():
    cfg = synthetic_config()
    rows = simulate(cfg)
    for entry in rows:
        entry.update(choice=0, feedback=1)
    measured, _ = measure_stream(rows)
    assert measured["by_feedback"]["0"] == {
        "denominator": 0, "stay": 0, "switch": 0, "stay_rate": None, "switch_rate": None}
    assert measured["by_feedback"]["1"]["stay_rate"] == 1


@pytest.mark.parametrize("uniforms", [
    np.array([]), np.array([1.]), np.array([-.01]), np.array([np.nan]),
    np.array([np.inf]), np.array([[.1, .2]]), np.array([True, False]),
])
def test_policy_rejects_invalid_original_uniforms(uniforms):
    with pytest.raises(ValueError):
        calibration.policy_actions(calibration.POLICIES[0], uniforms)


def test_unregistered_policy_and_environment_cannot_silently_change_scientific_scope():
    cfg = synthetic_config()
    with pytest.raises(ValueError):
        calibration.policy_actions({"id": "custom", "kind": "iid", "p1": 1.0}, np.array([.1]))
    with pytest.raises(ValueError):
        calibration.simulate(cfg, 3, cfg["schedules"][0], cfg["policies"][0],
                             {"id": "custom", "reward1_given_choice_state0": [1., 1.]})
