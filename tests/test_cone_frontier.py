"""Invented saved paths only; no existing scientific experiment is read."""
import copy

import numpy as np
import pytest

from drosolearn.cone_frontier import (
    SCIENCE, analyze_saved, candidate_problem, stats, summarize, validate_config,
)
from drosolearn.selective_geometry import selective_direction


def invented_saved_path():
    """A bounded, deterministic four-input history with partial/full memory."""
    cfg = copy.deepcopy(SCIENCE)
    cfg.update(scope="integration_test", model_seeds=[90001], training_trials=8)
    d = np.asarray([[1., 0., 0., 0.], [.6, .8, 0., 0.],
                    [.3, .1, np.sqrt(.9), 0.], [.2, .1, .1, np.sqrt(.94)]])
    inverse = np.linalg.pinv(d, rcond=1e-12)
    gram = d @ d.T
    isolated = inverse.T / np.linalg.norm(inverse.T, axis=1)[:, None]
    sources = np.asarray([[0, 0, 1, 2, 2, 3, 0, 0]], dtype=np.int32)
    actions = np.asarray([0, 1, 1, 0, 1, 1, 1, 1], dtype=np.int8)
    probabilities = np.asarray([.5, .4, .3, .5, .6, .25, .4, .45])
    y = np.asarray(cfg["labels"], dtype=np.int8)
    rewards = actions * y[sources[0]]
    factors = cfg["head_lr"] * (actions - probabilities) * rewards
    before = np.zeros((8, 4), dtype=np.int8)
    after = before.copy()
    directions = np.empty((8, 4))
    geometry = {"numeric_selective_qualified": True, "unit_rows": d,
                "pseudoinverse": inverse, "base_influences": gram, "tolerance": 1e-8}
    memory = np.zeros(4, dtype=np.int8)
    for trial, source in enumerate(sources[0]):
        before[trial] = memory
        if actions[trial]:
            memory[source] = rewards[trial]
        after[trial] = memory
        directions[trial] = selective_direction(geometry, int(source), memory)["direction"]
    shape = (3, 2, 4, 1, 8)
    parent = {"model_seeds": np.asarray(cfg["model_seeds"]), "targets": (y > 0).astype(int),
              "schedules": sources,
              "trial_actions": np.broadcast_to(actions, shape).copy(),
              "trial_rewards": np.broadcast_to(rewards, shape).copy(),
              "trial_probabilities": np.broadcast_to(probabilities, shape).copy(),
              "trial_update_factors": np.broadcast_to(factors, shape).copy(),
              "trial_known_before": np.broadcast_to(before, shape + (4,)).copy(),
              "trial_known_after": np.broadcast_to(after, shape + (4,)).copy(),
              "trial_routing_signs": np.broadcast_to(after, shape + (4,)).copy(),
              "trial_directions": np.broadcast_to(directions, shape + (4,)).copy(),
              "trial_actual_influence": np.broadcast_to(directions @ d.T, shape + (4,)).copy()}
    for top in cfg["topologies"]:
        for rep in cfg["representations"]:
            prefix = "g_" + top + "_" + rep + "_"
            parent[prefix + "unit_rows"] = d[None].copy()
            parent[prefix + "base_influences"] = gram[None].copy()
            parent[prefix + "isolated_directions"] = isolated[None].copy()
    return cfg, parent


def saved_fixture():
    """Balanced two-seed bridge fixture; seed two never observes input three."""
    cfg, prototype = invented_saved_path()
    cfg["model_seeds"] = [90001, 90002]
    d = prototype["g_real_raw_unit_rows"][0]
    inverse, gram = np.linalg.pinv(d, rcond=1e-12), d @ d.T
    geometry = {"numeric_selective_qualified": True, "unit_rows": d,
                "pseudoinverse": inverse, "base_influences": gram, "tolerance": 1e-8}
    shape = (3, 2, 4, 2, 8)
    sources = np.tile(np.arange(4), (2, 2)).astype(np.int32)
    actions = np.asarray([[0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1, 1, 0]], dtype=np.int8)
    probabilities = np.tile([.5, .4, .3, .5, .6, .25, .4, .45], (2, 1))
    y = np.asarray(cfg["labels"], dtype=np.int8)
    rewards = actions * y[sources]
    factors = cfg["head_lr"] * (actions - probabilities) * rewards
    before = np.zeros((2, 8, 4), dtype=np.int8)
    after = before.copy()
    directions, desired = np.zeros((2, 8, 4)), np.zeros((2, 8, 4))
    for si in range(2):
        memory = np.zeros(4, dtype=np.int8)
        for trial, source in enumerate(sources[si]):
            before[si, trial] = memory
            if actions[si, trial]:
                memory[source] = rewards[si, trial]
            after[si, trial] = memory
            result = selective_direction(geometry, int(source), memory)
            directions[si, trial], desired[si, trial] = result["direction"], result["desired_influence"]
    parent = {"model_seeds": np.asarray(cfg["model_seeds"]), "targets": (y > 0).astype(int),
              "schedules": sources}
    scalars = {"actions": actions, "rewards": rewards, "probabilities": probabilities,
               "update_factors": factors, "head_steps": np.abs(factors)}
    for field, value in scalars.items():
        parent["trial_" + field] = np.broadcast_to(value, shape).copy()
    vectors = {"known_before": before, "known_after": after, "routing_signs": after,
               "directions": directions, "actual_influence": directions @ d.T,
               "desired_influence": desired}
    for field, value in vectors.items():
        parent["trial_" + field] = np.broadcast_to(value, shape + (4,)).copy()
    for field in ["internal_steps", "value_steps", "internal_update_calls", "value_update_calls"]:
        parent["trial_" + field] = np.zeros(shape)
    for top in cfg["topologies"]:
        for rep in cfg["representations"]:
            prefix = "g_" + top + "_" + rep + "_"
            for field in ["unit_rows", "base_influences", "isolated_directions"]:
                parent[prefix + field] = np.repeat(prototype[prefix + field], 2, axis=0)
            parent[prefix + "pseudoinverse"] = np.repeat(inverse[None], 2, axis=0)
    return cfg, parent


@pytest.fixture(scope="module")
def saved_analysis():
    cfg, parent = invented_saved_path()
    arrays, rows = analyze_saved(cfg, parent)
    return cfg, parent, arrays, rows


def test_unknown_signs_are_absent_from_observed_constraints():
    d = np.eye(4)
    a, objective, known, helpful = candidate_problem(d, 0, [1, 0, -1, 0], "own")
    assert np.array_equal(known, [0, 2])
    assert np.array_equal(a, [d[0], -d[2]])
    assert np.array_equal(objective, d[0]) and helpful.size == 0


def test_negative_source_orients_known_destination_margin():
    d = np.eye(4)
    a, objective, known, _ = candidate_problem(d, 2, [1, 0, -1, -1], "own")
    assert np.array_equal(known, [0, 2, 3])
    assert np.array_equal(a, [-d[0], d[2], d[3]])
    assert np.array_equal(objective, d[2])


def test_balanced_objective_uses_only_known_base_helpful_destinations():
    cfg, parent = invented_saved_path()
    d = parent["g_real_raw_unit_rows"][0]
    _, objective, _, helpful = candidate_problem(d, 0, [1, 1, -1, 0], "balanced")
    assert np.array_equal(helpful, [1])
    assert np.allclose(objective, .5 * d[0] + .5 * d[1])
    assert not np.allclose(objective, .5 * d[0] + .5 * (d[1] - d[2]) / 2.)


def test_balanced_falls_back_to_own_when_no_known_help_exists():
    d = np.eye(4)
    _, balanced, _, helpful = candidate_problem(d, 1, [0, 1, -1, 0], "balanced")
    assert not helpful.size and np.array_equal(balanced, d[1])


def test_unknown_source_is_not_silently_given_its_target():
    with pytest.raises(ValueError, match="Actually known source"):
        candidate_problem(np.eye(4), 1, [1, 0, -1, -1], "own")


@pytest.mark.parametrize("source,objective", [(True, "own"), (4, "own"), (0, "help")])
def test_invalid_source_or_unregistered_objective_rejected(source, objective):
    with pytest.raises(ValueError):
        candidate_problem(np.eye(4), source, [1, 1, -1, -1], objective)


def test_missing_ratio_is_excluded_without_becoming_zero():
    result = stats([None, 2., 4.])
    assert result["mean"] == 3. and result["sd"] == pytest.approx(np.sqrt(2.))
    assert result["valid_count"] == 2 and result["total_count"] == 3
    assert stats([None]) == {"mean": None, "sd": None, "valid_count": 0, "total_count": 1}
    assert stats([2.])["sd"] == 0.


def test_nonfinite_statistic_rejected():
    with pytest.raises(ValueError):
        stats([1., np.nan])


def test_scientific_controls_cannot_change_in_a_fixture():
    cfg, _ = invented_saved_path()
    cfg["head_lr"] = .2
    with pytest.raises(ValueError, match="head_lr"):
        validate_config(cfg)


def test_fixture_cannot_use_actual_seeds_or_extend_the_path():
    cfg, _ = invented_saved_path()
    cfg["model_seeds"] = [17401]
    with pytest.raises(ValueError, match="Invented fixture"):
        validate_config(cfg)
    cfg["model_seeds"], cfg["training_trials"] = [90001], 400
    with pytest.raises(ValueError, match="Bounded invented"):
        validate_config(cfg)


def test_analysis_never_mutates_saved_path_or_configuration():
    cfg, parent = invented_saved_path()
    before_cfg, before_parent = copy.deepcopy(cfg), {k: v.copy() for k, v in parent.items()}
    analyze_saved(cfg, parent)
    assert cfg == before_cfg
    assert all(np.array_equal(value, before_parent[key]) for key, value in parent.items())


def test_nogo_has_zero_effect_and_no_optimization_case(saved_analysis):
    _, _, arrays, _ = saved_analysis
    for trial in [0, 3]:
        assert not arrays["candidate_margin_delta"][..., trial, :, :].any()
        assert not arrays["candidate_step_lengths"][..., trial, :].any()
        assert np.all(arrays["case_indices"][..., trial, :] == -1)
    assert np.array_equal(arrays["known_after"][0, 0, 0, 3], [1, 1, 0, 0])


def test_current_go_reward_is_available_without_future_signs(saved_analysis):
    _, _, arrays, _ = saved_analysis
    at = (0, 0, 0, 1)
    case = arrays["case_indices"][at + (0,)]
    assert np.array_equal(arrays["known_after"][at], [1, 0, 0, 0])
    assert np.array_equal(arrays["case_keys"][case][-4:], [1, 0, 0, 0])
    assert np.any(arrays["case_constraints"][case, 0])
    assert not arrays["case_constraints"][case, 1:].any()


def test_cache_reuses_same_source_and_observed_record_only(saved_analysis):
    _, _, arrays, _ = saved_analysis
    assert np.array_equal(arrays["case_indices"][0, 0, 0, 6], arrays["case_indices"][0, 0, 0, 7])
    assert not np.array_equal(arrays["case_indices"][0, 0, 0, 1, :2],
                              arrays["case_indices"][0, 0, 0, 6, :2])
    assert np.array_equal(arrays["case_indices"][0, 0, 0, 1, 2:],
                          arrays["case_indices"][0, 0, 0, 6, 2:])
    assert len(arrays["case_keys"]) == 108


def test_signed_effects_and_cap_use_the_same_saved_scalar(saved_analysis):
    cfg, _, arrays, _ = saved_analysis
    expected = arrays["candidate_influences"] * arrays["update_factors"][..., None, None] * np.asarray(cfg["labels"])
    assert np.allclose(arrays["candidate_margin_delta"], expected)
    assert np.max(arrays["candidate_step_lengths"] - np.abs(arrays["update_factors"])[..., None]) < 1e-12
    assert np.max(arrays["candidate_step_lengths"]) < cfg["head_lr"]
    assert arrays["update_factors"][0, 0, 0, 4] < 0.
    assert arrays["candidate_margin_delta"][0, 0, 0, 4, 3, 2] > 0.


def test_retention_denominator_is_stored_selective_effect_not_raw(saved_analysis):
    _, _, _, rows = saved_analysis
    base = next(r for r in rows if (r["topology"], r["representation"], r["candidate"], r["phase"])
                == ("real", "raw", "parent_feedback", "all"))
    shared = next(r for r in rows if (r["topology"], r["representation"], r["candidate"], r["phase"])
                  == ("real", "raw", "shared_reference", "all"))
    assert base["own_retention"] == 1. and base["help_retention"] == 1.
    assert shared["own_margin"] > base["own_margin"]
    assert shared["own_retention"] == pytest.approx(shared["own_margin"] / base["own_margin"])


def test_full_phase_begins_on_the_reward_that_completes_memory(saved_analysis):
    _, _, arrays, rows = saved_analysis
    assert np.array_equal(arrays["full_known_after"][0, 0, 0], [False] * 5 + [True] * 3)
    selected = [r for r in rows if (r["topology"], r["representation"], r["candidate"])
                == ("real", "raw", "observed_own")]
    counts = {r["phase"]: (r["trial_count"], r["updated_trials"]) for r in selected}
    assert counts == {"all": (8, 6), "partial": (5, 3), "full": (3, 3)}
    assert selected[0]["own_margin"] == pytest.approx(selected[1]["own_margin"] + selected[2]["own_margin"])


@pytest.mark.parametrize("field", ["trial_known_before", "trial_routing_signs", "trial_update_factors"])
def test_corrupt_saved_information_or_scalar_is_rejected(field):
    cfg, parent = invented_saved_path()
    at = (0, 0, 3, 0, 1)
    if field == "trial_update_factors":
        parent[field][at] += .01
    else:
        parent[field][at + (3,)] = -1
    with pytest.raises(ValueError):
        analyze_saved(cfg, parent)


def test_probability_outside_unit_interval_cannot_supply_a_larger_step():
    cfg, parent = invented_saved_path()
    at = (0, 0, 3, 0, 1)
    parent["trial_probabilities"][at] = -.5
    parent["trial_update_factors"][at] = .15
    with pytest.raises(ValueError):
        analyze_saved(cfg, parent)


def test_summary_counts_are_derived_without_new_learning(saved_analysis):
    cfg, parent, arrays, rows = saved_analysis
    summary = summarize(cfg, arrays, rows)
    assert summary["counts"]["analyzed_parent_trial_records"] == 48
    assert summary["counts"]["nonzero_parent_updates"] == 36
    assert summary["counts"]["candidate_trial_records"] == 336
    assert summary["counts"]["candidate_destination_effects"] == 1344
    assert summary["counts"]["measurement_records"] == 126
    assert summary["counts"]["new_training_trials"] == summary["counts"]["new_random_draws"] == 0
    assert all(value for key, value in summary["validation"].items() if key.startswith("all_"))
    assert summary["interpretation"]["score_not_accuracy"]
    assert np.array_equal(parent["model_seeds"], [90001])


def test_summary_rejects_duplicate_rows_and_broken_phase_balance(saved_analysis):
    cfg, _, arrays, rows = saved_analysis
    with pytest.raises(ValueError, match="Complete unique"):
        summarize(cfg, arrays, rows[:-1] + [rows[0]])
    broken = copy.deepcopy(rows)
    broken[0]["own_margin"] += .01
    with pytest.raises(ValueError, match="phase gate"):
        summarize(cfg, arrays, broken)


def test_summary_rejects_a_candidate_exceeding_the_saved_scalar_cap(saved_analysis):
    cfg, _, arrays, rows = saved_analysis
    broken = {key: value.copy() for key, value in arrays.items()}
    broken["candidate_step_lengths"][0, 0, 0, 1, 3] = .2
    with pytest.raises(ValueError, match="unit-cap"):
        summarize(cfg, broken, rows)
