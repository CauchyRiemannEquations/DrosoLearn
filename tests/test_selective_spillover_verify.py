"""Invented matrix and feedback checks; research seeds are never sampled."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "selective_independent_tests", Path(__file__).resolve().parents[1] / "scripts/verify_selective_spillover.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def correlated_rows():
    return np.array([[1., 0., 0., 0.], [1., .2, 0., 0.],
                     [1., 0., .2, 0.], [1., 0., 0., .2]])


def compare_records(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, dict): compare_records(actual[key], value)
        elif isinstance(value, np.ndarray): assert np.allclose(actual[key], value, atol=1e-9, rtol=1e-9), key
        elif isinstance(value, float): assert actual[key] == pytest.approx(value, abs=1e-9), key
        else: assert actual[key] == value, key


@pytest.mark.parametrize("rows", [np.eye(4), correlated_rows(),
                                 np.array([[1., 0., 0., 0.], [-1., .2, 0., 0.],
                                           [1., 0., .2, 0.], [1., 0., 0., .2]])])
def test_new_helper_geometry_matches_independent(rows):
    from drosolearn.selective_geometry import selective_geometry
    compare_records(selective_geometry(rows), audit.independent_geometry(rows))


@pytest.mark.parametrize("source", range(4))
@pytest.mark.parametrize("signs", [np.zeros(4, int), np.array([1, 0, -1, 0]),
                                   np.array([1, 1, -1, -1]), np.array([-1, -1, 1, 1])])
def test_new_direction_helper_matches_independent(source, signs):
    from drosolearn.selective_geometry import selective_geometry, selective_direction
    rows = correlated_rows()
    compare_records(selective_direction(selective_geometry(rows), source, signs),
                    audit.independent_selective_direction(audit.independent_geometry(rows), source, signs))


@pytest.mark.parametrize("source", range(4))
def test_unknown_feedback_never_classifies_any_pair(source):
    geometry = audit.independent_geometry(correlated_rows())
    result = audit.independent_direction(geometry, source, np.zeros(4, int), "feedback_selective")
    assert not result["harm_mask"].any()
    assert np.allclose(result["direction"], geometry["unit_rows"][source])


@pytest.mark.parametrize("source", range(4))
def test_known_pair_removes_only_identified_negative_alignment_before_normalization(source):
    geometry = audit.independent_geometry(correlated_rows())
    result = audit.independent_direction(geometry, source, audit.SIGNS, "feedback_selective")
    gram = geometry["base_influences"][source]
    harmful = audit.SIGNS[source] * audit.SIGNS * gram < -1e-8
    assert np.array_equal(result["harm_mask"], harmful)
    assert np.array_equal(result["desired_influence"][~harmful], gram[~harmful])
    assert np.all(result["desired_influence"][harmful] == 0.)
    assert np.allclose(geometry["unit_rows"] @ result["unnormalized"], result["desired_influence"])
    assert np.allclose(result["actual_influence"][harmful], 0., atol=1e-10)
    assert result["direction_l2"] == pytest.approx(1.)


def test_negative_gram_can_be_helpful_for_opposite_sign_and_harmful_for_same_sign():
    rows = correlated_rows()
    rows[1] *= -1
    geometry = audit.independent_geometry(rows)
    known = np.array([1, -1, -1, -1])
    result = audit.independent_direction(geometry, 0, known, "feedback_selective")
    assert not result["harm_mask"][1]
    known[1] = 1
    result = audit.independent_direction(geometry, 0, known, "feedback_selective")
    assert result["harm_mask"][1]


def test_oracle_and_feedback_have_same_direction_given_same_explicit_signs():
    geometry = audit.independent_geometry(correlated_rows())
    for source in range(4):
        a = audit.independent_direction(geometry, source, audit.SIGNS, "oracle_selective")
        b = audit.independent_direction(geometry, source, audit.SIGNS, "feedback_selective")
        for field in a:
            assert np.allclose(a[field], b[field])


def test_nogo_provides_no_information_and_no_output_update():
    actor = audit.IndependentHead(correlated_rows(), "feedback_selective")
    before = actor.head.copy()
    record = actor.observe(2, 0, 0)
    assert np.array_equal(actor.known, [0, 0, 0, 0])
    assert np.array_equal(actor.head, before)
    assert np.array_equal(record["known_before"], record["known_after"])
    assert not record["harm_mask"].any()
    assert actor.calls == 1 and actor.path_l2 == 0.


def test_current_go_feedback_is_remembered_before_current_direction_and_only_observed_item():
    actor = audit.IndependentHead(correlated_rows(), "feedback_selective")
    first = actor.observe(0, 1, 1)
    assert np.array_equal(first["known_before"], [0, 0, 0, 0])
    assert np.array_equal(first["known_after"], [1, 0, 0, 0])
    second = actor.observe(2, 1, -1)
    assert np.array_equal(second["known_before"], [1, 0, 0, 0])
    assert np.array_equal(second["known_after"], [1, 0, -1, 0])
    assert np.array_equal(second["harm_mask"], [True, False, False, False])
    assert not second["harm_mask"][3]


@pytest.mark.parametrize("mode", audit.MODES)
def test_all_modes_unit_steps_but_realized_cumulative_costs_can_differ(mode):
    actor = audit.IndependentHead(correlated_rows(), mode)
    for source in (0, 1, 2, 3, 0, 2):
        record = actor.observe(source, 1, int(audit.SIGNS[source]))
        assert record["head_step_l2"] == pytest.approx(abs(record["factor"]))
        assert record["head_step_l2"] <= .1
    assert actor.calls == 6
    assert actor.path_l2 <= .6
    assert actor.squared_l2 <= .06
    assert np.linalg.norm(actor.head) <= actor.path_l2 + 1e-12


def test_all_memory_starts_unknown_and_only_oracle_routes_with_initial_privileged_signs():
    for mode in audit.MODES:
        actor = audit.IndependentHead(correlated_rows(), mode)
        assert np.array_equal(actor.known, np.zeros(4))
        record = actor.observe(0, 0, 0)
        assert np.array_equal(record["routing_signs"], audit.SIGNS if mode == "oracle_selective" else np.zeros(4))


@pytest.mark.parametrize("index,action,reward", [(0, 0, -1), (0, 0, 1), (0, 1, 0),
                                               (4, 1, 1), (-1, 1, 1), (0, 2, 1), (0, 1, 2)])
def test_invalid_observation_contract_rejected(index, action, reward):
    actor = audit.IndependentHead(correlated_rows(), "feedback_selective")
    with pytest.raises(ValueError): actor.observe(index, action, reward)


def test_conflicting_feedback_rejected_without_hiding_previous_sign():
    actor = audit.IndependentHead(correlated_rows(), "feedback_selective")
    actor.observe(0, 1, 1)
    before = actor.head.copy()
    with pytest.raises(ValueError): actor.observe(0, 1, -1)
    assert np.array_equal(actor.head, before)
    assert actor.known[0] == 1


@pytest.mark.parametrize("known", [np.zeros(3, int), np.array([0, 0, 0, 2]), np.zeros(4, float)])
def test_invalid_observed_signs_are_not_implicitly_coerced(known):
    geometry = audit.independent_geometry(correlated_rows())
    with pytest.raises(ValueError): audit.independent_direction(geometry, 0, known, "feedback_selective")


def tiny_graph():
    n = 16
    return {"pre": np.arange(n), "post": np.arange(n), "counts": np.ones(n), "n_kc": n, "n_mbon": n}


def fixture_config():
    return {"study_id": "selective_spillover_v1", "scope": "integration_test",
        "module": "drosolearn.selective_spillover", "protocol": "docs/selective-spillover-protocol.md",
        "topologies": list(audit.TOPOLOGIES), "representations": list(audit.INPUTS), "methods": list(audit.MODES),
        "model_seeds": [90083, 90084], "training_trials": 8, "checkpoints": [0, 4, 8],
        "head_lr": .1, "rng_namespace": 202610196, "svd_rcond": 1e-12, "numeric_tolerance": 1e-8,
        "exact_condition_max": 1e6, "acquisition_threshold": .8, "acquisition_gain": .25,
        "class_threshold": .7, "population_pass_count": 2, "effect_threshold": .05, "positive_seed_count": 2,
        "normalization": "per_input_initial_fixed_MBON_gap_unit_L2", "input_catalog": audit.INPUT_CATALOG,
        "budget_convention": "unit_directions_common_eta_ceiling_with_observed_reward_memory",
        "graph": "data/flywire_783_mb_left_kc512_s701", "external_manifest": None}


@pytest.fixture(scope="module")
def independently_replayed():
    cfg = fixture_config()
    arrays, rows = audit.expected_artifacts(cfg, tiny_graph())
    return cfg, arrays, rows


def test_complete_invented_replay_and_preflight_counts(independently_replayed):
    cfg, arrays, rows = independently_replayed
    assert arrays["trial_actions"].shape == (3, 2, 4, 2, 8)
    assert arrays["trial_directions"].shape == (3, 2, 4, 2, 8, 16)
    assert arrays["checkpoint_probabilities"].shape == (3, 2, 4, 2, 3, 4)
    assert len(rows) == 384
    assert arrays["g_real_raw_preflight_direction_count"].tolist() == [324, 324]
    assert np.max(arrays["trial_head_steps"]) <= .1 + 1e-12
    assert np.allclose(arrays["trial_cumulative_head_path_l2"], np.cumsum(arrays["trial_head_steps"], axis=-1))
    assert np.allclose(arrays["trial_cumulative_head_squared_l2"], np.cumsum(arrays["trial_head_steps"] ** 2, axis=-1))
    assert all(audit.information_checks(cfg, arrays).values())
    audit.check_artifacts(arrays, rows, arrays, rows)
    summary = audit.expected_summary(cfg, arrays, rows)
    assert summary["counts"]["lookup_autonomous_training_trials"] == 0
    assert summary["counts"]["preflight_direction_evaluations"] == 3888
    assert summary["counts"]["runtime_selective_direction_evaluations"] == 192


@pytest.mark.parametrize("key,value", [("methods", list(reversed(audit.MODES))), ("head_lr", .2),
    ("rng_namespace", 99), ("svd_rcond", 1e-6), ("normalization", "different"),
    ("effect_threshold", .01), ("class_threshold", .2), ("budget_convention", "unequal_steps")])
def test_registered_parameters_are_fixed_in_invented_fixtures(key, value):
    cfg = fixture_config()
    cfg[key] = value
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


def test_invented_scope_cannot_open_research_execution():
    cfg = fixture_config()
    audit.validate_science(cfg, allow_fixture=True)
    with pytest.raises(ValueError): audit.validate_science(cfg)
    cfg["model_seeds"] = [17401]
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)
    cfg = fixture_config()
    cfg["training_trials"] = 400
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


@pytest.mark.parametrize("field", ["inputs", "targets", "schedules", "uniforms", "trial_actions",
    "trial_probabilities", "trial_known_before", "trial_known_after", "trial_routing_signs",
    "trial_harm_mask", "trial_desired_influence", "trial_actual_influence", "trial_directions",
    "trial_direction_residuals", "trial_memory_changed", "trial_memory_observation_calls",
    "trial_memory_discoveries", "first_discovery_trials", "checkpoint_known", "checkpoint_heads",
    "checkpoint_lookup_probabilities", "g_real_raw_unit_rows", "g_real_raw_pseudoinverse",
    "g_real_raw_preflight_direction_count", "g_ann_identity_preflight_max_residuals",
    "w_real_raw_weights", "finalfixed_rewired_identity_probe_mbon"])
def test_complete_state_and_dynamic_geometry_tampering_rejected(independently_replayed, field):
    _, original, rows = independently_replayed
    arrays = copy.deepcopy(original)
    flat = arrays[field].reshape(-1)
    if flat.dtype.kind == "b": flat[-1] = not flat[-1]
    else: flat[-1] += 1 if flat.dtype.kind in "iu" else .25
    with pytest.raises(ValueError): audit.check_artifacts(original, rows, arrays, rows)


def test_csv_state_or_field_tampering_rejected(independently_replayed):
    _, arrays, original = independently_replayed
    rows = copy.deepcopy(original)
    rows[0]["known_after_0"] += 1
    with pytest.raises(ValueError): audit.check_artifacts(arrays, original, arrays, rows)
    rows = copy.deepcopy(original)
    del rows[0]["direction_residual_max"]
    with pytest.raises(ValueError): audit.check_artifacts(arrays, original, arrays, rows)


def write_input_catalog(root):
    path = root / audit.INPUT_CATALOG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"study_id": "representation_update_v1", "representations": list(audit.INPUTS),
        "catalog": {key: value.tolist() for key, value in audit.prior.independent_catalog_arrays().items()}}), encoding="utf8")


def test_full_current_producer_matches_independent_fixture(monkeypatch, tmp_path):
    from drosolearn import selective_spillover as producer
    from drosolearn.graph import Projection
    raw = tiny_graph()
    n = raw["n_kc"]
    graph = Projection(raw["pre"], raw["post"], raw["counts"], n, n,
                       [str(i) for i in range(n)], [str(i) for i in range(n)], {})
    monkeypatch.setattr(producer, "load_projection", lambda _: graph)
    cfg = fixture_config()
    write_input_catalog(tmp_path)
    arrays, rows = producer.produce(cfg, tmp_path)
    expected, expected_rows = audit.expected_artifacts(cfg, raw)
    audit.check_artifacts(expected, expected_rows, arrays, rows)
    summary = producer.summarize(cfg, arrays, rows)
    receipt = audit.verify_against_graph(cfg, raw, arrays, rows, summary, allow_fixture=True)
    assert receipt["independently_replayed_training_actions"] == 384
    assert receipt["feedback_memory_from_obtained_Go_only_checked"]


def summary_gate_fixture(independently_replayed):
    cfg, original, rows = independently_replayed
    arrays = copy.deepcopy(original)
    arrays["checkpoint_probabilities"][:] = .5
    return cfg, arrays, rows


def set_final_correct(arrays, method, value, seed=slice(None)):
    arrays["checkpoint_probabilities"][0, 0, method, seed, -1] = np.where(audit.TARGETS, value, 1. - np.asarray(value))


def test_selective_advantage_can_pass_without_acquisition(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 3, .7)
    summary = audit.expected_summary(cfg, arrays, rows)
    assert summary["H_selective"]["supported"]
    assert not summary["H_selective"]["requires_acquisition"]
    assert not summary["H_acquisition"]["supported"]


def test_acquisition_can_pass_without_selective_advantage(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 3, .94)
    set_final_correct(arrays, 0, .90)
    set_final_correct(arrays, 1, .90)
    summary = audit.expected_summary(cfg, arrays, rows)
    assert summary["H_acquisition"]["supported"]
    assert not summary["H_selective"]["supported"]


def test_privileged_oracle_cannot_replace_feedback_primary(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 2, .99)
    summary = audit.expected_summary(cfg, arrays, rows)
    assert not summary["H_acquisition"]["supported"]
    assert not summary["H_selective"]["supported"]
    assert not summary["H_selective"]["oracle_can_replace_primary"]


@pytest.mark.parametrize("endpoint", [0, 1])
def test_primary_must_outperform_both_controls(independently_replayed, endpoint):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 3, .7)
    set_final_correct(arrays, endpoint, .69)
    assert not audit.expected_summary(cfg, arrays, rows)["H_selective"]["supported"]


def test_class_gate_survives_high_overall_accuracy(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 3, [.99, .99, .65, .65])
    summary = audit.expected_summary(cfg, arrays, rows)
    assert summary["H_selective"]["supported"]
    assert not summary["H_acquisition"]["supported"]


@pytest.mark.parametrize("field,value", [("trial_head_steps", .101), ("trial_cumulative_head_path_l2", .81),
    ("trial_cumulative_head_squared_l2", .081), ("trial_expected_head_steps", .2),
    ("trial_direction_residuals", .01)])
def test_nonprimary_numeric_failure_closes_all_positive_claims(independently_replayed, field, value):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 3, .95)
    arrays[field][2, 1, 2, 0, -1] = value
    summary = audit.expected_summary(cfg, arrays, rows)
    assert not summary["H_selective"]["supported"]
    assert not summary["H_acquisition"]["supported"]


@pytest.mark.parametrize("field", ["trial_routing_signs", "trial_known_after", "trial_memory_changed",
                                  "checkpoint_lookup_probabilities"])
def test_information_leak_or_reference_mismatch_closes_all_claims(independently_replayed, field):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 3, .95)
    flat = arrays[field].reshape(-1)
    flat[0] += .25 if flat.dtype.kind == "f" else 1
    summary = audit.expected_summary(cfg, arrays, rows)
    assert not all(summary["information_validation"].values())
    assert not summary["H_selective"]["supported"] and not summary["H_acquisition"]["supported"]


def test_scientific_summary_tampering_rejected(independently_replayed):
    cfg, arrays, rows = independently_replayed
    summary = audit.expected_summary(cfg, arrays, rows)
    audit.verify_against_graph(cfg, tiny_graph(), arrays, rows, summary, allow_fixture=True)
    summary["interpretation"]["feedback_uses_hidden_targets_for_routing"] = True
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, tiny_graph(), arrays, rows, summary, allow_fixture=True)


def test_large_mean_effect_does_not_replace_positive_seed_count(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 3, .8)
    set_final_correct(arrays, 1, .85, seed=1)
    summary = audit.expected_summary(cfg, arrays, rows)
    comparison = summary["H_selective"]["endpoint_comparisons"]["isolated"]
    assert comparison["mean_effect"] > .1 and comparison["positive_seeds"] == 1
    assert not comparison["practical_advantage_pass"] and not summary["H_selective"]["supported"]


def test_nonprimary_endpoint_beyond_path_closes_claims(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 3, .95)
    path = arrays["trial_cumulative_head_path_l2"][2, 1, 2, 0, -1]
    arrays["checkpoint_heads"][2, 1, 2, 0, -1] = 0.
    arrays["checkpoint_heads"][2, 1, 2, 0, -1, 0] = path + .01
    summary = audit.expected_summary(cfg, arrays, rows)
    assert not summary["budget_validation"]["head_endpoint_within_path_all"]
    assert not summary["H_selective"]["supported"] and not summary["H_acquisition"]["supported"]


@pytest.mark.parametrize("field,value", [("numeric_selective_qualified", False), ("preflight_direction_count", 323),
                                       ("preflight_max_residuals", .01), ("retained_rank", 3)])
def test_nonprimary_static_failure_closes_claims(independently_replayed, field, value):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 3, .95)
    arrays["g_ann_identity_" + field][-1] = value
    summary = audit.expected_summary(cfg, arrays, rows)
    assert not summary["budget_validation"]["all_static_numeric_gates_passed"]
    assert not summary["H_selective"]["supported"] and not summary["H_acquisition"]["supported"]


def test_direct_memory_policy_cannot_pass_by_preserving_action_uniform_rule(independently_replayed):
    cfg, original, _ = independently_replayed
    arrays = copy.deepcopy(original)
    arrays["trial_probabilities"][0, 0, 3] = .999
    arrays["trial_actions"][0, 0, 3] = arrays["uniforms"] < .999
    checks = audit.information_checks(cfg, arrays)
    assert not checks["head_policy_does_not_directly_lookup_memory"]


def test_unobserved_current_source_never_inherits_sign_from_nogo_reward(independently_replayed):
    cfg, original, _ = independently_replayed
    arrays = copy.deepcopy(original)
    choices = arrays["trial_actions"][0, 0, 3]
    before = arrays["trial_known_before"][0, 0, 3]
    candidates = []
    for si, schedule in enumerate(arrays["schedules"]):
        for t, source in enumerate(schedule):
            if choices[si, t] == 0 and before[si, t, source] == 0:
                candidates.append((si, t, source))
    assert candidates
    si, t, source = candidates[0]
    arrays["trial_known_after"][0, 0, 3, si, t, source] = audit.SIGNS[source]
    checks = audit.information_checks(cfg, arrays)
    assert not checks["NoGo_does_not_reveal_unknown_sign"]
    assert not checks["observed_memory_from_obtained_Go_rewards_only"]
