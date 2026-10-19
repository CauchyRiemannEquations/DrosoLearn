"""Invented matrix and short fixture verification; no research outcomes."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "spillover_independent_tests",
    Path(__file__).resolve().parents[1] / "scripts/verify_spillover_tradeoff.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def correlated_rows():
    return np.array([[1., 0., 0., 0.], [1., .05, 0., 0.],
                     [1., 0., .05, 0.], [1., 0., 0., .05]])


@pytest.mark.parametrize("rows", [np.eye(4), np.ones((4, 1)), correlated_rows(),
                                 np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 2., 3.]])])
def test_invented_helper_matches_independent_geometry(rows):
    from drosolearn.spillover_geometry import spillover_geometry
    actual, expected = spillover_geometry(rows), audit.independent_tradeoff_report(rows)
    def compare(a, b):
        assert a.keys() == b.keys()
        for key, value in b.items():
            if isinstance(value, dict): compare(a[key], value)
            elif isinstance(value, np.ndarray): assert np.allclose(a[key], value, atol=1e-9, rtol=1e-9), key
            elif isinstance(value, float): assert a[key] == pytest.approx(value, abs=1e-9), key
            else: assert a[key] == value, key
    compare(actual, expected)


def test_endpoints_exact_and_every_blend_unit_norm_and_analytic_formula():
    rows = correlated_rows()
    record = audit.independent_tradeoff_report(rows)
    endpoints = audit.budget.independent_budget_report(rows)
    assert np.array_equal(record["directions"][0], endpoints["directions"][0])
    assert np.array_equal(record["directions"][-1], endpoints["directions"][2])
    assert np.allclose(record["direction_norms"], 1.)
    assert np.all(record["mixture_norms"] ** 2 >= .5 - 1e-12)
    assert np.allclose(record["own_influence"], record["analytic_own_influence"])
    assert np.all(np.diff(record["own_influence"], axis=0) <= 1e-12)
    assert not record["alpha_is_fraction_of_spillover_removed"]
    assert not record["all_unit_directions_optimized"]
    assert record["numeric_spillover_qualified"]


@pytest.mark.parametrize("factor", [-.1, -.05, 0., .012, .05])
def test_same_counterfactual_factor_equalizes_step_length_only(factor):
    record = audit.independent_tradeoff_report(correlated_rows())
    assert np.allclose(np.linalg.norm(factor * record["directions"], axis=-1), abs(factor))
    assert not np.allclose(record["own_influence"][0], record["own_influence"][-1])


@pytest.mark.parametrize("alpha", audit.ALPHAS)
def test_observed_reward_actor_actual_cost_and_nogo_zero(alpha):
    actor = audit.IndependentHead(correlated_rows(), alpha)
    first = actor.observe(0, 1, 1)
    assert first["factor"] == .05
    assert first["head_step_l2"] == pytest.approx(.05)
    assert actor.path_l2 == first["head_step_l2"]
    assert actor.squared_l2 == first["head_step_squared_l2"]
    before = actor.head.copy()
    second = actor.observe(1, 0, 0)
    assert np.array_equal(actor.head, before) and second["head_step_l2"] == 0.
    assert actor.calls == 2
    with pytest.raises(ValueError): actor.observe(0, 0, -1)
    with pytest.raises(ValueError): actor.observe(4, 1, 1)


def test_autonomous_costs_need_not_match_across_unit_blends():
    a = audit.IndependentHead(correlated_rows(), 0.)
    b = audit.IndependentHead(correlated_rows(), .5)
    for index in (0, 1, 0, 2, 3, 1, 2, 0):
        a.observe(index, 1, 1 if index < 2 else -1)
        b.observe(index, 1, 1 if index < 2 else -1)
    assert a.path_l2 != pytest.approx(b.path_l2, abs=1e-8)
    assert a.squared_l2 != pytest.approx(b.squared_l2, abs=1e-8)


@pytest.mark.parametrize("rows", [np.zeros((4, 2)), np.ones((3, 2)), np.full((4, 2), np.nan)])
def test_invalid_geometry_rejected_without_fallback(rows):
    with pytest.raises(ValueError): audit.independent_tradeoff_report(rows)


def tiny_graph():
    n = 16
    return {"pre": np.arange(n), "post": np.arange(n), "counts": np.ones(n), "n_kc": n, "n_mbon": n}


def fixture_config():
    return {"study_id": "spillover_tradeoff_v1", "scope": "integration_test",
            "module": "drosolearn.spillover_tradeoff", "protocol": "docs/spillover-tradeoff-protocol.md",
            "topologies": list(audit.TOPOLOGIES), "representations": list(audit.INPUTS), "alphas": list(audit.ALPHAS),
            "primary_alpha": .5, "model_seeds": [90081, 90082], "training_trials": 8, "checkpoints": [0, 4, 8],
            "head_lr": .1, "rng_namespace": 202610195, "svd_rcond": 1e-12, "numeric_tolerance": 1e-8,
            "exact_condition_max": 1e6, "acquisition_threshold": .8, "acquisition_gain": .25,
            "class_threshold": .7, "population_pass_count": 2, "effect_threshold": .05, "positive_seed_count": 2,
            "normalization": "per_input_initial_fixed_MBON_gap_unit_L2", "input_catalog": audit.INPUT_CATALOG,
            "budget_convention": "all_mixture_directions_unit_L2_common_eta_step_ceiling",
            "graph": "data/flywire_783_mb_left_kc512_s701", "external_manifest": None}


@pytest.fixture
def independently_replayed():
    cfg = fixture_config()
    arrays, rows = audit.expected_artifacts(cfg, tiny_graph())
    return cfg, arrays, rows


def test_full_invented_replay_has_complete_alpha_axis_and_cost_bounds(independently_replayed):
    cfg, arrays, rows = independently_replayed
    assert arrays["trial_actions"].shape == (3, 2, 5, 2, 8)
    assert arrays["checkpoint_probabilities"].shape == (3, 2, 5, 2, 3, 4)
    assert np.array_equal(arrays["alphas"], audit.ALPHAS)
    assert len(rows) == 480
    assert arrays["trial_head_update_calls"][..., -1].sum() == 480
    assert np.max(arrays["trial_head_steps"]) <= .1 + 1e-12
    assert np.max(arrays["trial_cumulative_head_path_l2"][..., -1]) <= .8 + 1e-12
    assert np.allclose(np.cumsum(arrays["trial_head_steps"], axis=-1), arrays["trial_cumulative_head_path_l2"])
    assert np.allclose(np.cumsum(arrays["trial_head_steps"] ** 2, axis=-1), arrays["trial_cumulative_head_squared_l2"])
    audit.check_artifacts(arrays, rows, arrays, rows)


@pytest.mark.parametrize("key,value", [("head_lr", .2), ("rng_namespace", 99), ("svd_rcond", 1e-6),
                                       ("normalization", "old_rms"), ("effect_threshold", .01),
                                       ("class_threshold", .2), ("primary_alpha", .75), ("alphas", [0., .5, 1.])])
def test_fixed_preregistered_controls_rejected(key, value):
    cfg = fixture_config()
    cfg[key] = value
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


def test_invented_scope_cannot_open_research_execution():
    cfg = fixture_config()
    audit.validate_science(cfg, allow_fixture=True)
    with pytest.raises(ValueError): audit.validate_science(cfg)
    cfg["model_seeds"] = [17301]
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)
    cfg = fixture_config()
    cfg["training_trials"] = 400
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


@pytest.mark.parametrize("field", ["inputs", "targets", "schedules", "uniforms", "alphas", "trial_actions", "trial_probabilities",
                                    "trial_head_steps", "trial_update_factors", "trial_expected_head_steps",
                                    "trial_head_step_squared_l2", "trial_cumulative_head_path_l2",
                                    "trial_cumulative_head_squared_l2", "trial_head_norms", "checkpoint_heads",
                                    "w_real_raw_weights", "finalfixed_ann_identity_probe_kc",
                                    "t_real_raw_direction_norms", "t_rewired_identity_own_influence",
                                    "t_ann_raw_numeric_spillover_qualified", "c_real_raw_absolute_logit_ceiling",
                                    "c_ann_identity_correct_probability_ceiling", "t_real_identity_analytic_own_influence",
                                    "t_real_raw_endpoint_cosines", "t_rewired_raw_mixture_norms"])
def test_complete_state_geometry_cost_tampering_rejected(independently_replayed, field):
    _, original, rows = independently_replayed
    arrays = {key: value.copy() for key, value in original.items()}
    expected = {key: value.copy() for key, value in original.items()}
    flat = arrays[field].reshape(-1)
    if flat.dtype.kind == "b": flat[-1] = not flat[-1]
    else: flat[-1] += 1 if flat.dtype.kind in "iu" else .25
    with pytest.raises(ValueError): audit.check_artifacts(expected, rows, arrays, rows)


def test_csv_cost_or_field_tampering_rejected(independently_replayed):
    _, arrays, original = independently_replayed
    rows = copy.deepcopy(original)
    rows[0]["cumulative_head_path_l2"] += .01
    with pytest.raises(ValueError): audit.check_artifacts(arrays, original, arrays, rows)
    rows = copy.deepcopy(original)
    del rows[0]["head_step_squared_l2"]
    with pytest.raises(ValueError): audit.check_artifacts(arrays, original, arrays, rows)


def write_input_catalog(root):
    path = root / audit.INPUT_CATALOG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"study_id": "representation_update_v1", "representations": list(audit.INPUTS),
                               "catalog": {key: value.tolist() for key, value in audit.prior.independent_catalog_arrays().items()}}), encoding="utf8")


def test_new_producer_matches_complete_invented_artifacts(monkeypatch, tmp_path):
    from drosolearn import spillover_tradeoff as producer
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
    assert receipt["independently_replayed_training_actions"] == 480
    assert receipt["primary_midpoint_contrasted_against_both_endpoints_checked"]


def summary_gate_fixture(independently_replayed):
    """Invented probabilities isolate decision gates, not replay trajectories."""
    cfg, original, rows = independently_replayed
    arrays = copy.deepcopy(original)
    arrays["checkpoint_probabilities"][:] = .5
    for key in arrays:
        if key.startswith("c_") and key.endswith("correct_probability_ceiling"):
            arrays[key][:] = 1.
    return cfg, arrays, rows


def set_final_correct(arrays, alpha_index, value, seed=slice(None)):
    value = np.asarray(value)
    arrays["checkpoint_probabilities"][0, 0, alpha_index, seed, -1] = np.where(audit.TARGETS == 1, value, 1 - value)


def test_tradeoff_supported_without_acquisition_does_not_claim_learning(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 2, .7)
    decision = audit.expected_summary(cfg, arrays, rows)
    assert decision["H_tradeoff"]["supported"]
    assert not decision["H_tradeoff"]["requires_acquisition"]
    assert not decision["H_acquisition"]["supported"]
    assert decision["H_acquisition"]["acquisition_pass_seeds"] == 0


def test_acquisition_supported_without_tradeoff_advantage(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 0, .90)
    set_final_correct(arrays, 2, .94)
    set_final_correct(arrays, 4, .90)
    decision = audit.expected_summary(cfg, arrays, rows)
    assert decision["H_acquisition"]["supported"]
    assert not decision["H_acquisition"]["requires_tradeoff_advantage"]
    assert not decision["H_tradeoff"]["supported"]
    assert decision["H_acquisition"]["acquisition_pass_seeds"] == 2


@pytest.mark.parametrize("strong_endpoint", [0, 4])
def test_midpoint_must_outperform_each_endpoint(strong_endpoint, independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 2, .7)
    set_final_correct(arrays, strong_endpoint, .69)
    decision = audit.expected_summary(cfg, arrays, rows)["H_tradeoff"]
    assert not decision["supported"]
    endpoint = "shared" if strong_endpoint == 0 else "isolated"
    assert not decision["endpoint_comparisons"][endpoint]["practical_advantage_pass"]


def test_large_mean_effect_cannot_replace_positive_seed_gate(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 2, .8)
    set_final_correct(arrays, 4, .85, seed=1)
    decision = audit.expected_summary(cfg, arrays, rows)["H_tradeoff"]
    assert decision["endpoint_comparisons"]["isolated"]["mean_effect"] > .1
    assert decision["endpoint_comparisons"]["isolated"]["positive_seeds"] == 1
    assert not decision["supported"]


def test_class_gate_remains_even_with_high_overall_accuracy(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 2, [.99, .99, .65, .65])
    decision = audit.expected_summary(cfg, arrays, rows)
    assert decision["H_tradeoff"]["supported"]
    assert not decision["H_acquisition"]["supported"]
    assert decision["H_acquisition"]["acquisition_pass_seeds"] == 0


def test_no_best_observed_interior_replaces_predeclared_midpoint(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 1, .99)
    set_final_correct(arrays, 3, .99)
    decision = audit.expected_summary(cfg, arrays, rows)
    assert not decision["H_tradeoff"]["supported"]
    assert not decision["H_acquisition"]["supported"]
    assert decision["primary_alpha"] == .5


def test_nonprimary_failed_geometry_closes_both_claims(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 2, .95)
    arrays["t_ann_identity_numeric_spillover_qualified"][-1] = False
    decision = audit.expected_summary(cfg, arrays, rows)
    assert decision["H_acquisition"]["acquisition_pass_seeds"] == 2
    assert not decision["H_tradeoff"]["numeric_spillover_qualified_all"]
    assert not decision["H_tradeoff"]["supported"]
    assert not decision["H_acquisition"]["supported"]


@pytest.mark.parametrize("field,value", [("trial_head_steps", .101), ("trial_cumulative_head_path_l2", .81)])
def test_any_alpha_actual_budget_violation_closes_claims(independently_replayed, field, value):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 2, .95)
    arrays[field][2, 1, 3, 0, -1] = value
    decision = audit.expected_summary(cfg, arrays, rows)
    assert not decision["budget_validation"]["common_bounds_valid_all"]
    assert not decision["H_tradeoff"]["supported"] and not decision["H_acquisition"]["supported"]


def test_step_formula_mismatch_closes_both_hypotheses(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 2, .95)
    arrays["trial_expected_head_steps"][2, 1, 3, 0, -1] += .01
    decision = audit.expected_summary(cfg, arrays, rows)
    assert decision["budget_validation"]["common_bounds_valid_all"]
    assert not decision["H_tradeoff"]["supported"] and not decision["H_acquisition"]["supported"]


def test_probability_ceiling_violation_closes_both_hypotheses(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 2, .95)
    arrays["c_real_raw_correct_probability_ceiling"][:, 2] = .9
    decision = audit.expected_summary(cfg, arrays, rows)
    assert not decision["budget_validation"]["analytic_probability_ceiling_valid_all"]
    assert not decision["H_tradeoff"]["supported"] and not decision["H_acquisition"]["supported"]


def test_scientific_summary_tampering_rejected(independently_replayed):
    cfg, arrays, rows = independently_replayed
    summary = audit.expected_summary(cfg, arrays, rows)
    audit.verify_against_graph(cfg, tiny_graph(), arrays, rows, summary, allow_fixture=True)
    summary = copy.deepcopy(summary)
    summary["interpretation"]["posthoc_best_alpha_is_confirmatory"] = True
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, tiny_graph(), arrays, rows, summary, allow_fixture=True)


def test_nonprimary_energy_ceiling_violation_closes_both_hypotheses(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 2, .95)
    arrays["trial_cumulative_head_squared_l2"][2, 1, 1, 0, -1] = cfg["training_trials"] * cfg["head_lr"] ** 2 + .01
    decision = audit.expected_summary(cfg, arrays, rows)
    assert not decision["budget_validation"]["squared_step_sum_valid_all"]
    assert not decision["budget_validation"]["common_bounds_valid_all"]
    assert not decision["H_tradeoff"]["supported"] and not decision["H_acquisition"]["supported"]


def test_nonprimary_endpoint_exceeding_actual_path_closes_both_hypotheses(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    set_final_correct(arrays, 2, .95)
    actual_path = arrays["trial_cumulative_head_path_l2"][2, 1, 3, 0, -1]
    arrays["checkpoint_heads"][2, 1, 3, 0, -1, :] = 0.
    arrays["checkpoint_heads"][2, 1, 3, 0, -1, 0] = actual_path + .01
    decision = audit.expected_summary(cfg, arrays, rows)
    assert not decision["budget_validation"]["head_endpoint_within_path_all"]
    assert not decision["budget_validation"]["common_bounds_valid_all"]
    assert not decision["H_tradeoff"]["supported"] and not decision["H_acquisition"]["supported"]
