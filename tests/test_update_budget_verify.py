"""Invented-only independent cost-control algebra; no research seed draws."""
import importlib.util
import copy
import json
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "update_budget_independent_test",
    Path(__file__).resolve().parents[1] / "scripts/verify_update_budget.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def correlated_rows():
    return np.array([[1., 0., 0., 0.], [1., .05, 0., 0.],
                     [1., 0., .05, 0.], [1., 0., 0., .05]])


def test_unit_projected_and_shared_have_equal_conditional_l2():
    g = audit.independent_directions(correlated_rows())
    assert g["conditional_l2_residual"] < 1e-12
    assert np.allclose(np.linalg.norm(g["shared"], axis=1), 1.)
    assert np.allclose(np.linalg.norm(g["projected_l2"], axis=1), 1.)
    assert np.min(np.linalg.norm(g["projected_own"], axis=1)) > 10.
    assert np.allclose(g["projected_own"] @ g["shared"].T, np.eye(4))
    assert np.allclose(np.diag(g["projected_l2"] @ g["shared"].T), 1. / g["projected_own_norms"])
    assert np.max(g["projected_l2_own_effect"]) < .1


@pytest.mark.parametrize("scale", [-.1, -.05, 0., .012, .05])
def test_counterfactual_same_scalar_step_matches_size_not_own_effect(scale):
    g = audit.independent_directions(correlated_rows())
    assert np.allclose(np.linalg.norm(scale * g["shared"], axis=1),
                       np.linalg.norm(scale * g["projected_l2"], axis=1))
    own = np.sum((scale * g["projected_l2"]) * g["shared"], axis=1)
    assert np.allclose(own, scale / g["projected_own_norms"])


@pytest.mark.parametrize("rule", audit.UPDATES)
def test_reward_only_actor_actual_new_minus_old_cost_and_zero_nogo(rule):
    actor = audit.IndependentHead(correlated_rows(), rule)
    first = actor.observe(0, 1, 1)
    assert first["factor"] == .05
    assert actor.calls == 1 and actor.path_l2 == first["head_step_l2"]
    assert actor.squared_l2 == first["head_step_squared_l2"]
    unchanged = actor.head.copy()
    second = actor.observe(1, 0, 0)
    assert second["head_step_l2"] == 0 and actor.calls == 2
    assert np.array_equal(actor.head, unchanged)
    with pytest.raises(ValueError): actor.observe(0, 0, -1)
    with pytest.raises(ValueError): actor.observe(4, 1, 1)


def test_conditional_matching_does_not_equal_actual_autonomous_path_cost():
    rows = correlated_rows()
    a = audit.IndependentHead(rows, "shared")
    b = audit.IndependentHead(rows, "projected_l2")
    for index in (0, 1, 0, 2, 3, 1, 2, 0):
        a.observe(index, 1, 1 if index < 2 else -1)
        b.observe(index, 1, 1 if index < 2 else -1)
    assert a.path_l2 != pytest.approx(b.path_l2, abs=1e-8)
    assert a.squared_l2 != pytest.approx(b.squared_l2, abs=1e-8)
    assert np.linalg.norm(a.head) <= a.path_l2 + 1e-12
    assert np.linalg.norm(b.head) <= b.path_l2 + 1e-12


@pytest.mark.parametrize("rows", [np.zeros((4, 2)), np.ones((3, 2)), np.full((4, 2), np.nan)])
def test_invalid_geometry_rejected_without_cost_fallback(rows):
    with pytest.raises(ValueError): audit.independent_directions(rows)


@pytest.mark.parametrize("rows", [np.eye(4), np.ones((4, 1)), correlated_rows(),
                                 np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 2., 3.]])])
def test_current_helper_matches_independent_budget_geometry_on_invented_cases(rows):
    from drosolearn.update_budget_geometry import update_budget_geometry
    actual, expected = update_budget_geometry(rows), audit.independent_budget_report(rows)
    def compare(a, b):
        assert a.keys() == b.keys()
        for key, value in b.items():
            if isinstance(value, dict): compare(a[key], value)
            elif isinstance(value, np.ndarray): assert np.allclose(a[key], value, atol=1e-9, rtol=1e-9), key
            elif isinstance(value, float): assert a[key] == pytest.approx(value, abs=1e-9), key
            else: assert a[key] == value, key
    compare(actual, expected)


def tiny_graph():
    n = 16
    return {"pre": np.arange(n), "post": np.arange(n), "counts": np.ones(n), "n_kc": n, "n_mbon": n}


def fixture_config():
    return {"study_id": "update_budget_v1", "scope": "integration_test",
            "module": "drosolearn.update_budget", "protocol": "docs/update-budget-protocol.md",
            "topologies": list(audit.TOPOLOGIES), "representations": list(audit.INPUTS), "updates": list(audit.UPDATES),
            "model_seeds": [90081, 90082], "training_trials": 8, "checkpoints": [0, 4, 8],
            "head_lr": .1, "rng_namespace": 202610194, "svd_rcond": 1e-12, "numeric_tolerance": 1e-8,
            "exact_condition_max": 1e6, "acquisition_threshold": .8, "acquisition_gain": .25,
            "class_threshold": .7, "population_pass_count": 2, "effect_threshold": .1, "positive_seed_count": 2,
            "normalization": "per_input_initial_fixed_MBON_gap_unit_L2", "input_catalog": audit.INPUT_CATALOG,
            "budget_convention": "unit_direction_equal_conditional_L2_and_common_eta_step_ceiling",
            "graph": "data/flywire_783_mb_left_kc512_s701", "external_manifest": None}


@pytest.fixture
def independently_replayed():
    cfg = fixture_config()
    arrays, rows = audit.expected_artifacts(cfg, tiny_graph())
    return cfg, arrays, rows


def test_full_independent_fixture_counts_and_actual_unit_cost_bounds(independently_replayed):
    cfg, arrays, rows = independently_replayed
    assert arrays["trial_actions"].shape == (3, 2, 3, 2, 8)
    assert arrays["checkpoint_probabilities"].shape == (3, 2, 3, 2, 3, 4)
    assert len(rows) == 288
    assert arrays["trial_head_update_calls"][..., -1].sum() == 288
    unit = arrays["trial_head_steps"][:, :, [0, 2]]
    assert np.max(unit) <= .1 + 1e-12
    assert np.max(arrays["trial_cumulative_head_path_l2"][:, :, [0, 2], :, -1]) <= .8 + 1e-12
    assert np.allclose(np.cumsum(arrays["trial_head_steps"], axis=-1), arrays["trial_cumulative_head_path_l2"])
    assert np.allclose(np.cumsum(arrays["trial_head_steps"] ** 2, axis=-1), arrays["trial_cumulative_head_squared_l2"])
    assert np.allclose(arrays["trial_head_steps"], arrays["trial_expected_head_steps"])
    audit.check_artifacts(arrays, rows, arrays, rows)


@pytest.mark.parametrize("field", ["inputs", "targets", "schedules", "uniforms", "trial_actions", "trial_probabilities",
                                    "trial_head_steps", "trial_update_factors", "trial_expected_head_steps",
                                    "trial_head_step_squared_l2", "trial_cumulative_head_path_l2",
                                    "trial_cumulative_head_squared_l2", "trial_head_norms", "checkpoint_heads",
                                    "w_real_raw_weights", "finalfixed_ann_identity_probe_kc",
                                    "b_real_raw_direction_norms", "b_rewired_identity_own_influence",
                                    "b_ann_raw_numeric_budget_qualified", "c_real_raw_absolute_logit_ceiling",
                                    "c_ann_identity_correct_probability_ceiling"])
def test_all_cost_state_and_geometry_tampering_rejected(independently_replayed, field):
    _, original, rows = independently_replayed
    arrays = {key: value.copy() for key, value in original.items()}
    expected = {key: value.copy() for key, value in original.items()}
    flat = arrays[field].reshape(-1)
    if flat.dtype.kind == "b": flat[-1] = not flat[-1]
    else: flat[-1] += 1 if flat.dtype.kind in "iu" else .25
    with pytest.raises(ValueError): audit.check_artifacts(expected, rows, arrays, rows)


def test_csv_recorded_cost_or_missing_field_rejected(independently_replayed):
    _, arrays, original = independently_replayed
    rows = copy.deepcopy(original)
    rows[0]["cumulative_head_path_l2"] += .01
    with pytest.raises(ValueError): audit.check_artifacts(arrays, original, arrays, rows)
    rows = copy.deepcopy(original)
    del rows[0]["head_step_squared_l2"]
    with pytest.raises(ValueError): audit.check_artifacts(arrays, original, arrays, rows)


@pytest.mark.parametrize("key,value", [("head_lr", .2), ("rng_namespace", 99), ("svd_rcond", 1e-6),
                                       ("normalization", "old_rms"), ("effect_threshold", .01),
                                       ("class_threshold", .2), ("updates", ["shared", "projected_own"])])
def test_fixed_independent_config_controls_rejected(key, value):
    cfg = fixture_config()
    cfg[key] = value
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


def test_fixture_budget_and_scientific_seeds_cannot_open_science():
    cfg = fixture_config()
    audit.validate_science(cfg, allow_fixture=True)
    with pytest.raises(ValueError): audit.validate_science(cfg)
    cfg["model_seeds"] = [17201]
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)
    cfg = fixture_config()
    cfg["training_trials"] = 400
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


def write_input_catalog(root):
    path = root / audit.INPUT_CATALOG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"study_id": "representation_update_v1", "representations": list(audit.INPUTS),
                               "catalog": {key: value.tolist() for key, value in audit.prior.independent_catalog_arrays().items()}}), encoding="utf8")


def test_new_producer_matches_full_independent_invented_graph_arithmetic(monkeypatch, tmp_path):
    from drosolearn import update_budget as producer
    from drosolearn.graph import Projection
    raw = tiny_graph()
    n = raw["n_kc"]
    graph = Projection(raw["pre"], raw["post"], raw["counts"], n, n,
                       [str(i) for i in range(n)], [str(i) for i in range(n)], {})
    monkeypatch.setattr(producer, "load_projection", lambda _: graph)
    cfg = fixture_config()
    write_input_catalog(tmp_path)
    arrays, rows = producer.produce(cfg, tmp_path)
    summary = producer.summarize(cfg, arrays, rows)
    result = audit.verify_against_graph(cfg, raw, arrays, rows, summary, allow_fixture=True)
    assert result["independently_replayed_training_actions"] == 288
    assert result["all_actual_step_path_energy_and_head_norms_replayed"]


def test_summary_claim_tampering_is_rejected(independently_replayed):
    cfg, arrays, rows = independently_replayed
    summary = audit.expected_summary(cfg, arrays, rows)
    audit.verify_against_graph(cfg, tiny_graph(), arrays, rows, summary, allow_fixture=True)
    summary = copy.deepcopy(summary)
    summary["budget_validation"]["actual_cumulative_cost_equalized"] = True
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, tiny_graph(), arrays, rows, summary, allow_fixture=True)


def summary_gate_fixture(independently_replayed):
    """Synthetic metrics to isolate decision gates, not a replayed trajectory."""
    cfg, original, rows = independently_replayed
    arrays = copy.deepcopy(original)
    arrays["checkpoint_probabilities"][:] = .5
    arrays["checkpoint_probabilities"][0, 0, 2, :, -1, :2] = .95
    arrays["checkpoint_probabilities"][0, 0, 2, :, -1, 2:] = .05
    for key in arrays:
        if key.startswith("c_") and key.endswith("correct_probability_ceiling"):
            arrays[key][:] = 1.
    assert audit.expected_summary(cfg, arrays, rows)["H_budget"]["supported"]
    return cfg, arrays, rows


def test_mean_success_does_not_override_class_acquisition_gate(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    arrays["checkpoint_probabilities"][0, 0, 2, :, -1, :2] = .99
    arrays["checkpoint_probabilities"][0, 0, 2, :, -1, 2:] = .35
    decision = audit.expected_summary(cfg, arrays, rows)["H_budget"]
    assert decision["mean_effect"] > .3
    assert decision["treated_acquisition_pass_seeds"] == 0
    assert not decision["supported"]


def test_nonprimary_invalid_geometry_closes_primary_claim(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    arrays["b_ann_identity_numeric_budget_qualified"][-1] = False
    decision = audit.expected_summary(cfg, arrays, rows)["H_budget"]
    assert decision["treated_acquisition_pass_seeds"] == 2
    assert not decision["numeric_budget_qualified_all"] and not decision["supported"]


@pytest.mark.parametrize("field,value", [("trial_head_steps", .101), ("trial_cumulative_head_path_l2", .81)])
def test_actual_step_or_path_bound_violation_closes_claim(independently_replayed, field, value):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    arrays[field][2, 1, 2, 0, -1] = value
    decision = audit.expected_summary(cfg, arrays, rows)
    assert not decision["budget_validation"]["common_bounds_valid_all"]
    assert not decision["H_budget"]["supported"]


def test_conservative_probability_bound_violation_closes_claim(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    arrays["c_real_raw_correct_probability_ceiling"][:, 2] = .9
    decision = audit.expected_summary(cfg, arrays, rows)
    assert not decision["budget_validation"]["analytic_probability_ceiling_valid_all"]
    assert not decision["H_budget"]["supported"]


def test_effect_size_gate_not_only_acquisition(independently_replayed):
    cfg, arrays, rows = summary_gate_fixture(independently_replayed)
    arrays["checkpoint_probabilities"][0, 0, 0, :, -1, :2] = .90
    arrays["checkpoint_probabilities"][0, 0, 0, :, -1, 2:] = .10
    decision = audit.expected_summary(cfg, arrays, rows)["H_budget"]
    assert decision["mean_effect"] == pytest.approx(.05)
    assert decision["positive_seeds"] == decision["treated_acquisition_pass_seeds"] == 2
    assert not decision["supported"]
