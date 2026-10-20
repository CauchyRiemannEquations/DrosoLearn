"""Invented-only independent checks for prospective memory-gated routing."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("self_blend_independent_tests",
    Path(__file__).resolve().parents[1] / "scripts/verify_self_blend_learning.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def mix_fixture():
    return np.eye(4), np.array([1, 1, -1, -1], np.int8), np.array([1., 0., 0., 0.]), np.array([.5, .5, -.5, -.5])


def test_fixed_mix_is_normalized_direction_mix_and_keeps_unit_influence_floors():
    d, observed, isolated, balanced = mix_fixture()
    result = audit.independent_mix(d, 0, observed, isolated, balanced)
    raw = .75 * isolated + .25 * balanced
    np.testing.assert_allclose(result["direction"], raw / np.linalg.norm(raw), atol=1e-14)
    assert np.linalg.norm(result["direction"]) == pytest.approx(1.)
    assert result["influence"][0] >= .75 * (d @ isolated)[0]
    aligned_other = observed[1:] * result["influence"][1:]
    assert np.all(aligned_other >= .25 * observed[1:] * (d @ balanced)[1:])
    assert result["help_floor_residual"] == result["own_floor_residual"] == result["safety_residual"] == 0.


def test_identical_ingredients_do_not_change_direction_or_norm():
    d, observed, isolated, _ = mix_fixture()
    result = audit.independent_mix(d, 0, observed, isolated, isolated)
    np.testing.assert_array_equal(result["direction"], isolated)
    assert result["raw_norm"] == result["normalization_scale"] == 1.


@pytest.mark.parametrize("bad", [np.zeros(4, np.int8), np.array([1, 1, -1, 0], np.int8), np.array([1., 1., -1., -1.])])
def test_mix_never_silently_fills_missing_reward_signs(bad):
    d, _, isolated, balanced = mix_fixture()
    with pytest.raises(ValueError): audit.independent_mix(d, 0, bad, isolated, balanced)


@pytest.mark.parametrize("ingredient", ["isolated", "balanced"])
def test_mix_does_not_normalize_invalid_ingredient_before_checking(ingredient):
    d, observed, isolated, balanced = mix_fixture()
    if ingredient == "isolated": isolated = 2 * isolated
    else: balanced = 2 * balanced
    with pytest.raises(ValueError): audit.independent_mix(d, 0, observed, isolated, balanced)


def test_mix_rejects_biologically_unobserved_or_harmful_balanced_direction():
    d, observed, isolated, balanced = mix_fixture()
    balanced[2] *= -1
    with pytest.raises(ValueError): audit.independent_mix(d, 0, observed, isolated, balanced)


def test_unit_influence_floor_is_not_a_probability_or_actual_movement_floor():
    d, observed, isolated, balanced = mix_fixture()
    mix = audit.independent_mix(d, 0, observed, isolated, balanced)["direction"]
    assert np.all(observed[1:] * mix[1:] >= .25 * observed[1:] * balanced[1:])
    factual_scalar, another_scalar = .001, .1
    assert np.all(np.abs(factual_scalar * mix[1:]) < .25 * np.abs(another_scalar * balanced[1:]))


def test_gate_starts_isolated_with_empty_observed_memory():
    result = audit.independent_gate(np.zeros(4, np.int8), "gated_balanced")
    assert result == {"full_known_before": False, "mode": "isolated", "balanced_branch": False}


def test_fourth_discovery_cannot_retroactively_change_its_own_update_mode():
    before = np.array([1, 1, -1, 0], np.int8)
    after = np.array([1, 1, -1, -1], np.int8)
    assert audit.independent_gate(before, "gated_balanced")["mode"] == "isolated"
    assert audit.independent_gate(after, "gated_balanced")["mode"] == "balanced"


@pytest.mark.parametrize("method", audit.METHODS[:4])
@pytest.mark.parametrize("known", [np.zeros(4, np.int8), np.array([1, 1, -1, -1], np.int8)])
def test_other_registered_methods_do_not_acquire_an_unrequested_gate(method, known):
    assert audit.independent_gate(known, method)["mode"] == method


@pytest.mark.parametrize("known", [np.zeros(3, int), np.zeros(4, float), np.array([1, 2, 0, 0])])
def test_gate_does_not_silently_complete_or_coerce_observation_table(known):
    with pytest.raises(ValueError): audit.independent_gate(known, "gated_balanced")


def test_unregistered_gate_method_rejected():
    with pytest.raises(ValueError): audit.independent_gate(np.zeros(4, int), "oracle_gated")


def fixture_config():
    from drosolearn.self_blend_learning import SCIENCE
    cfg = copy.deepcopy(SCIENCE)
    cfg.update(scope="integration_test", model_seeds=[90085, 90086], training_trials=16,
        checkpoints=[0, 4, 8, 16], population_pass_count=2, positive_seed_count=2, protection_nonworse_seed_count=2)
    return cfg


def tiny_graph():
    n = 16
    return {"pre": np.arange(n), "post": np.arange(n), "counts": np.ones(n), "n_kc": n, "n_mbon": n}


@pytest.fixture(scope="module")
def producer_fixture(tmp_path_factory):
    from drosolearn import self_blend_learning as producer
    from drosolearn.graph import Projection
    cfg, raw = fixture_config(), tiny_graph()
    root = tmp_path_factory.mktemp("gated_invented_replay")
    n = raw["n_kc"]
    graph = Projection(raw["pre"], raw["post"], raw["counts"], n, n,
        [str(i) for i in range(n)], [str(i) for i in range(n)], {})
    catalog = root / audit.INPUT_CATALOG
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({"study_id": "representation_update_v1", "representations": list(audit.REPRESENTATIONS),
        "catalog": {key: value.tolist() for key, value in audit.prior.independent_catalog_arrays().items()}}), encoding="utf8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(producer, "load_projection", lambda _: graph)
        arrays, rows = producer.produce(cfg, root)
    expected, wanted_rows = audit.expected_artifacts(cfg, raw, arrays)
    return cfg, raw, arrays, rows, expected, wanted_rows


def test_current_five_method_producer_matches_independent_fresh_replay(producer_fixture):
    cfg, _, arrays, rows, expected, wanted_rows = producer_fixture
    audit.validate_science(cfg, allow_fixture=True)
    audit.check_artifacts(arrays, expected, rows, wanted_rows)
    assert len(arrays) == 201 and len(rows) == 1152 and len(rows[0]) == 87
    assert arrays["trial_actions"].shape == (3, 2, 6, 2, 16)
    assert arrays["case_keys"].shape[1] == 9
    assert all(audit.information_checks(cfg, arrays).values())
    assert all(audit.gate_checks(cfg, arrays).values())


def test_complete_current_summary_matches_independent_five_method_summary(producer_fixture):
    from drosolearn.self_blend_learning import summarize
    cfg, graph, arrays, rows, _, _ = producer_fixture
    summary = summarize(cfg, arrays, rows)
    receipt = audit.verify_against_graph(cfg, graph, arrays, rows, summary, allow_fixture=True)
    assert receipt["independently_replayed_training_actions"] == 1152
    assert receipt["conditions"] == 36
    assert len(summary["families"]) == 36
    assert summary["counts"]["model_initializations"] == 72


def test_discovery_trial_and_later_gate_semantics_are_explicit_in_saved_fixture(producer_fixture):
    cfg, _, arrays, _, _, _ = producer_fixture
    n = cfg["training_trials"]
    assert arrays["trial_gate_switch_events"][:, :, 4].any()
    assert np.all(arrays["first_balanced_mode_trials"][:, :, :4] == n + 1)
    assert np.array_equal(arrays["first_discovery_trials"][:, :, 4], arrays["first_discovery_trials"][:, :, 1])
    for ti in range(3):
        for ri in range(2):
            for si in range(2):
                g = (ti, ri, 4, si)
                discovery = int(arrays["first_discovery_trials"][g].max())
                if discovery <= n:
                    assert arrays["trial_direction_modes"][g + (discovery - 1,)] == 1
                if discovery < n:
                    assert arrays["trial_direction_modes"][g + (discovery,)] == 3
                    assert arrays["trial_gate_switch_events"][g + (discovery,)] == 1
    partial_nogo = (arrays["trial_direction_modes"][:, :, 4] == 1) & (arrays["trial_actions"][:, :, 4] == 0)
    full_nogo = (arrays["trial_direction_modes"][:, :, 4] == 3) & (arrays["trial_actions"][:, :, 4] == 0)
    assert partial_nogo.any() and full_nogo.any()
    assert np.allclose(np.linalg.norm(arrays["trial_directions"][:, :, 4][partial_nogo], axis=-1), 1.)
    assert not arrays["trial_directions"][:, :, 4][full_nogo].any()


def test_balanced_and_gated_solver_caches_have_separate_method_identity(producer_fixture):
    _, _, arrays, _, _, _ = producer_fixture
    keys = arrays["case_keys"]
    assert set(keys[:, 3]) == {3, 4, 5}
    assert np.all(np.isin(keys[np.isin(keys[:, 3], (4, 5)), 5:], [-1, 1]))
    assert len({tuple(row) for row in keys}) == len(keys)
    without_method = [tuple(np.delete(row, 3)) for row in keys]
    assert len(set(without_method)) < len(keys)


@pytest.mark.parametrize("field", ["inputs", "targets", "schedules", "uniforms", "trial_actions", "trial_rewards",
    "trial_probabilities", "trial_known_before", "trial_known_after", "trial_routing_signs", "trial_harm_mask",
    "trial_desired_influence", "trial_actual_influence", "trial_directions", "trial_direction_residuals",
    "trial_direction_residuals_applicable", "trial_case_indices", "trial_direction_modes", "trial_gate_known_counts_before",
    "trial_blend_applicable", "trial_blend_isolated_directions", "trial_blend_balanced_directions", "trial_blend_raw_norm",
    "trial_blend_residuals", "first_blend_mode_trials",
    "trial_gate_full_before", "trial_gate_switch_events", "first_balanced_mode_trials", "trial_unknown_other_mask_before",
    "trial_unknown_other_mask_after", "trial_correct_probability_delta", "trial_aligned_margin_delta",
    "trial_unknown_other_negative_correct_probability_sum", "trial_memory_changed", "trial_memory_observation_calls",
    "first_discovery_trials", "checkpoint_known", "checkpoint_heads", "checkpoint_lookup_probabilities",
    "g_real_raw_unit_rows", "g_real_raw_preflight_direction_count", "w_real_raw_weights", "finalfixed_ann_identity_probe_mbon"])
def test_complete_gated_trajectory_corruption_is_rejected(producer_fixture, field):
    _, _, original, rows, wanted, wanted_rows = producer_fixture
    arrays = copy.deepcopy(original)
    flat = arrays[field].reshape(-1)
    flat[-1] = not flat[-1] if flat.dtype.kind == "b" else flat[-1] + (1 if flat.dtype.kind in "iu" else .03)
    with pytest.raises(ValueError): audit.check_artifacts(arrays, wanted, rows, wanted_rows)


@pytest.mark.parametrize("field", ["case_keys", "case_directions", "case_projections", "case_multipliers", "case_certificates"])
def test_gated_case_certificate_corruption_is_rejected_by_reoptimization(producer_fixture, field):
    cfg, graph, original, _, _, _ = producer_fixture
    arrays = copy.deepcopy(original)
    arrays[field].reshape(-1)[-1] += 1 if arrays[field].dtype.kind in "iu" else .1
    with pytest.raises(ValueError): audit.expected_artifacts(cfg, graph, arrays)


def test_gate_flags_detect_fourth_discovery_switched_too_early(producer_fixture):
    cfg, _, original, _, _, _ = producer_fixture
    arrays = copy.deepcopy(original)
    for ti in range(3):
        for ri in range(2):
            for si in range(2):
                at = (ti, ri, 4, si)
                trial = int(arrays["first_discovery_trials"][at].max())
                if trial <= cfg["training_trials"]:
                    arrays["trial_direction_modes"][at + (trial - 1,)] = 3
                    flags = audit.gate_checks(cfg, arrays)
                    assert not flags["fourth_discovery_update_is_isolated"]
                    assert not flags["mode_uses_known_before_only"]
                    return
    raise AssertionError("Invented fixture must exercise a fourth discovery")


def isolated_decision_fixture(producer_fixture, monkeypatch):
    """Test endpoint formulas alone; no fabricated trajectory is accepted."""
    cfg, _, original, rows, _, _ = producer_fixture
    info, gates = audit.information_checks(cfg, original), audit.gate_checks(cfg, original)
    monkeypatch.setattr(audit, "information_checks", lambda *_: info)
    monkeypatch.setattr(audit, "gate_checks", lambda *_: gates)
    arrays = copy.deepcopy(original)
    arrays["checkpoint_probabilities"][:] = .5
    arrays["trial_unknown_other_negative_correct_probability_sum"][:] = 0.
    return cfg, arrays, rows


def set_final(arrays, method, value, seed=slice(None)):
    arrays["checkpoint_probabilities"][0, 0, method, seed, -1] = np.where(audit.TARGETS, value, 1. - np.asarray(value))


def test_gain_and_acquisition_have_separate_gates(producer_fixture, monkeypatch):
    cfg, arrays, rows = isolated_decision_fixture(producer_fixture, monkeypatch)
    set_final(arrays, 5, .7)
    summary = audit.expected_summary(cfg, arrays, rows)
    assert summary["H_gain"]["supported"] and not summary["H_acquisition"]["supported"]
    for control in (1, 3, 4): set_final(arrays, control, .90)
    set_final(arrays, 5, .94)
    summary = audit.expected_summary(cfg, arrays, rows)
    assert summary["H_acquisition"]["supported"] and not summary["H_gain"]["supported"]


@pytest.mark.parametrize("control", [1, 3, 4])
def test_gain_requires_all_three_frozen_comparators(producer_fixture, monkeypatch, control):
    cfg, arrays, rows = isolated_decision_fixture(producer_fixture, monkeypatch)
    set_final(arrays, 5, .7)
    set_final(arrays, control, .69)
    assert not audit.expected_summary(cfg, arrays, rows)["H_gain"]["supported"]


def test_large_mean_effect_does_not_replace_positive_seed_count(producer_fixture, monkeypatch):
    cfg, arrays, rows = isolated_decision_fixture(producer_fixture, monkeypatch)
    set_final(arrays, 5, .8)
    set_final(arrays, 3, .85, seed=1)
    summary = audit.expected_summary(cfg, arrays, rows)
    comparison = summary["H_gain"]["endpoint_comparisons"]["balanced"]
    assert comparison["mean_effect"] > .1 and comparison["positive_seeds"] == 1
    assert not summary["H_gain"]["supported"]


def test_operational_protection_does_not_demonstrate_learning_or_a_new_mechanism(producer_fixture, monkeypatch):
    cfg, arrays, rows = isolated_decision_fixture(producer_fixture, monkeypatch)
    summary = audit.expected_summary(cfg, arrays, rows)
    assert summary["H_protection"]["supported"]
    assert summary["H_protection"]["operational_protection_expected_by_design"]
    assert not summary["H_protection"]["novel_learning_mechanism_demonstrated"]
    assert not summary["H_gain"]["supported"] and not summary["H_acquisition"]["supported"]


def test_protection_uses_maximum_seed_damage_not_a_small_mean(producer_fixture, monkeypatch):
    cfg, arrays, rows = isolated_decision_fixture(producer_fixture, monkeypatch)
    arrays["trial_unknown_other_negative_correct_probability_sum"][0, 0, 5, 0, 0] = 1.5e-8
    arrays["trial_unknown_other_negative_correct_probability_sum"][0, 0, (1, 4), :, 0] = 1e-6
    summary = audit.expected_summary(cfg, arrays, rows)
    assert arrays["trial_unknown_other_negative_correct_probability_sum"][0, 0, 5].sum(-1).mean() < 1e-8
    assert summary["H_protection"]["maximum_primary_seed_unknown_damage"] == pytest.approx(1.5e-8)
    assert all(value["all_seeds_nonworsening"] for value in summary["H_protection"]["comparisons"].values())
    assert not summary["H_protection"]["supported"]


def test_original_balanced_primary_cannot_replace_new_gated_primary(producer_fixture, monkeypatch):
    cfg, arrays, rows = isolated_decision_fixture(producer_fixture, monkeypatch)
    set_final(arrays, 3, .99)
    summary = audit.expected_summary(cfg, arrays, rows)
    assert not summary["H_gain"]["supported"] and not summary["H_acquisition"]["supported"]


@pytest.mark.parametrize("field,value", [("trial_head_steps", .101), ("trial_cumulative_head_path_l2", 1.61),
    ("trial_cumulative_head_squared_l2", .161), ("trial_expected_head_steps", .2), ("trial_direction_residuals", .01)])
def test_nonprimary_numeric_failure_is_fatal(producer_fixture, field, value):
    cfg, _, original, rows, _, _ = producer_fixture
    arrays = copy.deepcopy(original)
    arrays[field][2, 1, 2, 0, -1] = value
    with pytest.raises(ValueError): audit.expected_summary(cfg, arrays, rows)


@pytest.mark.parametrize("field", ["trial_direction_modes", "trial_gate_known_counts_before", "trial_gate_full_before",
    "trial_gate_switch_events", "first_balanced_mode_trials", "first_blend_mode_trials"])
def test_gate_integrity_failure_is_fatal_before_scientific_claims(producer_fixture, field):
    cfg, _, original, rows, _, _ = producer_fixture
    arrays = copy.deepcopy(original)
    flat = arrays[field].reshape(-1)
    flat[-1] = not flat[-1] if flat.dtype.kind == "b" else flat[-1] + 1
    with pytest.raises(ValueError): audit.expected_summary(cfg, arrays, rows)


@pytest.mark.parametrize("key,value", [("gate_switch_rule", "known_after_four"), ("protection_absolute_tolerance", .01),
    ("head_lr", .2), ("rng_namespace", 202610201), ("effect_threshold", .001), ("methods", ["gated_balanced"]),
    ("blend_isolated_weight", .5), ("blend_balanced_weight", .5)])
def test_fixed_science_and_gate_parameters_cannot_be_relaxed(key, value):
    cfg = fixture_config()
    cfg[key] = value
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


def test_fixture_scope_cannot_enter_actual_research():
    cfg = fixture_config()
    audit.validate_science(cfg, allow_fixture=True)
    with pytest.raises(ValueError): audit.validate_science(cfg)
    cfg["model_seeds"] = [17701]
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


def test_new_gate_uses_before_memory_and_has_a_distinct_post_mode():
    before = np.array([1, 1, -1, 0], np.int8)
    after = np.array([1, 1, -1, -1], np.int8)
    assert audit.independent_gate(before, "gated_self_blend")["mode"] == "isolated"
    result = audit.independent_gate(after, "gated_self_blend")
    assert result == {"full_known_before": True, "mode": "blend", "balanced_branch": True}


def test_saved_new_discovery_update_stays_isolated_then_switches_once(producer_fixture):
    cfg, _, arrays, _, _, _ = producer_fixture
    n = cfg["training_trials"]
    assert arrays["trial_blend_applicable"].any()
    assert np.all(arrays["first_blend_mode_trials"][:, :, :5] == n+1)
    assert np.all(arrays["first_balanced_mode_trials"][:, :, 5] == n+1)
    for ti in range(3):
        for ri in range(2):
            for si in range(2):
                at = (ti,ri,5,si)
                discovery = int(arrays["first_discovery_trials"][at].max())
                assert discovery == int(arrays["first_discovery_trials"][ti,ri,1,si].max())
                assert discovery == int(arrays["first_discovery_trials"][ti,ri,4,si].max())
                if discovery <= n:
                    ix = at+(discovery-1,)
                    assert arrays["trial_gate_known_counts_before"][ix] == 3
                    assert arrays["trial_direction_modes"][ix] == 1
                    assert not arrays["trial_blend_applicable"][ix]
                    assert arrays["trial_case_indices"][ix] == -1
                if discovery < n:
                    assert arrays["trial_direction_modes"][at+(discovery,)] == 4
                    assert arrays["trial_gate_switch_events"][at+(discovery,)] == 1
                    assert arrays["first_blend_mode_trials"][at] == discovery+1
                assert arrays["trial_gate_switch_events"][at].sum() == int(discovery < n)


def test_new_prefix_matches_both_isolated_and_original_gated_bytes(producer_fixture):
    cfg, _, arrays, _, _, _ = producer_fixture
    assert audit.gate_checks(cfg, arrays)["exact_gated_balanced_prefix_through_full_discovery"]
    for ti in range(3):
        for ri in range(2):
            for si in range(2):
                stop = min(int(arrays["first_discovery_trials"][ti,ri,5,si].max()),cfg["training_trials"])
                for name in ("actions","rewards","known_before","known_after","directions","update_factors"):
                    x = arrays["trial_"+name]
                    assert x[ti,ri,5,si,:stop].tobytes() == x[ti,ri,1,si,:stop].tobytes()
                    assert x[ti,ri,5,si,:stop].tobytes() == x[ti,ri,4,si,:stop].tobytes()


def test_underlying_certified_B_is_not_misidentified_as_the_actual_mixed_direction(producer_fixture):
    _, _, arrays, _, _, _ = producer_fixture
    active = arrays["trial_blend_applicable"]
    indices = arrays["trial_case_indices"][active]
    assert len(indices) and np.all(arrays["case_keys"][indices,3] == 5)
    b = arrays["case_directions"][indices]
    np.testing.assert_allclose(arrays["trial_blend_balanced_directions"][active],b,atol=1e-12)
    i = arrays["trial_blend_isolated_directions"][active]
    raw = .75*i+.25*b
    expected = raw/np.linalg.norm(raw,axis=-1)[:,None]
    np.testing.assert_allclose(arrays["trial_directions"][active],expected,atol=1e-12)
    assert np.max(np.abs(arrays["trial_directions"][active]-b)) > .01
    np.testing.assert_allclose(arrays["trial_blend_raw_norm"][active],np.linalg.norm(raw,axis=-1),atol=1e-12)


def test_partial_NoGo_keeps_isolated_direction_but_full_NoGo_has_no_mix_or_solver(producer_fixture):
    _, _, arrays, _, _, _ = producer_fixture
    early = (arrays["trial_direction_modes"][:,:,5] == 1)&(arrays["trial_actions"][:,:,5] == 0)
    late = (arrays["trial_direction_modes"][:,:,5] == 4)&(arrays["trial_actions"][:,:,5] == 0)
    assert early.any() and late.any()
    np.testing.assert_allclose(np.linalg.norm(arrays["trial_directions"][:,:,5][early],axis=-1),1.,atol=1e-12)
    assert not arrays["trial_directions"][:,:,5][late].any()
    assert not arrays["trial_blend_applicable"][:,:,5][late].any()
    assert np.all(arrays["trial_case_indices"][:,:,5][late] == -1)
    assert not arrays["trial_update_factors"][:,:,5][early|late].any()


@pytest.mark.parametrize("field", ["trial_blend_isolated_directions","trial_blend_balanced_directions",
    "trial_blend_raw_norm","trial_blend_residuals","trial_directions","trial_actual_influence",
    "trial_desired_influence","case_directions","case_keys"])
def test_blend_diagnostics_checked_from_ingredients_and_actual_case_not_stored_flag(producer_fixture,field):
    cfg, _, original, _, _, _ = producer_fixture
    arrays = copy.deepcopy(original)
    location = tuple(np.argwhere(arrays["trial_blend_applicable"])[0])
    if field.startswith("case_"):
        location = (int(arrays["trial_case_indices"][location]),)
    value = arrays[field][location]
    if np.ndim(value): arrays[field][location][0] += 1 if arrays[field].dtype.kind in "iu" else .02
    else: arrays[field][location] += .02
    with pytest.raises(ValueError): audit.blend_checks(cfg,arrays)


@pytest.mark.parametrize("field", ["trial_blend_applicable","trial_blend_isolated_directions",
    "trial_blend_balanced_directions","trial_blend_raw_norm","trial_blend_residuals"])
def test_old_control_cannot_acquire_hidden_mix_diagnostics(producer_fixture,field):
    cfg, _, original, _, _, _ = producer_fixture
    arrays = copy.deepcopy(original)
    location = (0,0,0,0,0)
    if arrays[field][location].ndim: arrays[field][location][0] = .01
    else: arrays[field][location] = True if arrays[field].dtype.kind == "b" else .01
    if field == "trial_blend_applicable": assert not audit.blend_checks(cfg,arrays)["only_actual_post_Go_has_blend_diagnostics"]
    else:
        if field in ("trial_blend_raw_norm","trial_blend_residuals"):
            with pytest.raises(ValueError): audit.blend_checks(cfg,arrays)
        else: assert not audit.blend_checks(cfg,arrays)["inactive_blend_padding_zero"]


def test_original_gated_success_cannot_replace_new_self_blend_primary(producer_fixture,monkeypatch):
    cfg, arrays, rows = isolated_decision_fixture(producer_fixture,monkeypatch)
    set_final(arrays,4,.99)
    result = audit.expected_summary(cfg,arrays,rows)
    assert not result["H_gain"]["supported"] and not result["H_acquisition"]["supported"]


def test_protection_requires_nonworsening_against_both_frozen_protected_controls(producer_fixture,monkeypatch):
    cfg, arrays, rows = isolated_decision_fixture(producer_fixture,monkeypatch)
    arrays["trial_unknown_other_negative_correct_probability_sum"][0,0,5,0,0] = 5e-9
    result = audit.expected_summary(cfg,arrays,rows)
    assert tuple(result["H_protection"]["comparisons"]) == ("gated_balanced","isolated")
    assert result["H_protection"]["supported"]


@pytest.mark.parametrize("path", [("H_gain","supported"),("H_acquisition","supported"),("H_protection","supported"),
    ("gate_validation","first_blend_mode_follows_full_discovery"),("blend_validation","only_actual_post_Go_has_blend_diagnostics"),
    ("counts","self_blend_direction_requests")])
def test_new_complete_summary_cannot_promote_failed_hypothesis_or_forge_mix_counts(producer_fixture,path):
    from drosolearn.self_blend_learning import summarize
    cfg, graph, arrays, rows, _, _ = producer_fixture
    summary = summarize(cfg,arrays,rows)
    key,field = path
    value = summary[key][field]
    summary[key][field] = not value if isinstance(value,bool) else value+1
    with pytest.raises(ValueError): audit.verify_against_graph(cfg,graph,arrays,rows,summary,allow_fixture=True)
