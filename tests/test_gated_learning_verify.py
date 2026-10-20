"""Invented-only independent checks for prospective memory-gated routing."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("gated_independent_tests",
    Path(__file__).resolve().parents[1] / "scripts/verify_gated_learning.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


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
    from drosolearn.gated_learning import SCIENCE
    cfg = copy.deepcopy(SCIENCE)
    cfg.update(scope="integration_test", model_seeds=[90085, 90086], training_trials=16,
        checkpoints=[0, 4, 8, 16], population_pass_count=2, positive_seed_count=2, protection_nonworse_seed_count=2)
    return cfg


def tiny_graph():
    n = 16
    return {"pre": np.arange(n), "post": np.arange(n), "counts": np.ones(n), "n_kc": n, "n_mbon": n}


@pytest.fixture(scope="module")
def producer_fixture(tmp_path_factory):
    from drosolearn import gated_learning as producer
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
    assert len(arrays) == 195 and len(rows) == 960 and len(rows[0]) == 84
    assert arrays["trial_actions"].shape == (3, 2, 5, 2, 16)
    assert arrays["case_keys"].shape[1] == 9
    assert all(audit.information_checks(cfg, arrays).values())
    assert all(audit.gate_checks(cfg, arrays).values())


def test_complete_current_summary_matches_independent_five_method_summary(producer_fixture):
    from drosolearn.gated_learning import summarize
    cfg, graph, arrays, rows, _, _ = producer_fixture
    summary = summarize(cfg, arrays, rows)
    receipt = audit.verify_against_graph(cfg, graph, arrays, rows, summary, allow_fixture=True)
    assert receipt["independently_replayed_training_actions"] == 960
    assert receipt["conditions"] == 30
    assert len(summary["families"]) == 30
    assert summary["counts"]["model_initializations"] == 60


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
    assert set(keys[:, 3]) == {3, 4}
    assert np.all(np.isin(keys[keys[:, 3] == 4, 5:], [-1, 1]))
    assert len({tuple(row) for row in keys}) == len(keys)
    without_method = [tuple(np.delete(row, 3)) for row in keys]
    assert len(set(without_method)) < len(keys)


@pytest.mark.parametrize("field", ["inputs", "targets", "schedules", "uniforms", "trial_actions", "trial_rewards",
    "trial_probabilities", "trial_known_before", "trial_known_after", "trial_routing_signs", "trial_harm_mask",
    "trial_desired_influence", "trial_actual_influence", "trial_directions", "trial_direction_residuals",
    "trial_direction_residuals_applicable", "trial_case_indices", "trial_direction_modes", "trial_gate_known_counts_before",
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
    set_final(arrays, 4, .7)
    summary = audit.expected_summary(cfg, arrays, rows)
    assert summary["H_gain"]["supported"] and not summary["H_acquisition"]["supported"]
    for control in (1, 2, 3): set_final(arrays, control, .90)
    set_final(arrays, 4, .94)
    summary = audit.expected_summary(cfg, arrays, rows)
    assert summary["H_acquisition"]["supported"] and not summary["H_gain"]["supported"]


@pytest.mark.parametrize("control", [1, 2, 3])
def test_gain_requires_all_three_frozen_comparators(producer_fixture, monkeypatch, control):
    cfg, arrays, rows = isolated_decision_fixture(producer_fixture, monkeypatch)
    set_final(arrays, 4, .7)
    set_final(arrays, control, .69)
    assert not audit.expected_summary(cfg, arrays, rows)["H_gain"]["supported"]


def test_large_mean_effect_does_not_replace_positive_seed_count(producer_fixture, monkeypatch):
    cfg, arrays, rows = isolated_decision_fixture(producer_fixture, monkeypatch)
    set_final(arrays, 4, .8)
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
    arrays["trial_unknown_other_negative_correct_probability_sum"][0, 0, 4, 0, 0] = 1.5e-8
    arrays["trial_unknown_other_negative_correct_probability_sum"][0, 0, 2:4, :, 0] = 1e-6
    summary = audit.expected_summary(cfg, arrays, rows)
    assert arrays["trial_unknown_other_negative_correct_probability_sum"][0, 0, 4].sum(-1).mean() < 1e-8
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
    "trial_gate_switch_events", "first_balanced_mode_trials"])
def test_gate_integrity_failure_is_fatal_before_scientific_claims(producer_fixture, field):
    cfg, _, original, rows, _, _ = producer_fixture
    arrays = copy.deepcopy(original)
    flat = arrays[field].reshape(-1)
    flat[-1] = not flat[-1] if flat.dtype.kind == "b" else flat[-1] + 1
    with pytest.raises(ValueError): audit.expected_summary(cfg, arrays, rows)


@pytest.mark.parametrize("key,value", [("gate_switch_rule", "known_after_four"), ("protection_absolute_tolerance", .01),
    ("head_lr", .2), ("rng_namespace", 202610201), ("effect_threshold", .001), ("methods", ["gated_balanced"])])
def test_fixed_science_and_gate_parameters_cannot_be_relaxed(key, value):
    cfg = fixture_config()
    cfg[key] = value
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


def test_fixture_scope_cannot_enter_actual_research():
    cfg = fixture_config()
    audit.validate_science(cfg, allow_fixture=True)
    with pytest.raises(ValueError): audit.validate_science(cfg)
    cfg["model_seeds"] = [17601]
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)
