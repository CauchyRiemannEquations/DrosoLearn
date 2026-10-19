"""Invented-only arithmetic checks for saved-path descriptive accounting."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("balanced_pathways_independent_tests",
    Path(__file__).resolve().parents[1] / "scripts/verify_balanced_pathways.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_finite_probability_changes_telescope_exactly_over_trials():
    path = np.array([[0., 0., 0., 0.], [.2, -.1, -.3, .4], [.1, .2, -.5, .3], [.5, .4, -.2, -.1]])
    record = audit.independent_probability_accounting(path[:-1], path[1:])
    expected = audit.sigmoid(path[-1] * audit.SIGNS) - .5
    assert np.allclose(record["delta"].sum(0), expected, atol=1e-15)
    assert np.allclose(record["positive"] - record["negative"], record["delta"])
    assert np.allclose(record["aligned_margin_delta"].sum(0), path[-1] * audit.SIGNS)


def test_sigmoid_saturation_means_same_margin_step_has_different_probability_effect():
    record = audit.independent_probability_accounting(np.array([[0., 8., 0., -8.]]),
                                                     np.array([[.1, 8.1, -.1, -8.1]]))
    assert record["delta"][0, 0] > record["delta"][0, 1] > 0.
    assert record["delta"][0, 2] > record["delta"][0, 3] > 0.
    assert np.allclose(record["aligned_margin_delta"], .1)


def test_positive_and_negative_mass_cannot_be_replaced_by_net_change():
    record = audit.independent_probability_accounting(np.array([[0., 0., 0., 0.]]),
                                                     np.array([[.2, -.1, -.4, .3]]))
    assert record["positive"].sum() > 0. and record["negative"].sum() > 0.
    assert record["positive"].sum() != pytest.approx(record["delta"].sum())


@pytest.mark.parametrize("gb,gs", [(0., 0.), (.02, -.03), (.02, 0.), (0., -.03), (-.04, -.01)])
def test_symmetric_product_identity_uses_both_actual_path_scalars(gb, gs):
    fb, fs = np.array([[.3, .2, -.1, .4]]), np.array([[.2, -.1, .4, .2]])
    result = audit.symmetric_product_accounting(np.array([gb]), np.array([gs]), fb, fs)
    assert np.allclose(result["direction_component"] + result["scalar_component"], gb * fb - gs * fs)


def test_zero_scalar_difference_leaves_only_direction_component():
    result = audit.symmetric_product_accounting(np.array([.02]), np.array([.02]),
        np.array([[.2, .3, .4, .5]]), np.array([[.5, .4, .3, .2]]))
    assert not result["scalar_component"].any()
    assert np.allclose(result["direction_component"], result["total_difference"])


def test_identical_direction_leaves_only_scalar_component():
    influence = np.array([[.2, .3, .4, .5]])
    result = audit.symmetric_product_accounting(np.array([.02]), np.array([-.03]), influence, influence)
    assert not result["direction_component"].any()
    assert np.allclose(result["scalar_component"], result["total_difference"])


def test_phase_uses_known_before_so_last_discovery_trial_remains_partial():
    before = np.array([[0, 0, 0, 0], [1, 1, -1, 0], [1, 1, -1, -1]])
    all_trials, partial, full = audit.phase_masks(before)
    assert all_trials.tolist() == [True, True, True]
    assert partial.tolist() == [True, True, False]
    assert full.tolist() == [False, False, True]
    assert np.array_equal(partial.astype(int) + full, all_trials)


def test_unknown_other_partition_does_not_include_unobserved_current_source():
    sources, known = np.array([0, 1, 2]), np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, -1, 0]])
    own, familiar, unknown = audit.destination_partition(sources, known)
    assert np.array_equal(own.sum(-1), [1, 1, 1])
    assert np.array_equal(familiar.sum(-1), [0, 1, 2])
    assert np.array_equal(unknown.sum(-1), [3, 2, 1])


@pytest.mark.parametrize("mutation", ["shape", "nan", "missing_destination"])
def test_probability_accounting_invalid_axes_rejected(mutation):
    before, after = np.zeros((2, 4)), np.zeros((2, 4))
    if mutation == "shape": after = np.zeros((1, 4))
    elif mutation == "nan": before[0, 0] = np.nan
    else: before, after = before[:, :3], after[:, :3]
    with pytest.raises(ValueError): audit.independent_probability_accounting(before, after)


def invented_parent(uniforms=None):
    """Eight hand-written saved steps; no RNG, anatomical graph, or learner call."""
    n, s, dimension = 8, 1, 4
    shape = (3, 2, 4, s, n)
    order = np.array([[0, 1, 2, 3, 0, 1, 2, 3]], np.int8)
    parent = {"model_seeds": np.array([90087]), "checkpoints": np.array([0, 4, 8]),
        "targets": np.array([1, 1, 0, 0]), "schedules": order, "uniforms": np.zeros((s, n)) if uniforms is None else np.asarray(uniforms, float),
        "trial_actions": np.ones(shape, np.int8), "trial_rewards": np.empty(shape, np.int8),
        "trial_update_factors": np.empty(shape), "trial_directions": np.empty(shape + (dimension,)),
        "trial_cumulative_head_path_l2": np.empty(shape),
        "trial_actual_influence": np.empty(shape + (4,)), "trial_probabilities": np.empty(shape),
        "trial_correct_probability_delta": np.empty(shape + (4,)), "trial_aligned_margin_delta": np.empty(shape + (4,)),
        "trial_known_before": np.empty(shape + (4,), np.int8), "trial_known_after": np.empty(shape + (4,), np.int8),
        "trial_unknown_other_mask_before": np.empty(shape + (4,), bool), "trial_unknown_other_mask_after": np.empty(shape + (4,), bool),
        "checkpoint_heads": np.empty(shape[:-1] + (3, dimension)), "checkpoint_logits": np.empty(shape[:-1] + (3, 4)),
        "checkpoint_probabilities": np.empty(shape[:-1] + (3, 4))}
    for ti, topology in enumerate(audit.TOPOLOGIES):
        for ri, representation in enumerate(audit.REPRESENTATIONS):
            d = np.eye(4)
            parent["g_" + topology + "_" + representation + "_unit_rows"] = d[None]
            for mi in range(4):
                head, known = np.zeros(4), np.zeros(4, np.int8)
                path = 0.
                parent["checkpoint_heads"][ti, ri, mi, 0, 0] = head
                parent["checkpoint_logits"][ti, ri, mi, 0, 0] = head
                parent["checkpoint_probabilities"][ti, ri, mi, 0, 0] = .5
                for t, source in enumerate(order[0]):
                    at = (ti, ri, mi, 0, t)
                    old = head.copy()
                    known_before = known.copy()
                    p = float(audit.sigmoid(old[source]))
                    action = int(parent["uniforms"][0, t] < p)
                    reward = int(audit.SIGNS[source]) if action else 0
                    direction = d[source].copy()
                    if mi == 2:
                        paired_source = source ^ 1
                        direction = .7 * d[source] + np.sqrt(.51) * d[paired_source]
                    if mi == 3 and not action: direction = np.zeros(4)
                    g = .1 * (action - p) * reward
                    head = old + g * direction
                    path += float(np.linalg.norm(head - old))
                    parent["trial_cumulative_head_path_l2"][at] = path
                    if action: known[source] = reward
                    q = audit.SIGNS * (audit.sigmoid(head) - audit.sigmoid(old))
                    parent["trial_rewards"][at] = reward
                    parent["trial_actions"][at] = action
                    parent["trial_probabilities"][at] = p
                    parent["trial_update_factors"][at] = g
                    parent["trial_directions"][at] = direction
                    parent["trial_actual_influence"][at] = direction
                    parent["trial_correct_probability_delta"][at] = q
                    parent["trial_aligned_margin_delta"][at] = audit.SIGNS * (head - old)
                    parent["trial_known_before"][at], parent["trial_known_after"][at] = known_before, known
                    other = np.arange(4) != source
                    parent["trial_unknown_other_mask_before"][at] = (known_before == 0) & other
                    parent["trial_unknown_other_mask_after"][at] = (known == 0) & other
                    if t + 1 in (4, 8):
                        ki = (4, 8).index(t + 1) + 1
                        parent["checkpoint_heads"][ti, ri, mi, 0, ki] = head
                        parent["checkpoint_logits"][ti, ri, mi, 0, ki] = head
                        parent["checkpoint_probabilities"][ti, ri, mi, 0, ki] = audit.sigmoid(head)
    return parent


def fixture_config():
    from drosolearn.balanced_pathways import SCIENCE
    result = copy.deepcopy(SCIENCE)
    result.update(scope="integration_test", model_seeds=[90087], training_trials=8)
    return result


@pytest.fixture(scope="module")
def independent_fixture():
    cfg, parent = fixture_config(), invented_parent()
    arrays = audit.expected_artifacts(cfg, parent)
    return cfg, parent, arrays, audit.expected_rows(cfg, arrays)


def test_invented_parent_reconstructs_nineteen_arrays_and_two_phases(independent_fixture):
    cfg, parent, arrays, rows = independent_fixture
    assert len(arrays) == 19 and tuple(arrays) == audit.FIELDS
    assert len(rows) == 144
    assert arrays["first_full_trials"].shape == (3, 2, 4, 1)
    assert np.all(arrays["first_full_trials"] == 4)
    assert np.all(arrays["phase_before"][..., :4] == 0)
    assert np.all(arrays["phase_before"][..., 4:] == 1)
    assert arrays["bridge_eligible"].all() and arrays["pair_action_cases"].max() == 3
    assert np.count_nonzero(arrays["objective_eligible"]) == 24
    assert np.all(arrays["objective_values"][..., 1] >= arrays["objective_values"][..., 0])
    assert all(row["exposures"] == 0 for row in rows if row["phase"] == "full" and row["category"] == "unknown_other")


def test_bridge_factual_endpoints_and_shapley_sum_match_recorded_effects(independent_fixture):
    _, _, arrays, _ = independent_fixture
    corners = arrays["bridge_q_cells"]
    assert np.allclose(corners[..., 0, :], arrays["probability_changes"][:, :, 2])
    assert np.allclose(corners[..., 7, :], arrays["probability_changes"][:, :, 3])
    assert np.allclose(arrays["bridge_q_shapley"].sum(-2), corners[..., 7, :] - corners[..., 0, :])
    assert np.allclose(arrays["bridge_harm_shapley"].sum(-2), np.maximum(-corners[..., 7, :], 0.) - np.maximum(-corners[..., 0, :], 0.))


def test_full_current_analysis_and_summary_match_independent_invented_fixture(independent_fixture):
    from drosolearn.balanced_pathways import analyze, summarize_arrays
    cfg, parent, expected, wanted_rows = independent_fixture
    arrays, rows = analyze(cfg, parent)
    audit.check_artifacts(arrays, expected, rows, wanted_rows)
    summary = summarize_arrays(cfg, arrays, rows, parent)
    result = audit.verify_science(cfg, parent, arrays, rows, summary, allow_fixture=True)
    assert result["parent_trial_records_checked"] == 192
    assert result["destination_effects_checked"] == 768
    assert summary["counts"]["new_training_trials"] == summary["counts"]["new_optimization_calls"] == 0


def test_divergent_saved_observations_separate_intersection_from_exposure_membership():
    from drosolearn.balanced_pathways import analyze, summarize_arrays
    cfg = fixture_config()
    parent = invented_parent([[0., .505, 0., 0., 0., 0., 0., 0.]])
    arrays, rows = analyze(cfg, parent)
    summary = summarize_arrays(cfg, arrays, rows, parent)
    audit.verify_science(cfg, parent, arrays, rows, summary, allow_fixture=True)
    assert np.any(arrays["pair_action_cases"] == 1)
    assert np.any(arrays["pair_unknown_membership"] == 3)
    assert arrays["first_full_trials"][0, 0, 2, 0] == 4
    assert arrays["first_full_trials"][0, 0, 3, 0] == 6
    assert not arrays["bridge_eligible"][0, 0, 0, 1]
    assert not arrays["bridge_q_cells"][0, 0, 0, 1].any()


def test_independent_bridge_evaluates_only_both_go_records(monkeypatch):
    parent = invented_parent([[0., .505, 0., 0., 0., 0., 0., 0.]])
    original, calls = audit.independent_bridge, []
    def tracked(zs, zb, gs, gb, fs, fb):
        calls.append(len(gs))
        return original(zs, zb, gs, gb, fs, fb)
    monkeypatch.setattr(audit, "independent_bridge", tracked)
    arrays = audit.expected_artifacts(fixture_config(), parent)
    assert sum(calls) == int(arrays["bridge_eligible"].sum())
    assert sum(calls) < arrays["bridge_eligible"].size


def test_no_discovery_no_bridge_no_objective_means_are_undefined():
    from drosolearn.balanced_pathways import analyze, summarize_arrays
    cfg, parent = fixture_config(), invented_parent(np.full((1, 8), .999))
    arrays, rows = analyze(cfg, parent)
    summary = summarize_arrays(cfg, arrays, rows, parent)
    audit.verify_science(cfg, parent, arrays, rows, summary, allow_fixture=True)
    assert np.all(arrays["first_full_trials"] == 9)
    assert not arrays["bridge_eligible"].any() and not arrays["objective_eligible"].any()
    assert not arrays["probability_changes"].any()
    assert all(contrast["means"]["objective_gain_mean"] is None for contrast in summary["contrasts"])


@pytest.mark.parametrize("field", audit.FIELDS)
def test_all_nineteen_output_artifacts_are_bound_to_independent_arithmetic(independent_fixture, field):
    _, _, expected, wanted_rows = independent_fixture
    arrays = copy.deepcopy(expected)
    flat = arrays[field].reshape(-1)
    flat[-1] = not flat[-1] if flat.dtype.kind == "b" else flat[-1] + (1 if flat.dtype.kind in "iu" else .03)
    with pytest.raises(ValueError): audit.check_artifacts(arrays, expected, wanted_rows, wanted_rows)


@pytest.mark.parametrize("field", ["exposures", "positive_probability_effects", "negative_probability_effects",
                                   "probability_signed_sum", "margin_negative_mass"])
def test_measurement_counts_and_masses_cannot_be_changed(independent_fixture, field):
    _, _, arrays, expected = independent_fixture
    rows = copy.deepcopy(expected)
    rows[0][field] += 1
    with pytest.raises(ValueError): audit.check_artifacts(arrays, arrays, rows, expected)


@pytest.mark.parametrize("key,value", [("phase_definition", "known_after"), ("bridge_eligibility", "all_trials"),
    ("new_confirmatory_hypothesis", True), ("bridge_allocation", "linearized_sigmoid"),
    ("numeric_tolerance", .001), ("methods", ["balanced"]), ("parent_result", "results/other")])
def test_fixed_descriptive_study_choices_cannot_be_relaxed(key, value):
    cfg = fixture_config()
    cfg[key] = value
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


def test_fixture_cannot_run_as_actual_study():
    cfg = fixture_config()
    audit.validate_science(cfg, allow_fixture=True)
    with pytest.raises(ValueError): audit.validate_science(cfg)
    cfg["model_seeds"] = [17501]
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


def test_analysis_and_verifier_do_not_draw_rng_or_run_optimization(monkeypatch, independent_fixture):
    from drosolearn.balanced_pathways import analyze, summarize_arrays
    def forbidden(*args, **kwargs): raise AssertionError("Read-only arithmetic may not draw or optimize")
    for name in ("default_rng", "SeedSequence", "rand", "random"):
        monkeypatch.setattr(np.random, name, forbidden)
    for name in ("svd", "pinv", "lstsq"):
        monkeypatch.setattr(np.linalg, name, forbidden)
    cfg, parent, _, _ = independent_fixture
    arrays, rows = analyze(cfg, parent)
    summary = summarize_arrays(cfg, arrays, rows, parent)
    audit.verify_science(cfg, parent, arrays, rows, summary, allow_fixture=True)


@pytest.mark.parametrize("field", ["trial_update_factors", "trial_directions", "trial_actual_influence", "trial_correct_probability_delta",
    "trial_aligned_margin_delta", "trial_probabilities", "trial_actions", "trial_rewards", "trial_known_before",
    "trial_known_after", "trial_unknown_other_mask_before", "trial_unknown_other_mask_after", "checkpoint_heads",
    "checkpoint_logits", "checkpoint_probabilities", "targets", "model_seeds", "checkpoints", "schedules", "uniforms"])
def test_parent_factual_corruption_is_rejected(independent_fixture, field):
    cfg, original, _, _ = independent_fixture
    parent = copy.deepcopy(original)
    flat = parent[field].reshape(-1)
    flat[-1] = not flat[-1] if flat.dtype.kind == "b" else flat[-1] + (1 if flat.dtype.kind in "iu" else .03)
    if field == "uniforms": flat[-1] = .999
    with pytest.raises(ValueError): audit.independent_parent_state(cfg, parent)
