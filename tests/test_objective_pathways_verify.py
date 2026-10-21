"""Invented-only adversarial checks for independent saved objective paths."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("independent_objective_pathways", ROOT / "scripts/verify_objective_pathways.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_all_eight_cells_use_recorded_state_scalar_direction_bits():
    zc, zn = np.array([.1, -.3, .4, -.2]), np.array([.4, -.1, .2, -.5])
    gc, gn = np.asarray(.04), np.asarray(-.03)
    fc, fn = np.array([1., .2, -.1, .3]), np.array([.8, .3, -.2, .1])
    values = audit.independent_bridge(zc, zn, gc, gn, fc, fn)
    for code in range(8):
        z, g, f = (zn if code & 4 else zc), (gn if code & 2 else gc), (fn if code & 1 else fc)
        assert np.allclose(values[code], audit.SIGNS * (audit.sigmoid(z + g * f) - audit.sigmoid(z)))


@pytest.mark.parametrize("bit,index", [(4, 0), (2, 1), (1, 2)])
def test_single_factor_effect_stays_in_its_own_symmetric_allocation(bit, index):
    values = np.array([[1., -2., 3., -4.] if code & bit else [0.] * 4 for code in range(8)])
    allocation = audit.independent_shapley(values)
    assert np.allclose(allocation[index], values[bit])
    assert not np.delete(allocation, index, axis=0).any()


def test_three_way_interaction_is_split_equally_and_preserves_integer_effects():
    values = np.zeros((8, 4), int)
    values[7] = [3, -6, 9, -12]
    allocation = audit.independent_shapley(values)
    assert np.allclose(allocation, np.broadcast_to(values[7] / 3., (3, 4)))
    assert np.allclose(allocation.sum(axis=0), values[7] - values[0])


def test_positive_and_negative_masses_are_split_before_allocating():
    values = np.array([[1. - .2 * bool(code & 1)] * 4 for code in range(8)])
    positive = audit.independent_shapley(np.maximum(values, 0.))
    negative = audit.independent_shapley(np.maximum(-values, 0.))
    signed = audit.independent_shapley(values)
    assert np.all(positive[2] < 0.) and not negative.any()
    assert np.allclose(signed, positive - negative)


def test_saturated_output_state_can_reduce_gain_without_smaller_margin():
    values = audit.independent_bridge(np.zeros(4), np.full(4, 8.), np.asarray(.1), np.asarray(.1), np.ones(4), np.ones(4))
    split = audit.independent_shapley(values)
    assert np.all(audit.sigmoid(np.full(4, 8.)) > .5)
    assert split[0, 0] < 0. and np.allclose(split[1:], 0.)


@pytest.mark.parametrize("bad", [np.full(4, np.nan), np.ones(4, complex), np.zeros(3)])
def test_nonfinite_complex_or_wrong_shape_saved_states_are_rejected(bad):
    with pytest.raises(ValueError):
        audit.independent_bridge(bad, np.zeros(4), np.asarray(.1), np.asarray(.1), np.ones(4), np.ones(4))


@pytest.mark.parametrize("bad", [np.zeros(8), np.zeros((7, 4)), np.full((8, 4), np.inf), np.ones((8, 4), complex)])
def test_shapley_rejects_invalid_or_nonfinite_cell_tables(bad):
    with pytest.raises(ValueError):
        audit.independent_shapley(bad)


@pytest.fixture(scope="module")
def saved_fixture():
    fixture_spec = importlib.util.spec_from_file_location("invented_objective_paths", ROOT / "tests/test_objective_pathways.py")
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture_module)
    from drosolearn import objective_pathways as producer
    cfg, parent = fixture_module.fixture()
    arrays, rows = producer.analyze(cfg, parent)
    summary = producer.summarize_arrays(cfg, arrays, rows, parent)
    return cfg, parent, arrays, rows, summary


def test_saved_fixture_matches_all_independent_artifacts_rows_and_complete_summary(saved_fixture):
    cfg, parent, arrays, rows, summary = saved_fixture
    counts = audit.verify_science(cfg, parent, arrays, rows, summary, allow_fixture=True)
    assert counts["arrays"] == 16
    assert counts["parent_trial_records_checked"] == 48 * len(cfg["model_seeds"]) * cfg["training_trials"]
    assert counts["measurement_records_checked"] == 216 * len(cfg["model_seeds"])
    assert counts["source_action_reward_strata_checked"] == 48 * len(cfg["model_seeds"]) * 72
    assert len(summary["families"]) == 18 and len(summary["actual_paths"]) == 48


def test_same_gate_prefix_only_both_go_crossings_and_all_one_sided_actuals_survive(saved_fixture):
    cfg, parent, arrays, rows, summary = saved_fixture
    assert np.array_equal(arrays["bridge_eligible"], arrays["reference_post"][:, :, None] & (arrays["pair_action_cases"] == 3))
    for name in ("bridge_q_cells", "bridge_q_shapley", "bridge_positive_shapley", "bridge_negative_shapley", "factual_pair_diagnostics"):
        assert not arrays[name][~arrays["bridge_eligible"]].any()
    for family in summary["families"]:
        for record in family["seeds"]:
            assert sum(record["case_counts"]) == record["reference_post_trials"]
            assert record["case_counts"][3] == record["both_Go_trials"]
            for key, factual in (("component_endpoint_contributions", "factual_post_endpoint_difference"),
                    ("component_positive_endpoint_allocations", "factual_post_positive_endpoint_difference"),
                    ("component_negative_endpoint_allocations", "factual_post_negative_endpoint_difference")):
                assert np.allclose(np.asarray(record[key]).sum(axis=-1), record[factual])
            assert np.isclose(sum(record["factual_pre_endpoint_difference"]) + sum(record["factual_post_endpoint_difference"]), record["final_endpoint_difference"])


def test_actual_reward_classes_and_impossible_strata_have_honest_counts_and_null_means(saved_fixture):
    cfg, parent, arrays, rows, summary = saved_fixture
    for family in summary["actual_paths"]:
        for record in family["seeds"]:
            strata = record["source_action_reward_strata"]
            assert len(strata) == 72
            for phase in ("all", "pre", "post"):
                selected = [row for row in strata if row["phase"] == phase]
                assert sum(row["trials"] for row in selected) == record["phases"][phase]["trials"]
                for row in selected:
                    if row["trials"] == 0:
                        assert row["source_probability_mean"] is None
                    if row["action"] == 0:
                        assert row["absolute_scalar_sum"] == row["head_path_l2"] == 0.
                    if row["action"] == 0 and row["reward"] != 0:
                        assert row["trials"] == 0
                    if row["action"] == 1 and row["reward"] == 0:
                        assert row["trials"] == 0


def test_paired_go_size_identity_uses_identical_obtained_reward_and_own_probability(saved_fixture):
    cfg, parent, arrays, rows, summary = saved_fixture
    for ti in range(3):
        for ri in range(2):
            for ci, (candidate, control) in enumerate(audit.COMPARISONS):
                nm, cm = audit.METHODS.index(candidate), audit.METHODS.index(control)
                for si in range(len(cfg["model_seeds"])):
                    mask = arrays["bridge_eligible"][ti, ri, ci, si]
                    c, new = (ti, ri, cm, si), (ti, ri, nm, si)
                    assert np.array_equal(parent["trial_rewards"][c][mask], parent["trial_rewards"][new][mask])
                    wanted = -.1 * parent["trial_rewards"][c][mask] * (parent["trial_probabilities"][new][mask] - parent["trial_probabilities"][c][mask])
                    assert np.allclose(parent["trial_update_factors"][new][mask] - parent["trial_update_factors"][c][mask], wanted)


@pytest.mark.parametrize("field", audit.FIELDS)
def test_each_derived_numeric_array_is_independently_bound(saved_fixture, field):
    _, _, original, rows, _ = saved_fixture
    arrays = copy.deepcopy(original)
    flat = arrays[field].reshape(-1)
    flat[-1] = not flat[-1] if flat.dtype.kind == "b" else flat[-1] + (1 if flat.dtype.kind in "iu" else .03)
    with pytest.raises(ValueError):
        audit.check_artifacts(arrays, original, rows, rows)


@pytest.mark.parametrize("field", ["trial_update_factors", "trial_directions", "trial_correct_probability_delta", "trial_aligned_margin_delta",
    "checkpoint_heads", "checkpoint_probabilities", "trial_probabilities", "trial_head_steps", "trial_head_step_squared_l2",
    "trial_cumulative_head_path_l2", "trial_cumulative_head_squared_l2", "trial_unknown_other_negative_correct_probability_sum"])
def test_parent_numeric_fact_corruption_cannot_enter_an_accounting(saved_fixture, field):
    cfg, original, _, _, _ = saved_fixture
    parent = copy.deepcopy(original)
    parent[field].reshape(-1)[0] += .03
    with pytest.raises(ValueError):
        audit.expected_artifacts(cfg, parent)


@pytest.mark.parametrize("field", ["trial_known_before", "trial_known_after", "trial_unknown_other_mask_before", "trial_unknown_other_mask_after",
    "first_balanced_mode_trials", "first_blend_mode_trials", "first_uniform_mode_trials", "first_probability_mode_trials"])
def test_recorded_knowledge_or_mode_boundary_corruption_is_rejected(saved_fixture, field):
    cfg, original, _, _, _ = saved_fixture
    parent = copy.deepcopy(original)
    if field.startswith("first_"):
        method = {"first_balanced_mode_trials": 4, "first_blend_mode_trials": 5, "first_uniform_mode_trials": 6, "first_probability_mode_trials": 7}[field]
        parent[field][0, 0, method, 0] += 1
    else:
        parent[field].flat[0] = not parent[field].flat[0] if parent[field].dtype.kind == "b" else parent[field].flat[0] + 1
    with pytest.raises(ValueError):
        audit.expected_artifacts(cfg, parent)


@pytest.mark.parametrize("mutation", ["uniform", "action", "reward", "target", "source", "seed"])
def test_actual_choice_reward_and_fixed_input_identity_cannot_be_substituted(saved_fixture, mutation):
    cfg, original, _, _, _ = saved_fixture
    parent = copy.deepcopy(original)
    if mutation == "uniform":
        parent["uniforms"][0, 0] = .999
    elif mutation == "action":
        parent["trial_actions"].flat[0] = 1 - parent["trial_actions"].flat[0]
    elif mutation == "reward":
        parent["trial_rewards"].flat[0] *= -1
    elif mutation == "target":
        parent["targets"][0] = 0
    elif mutation == "source":
        parent["schedules"][0, 0] = 4
    else:
        parent["model_seeds"][0] += 1
    with pytest.raises(ValueError):
        audit.expected_artifacts(cfg, parent)


@pytest.mark.parametrize("field", ["reference_post_trials", "both_Go_trials", "component_trials", "probability_signed_sum", "positive_mass_allocation",
    "negative_mass_allocation", "endpoint_contribution", "positive_endpoint_allocation", "negative_endpoint_allocation"])
def test_csv_component_counts_units_and_allocations_are_bound(saved_fixture, field):
    _, _, arrays, original, _ = saved_fixture
    rows = copy.deepcopy(original)
    rows[0][field] += 1
    with pytest.raises(ValueError):
        audit.check_artifacts(arrays, arrays, rows, original)


@pytest.mark.parametrize("key,value", [("bridge_eligibility", "any_actual_Go"), ("bridge_bits", ["direction", "scalar", "state"]),
    ("bridge_allocation", "linearized"), ("comparisons", [["gated_uniform", "isolated"]]), ("numeric_tolerance", .001),
    ("new_confirmatory_hypothesis", True), ("phase_definition", "known_after"), ("choice_reward_phases", ["post"]),
    ("choice_reward_actions", [1]), ("choice_reward_values", [1]), ("baseline_commit", "0" * 40)])
def test_registered_choices_cannot_be_relaxed_for_favorable_accounting(saved_fixture, key, value):
    cfg = copy.deepcopy(saved_fixture[0])
    cfg[key] = value
    with pytest.raises(ValueError):
        audit.validate_science(cfg, allow_fixture=True)


def test_invented_fixture_cannot_enter_real_registered_analysis(saved_fixture):
    cfg = copy.deepcopy(saved_fixture[0])
    audit.validate_science(cfg, allow_fixture=True)
    with pytest.raises(ValueError):
        audit.validate_science(cfg)
    cfg["model_seeds"] = [17801]
    with pytest.raises(ValueError):
        audit.validate_science(cfg, allow_fixture=True)


@pytest.mark.parametrize("target", ["component", "diagnostic", "unpaired", "endpoint", "first_divergence", "stratum_count", "stratum_null",
    "cost", "unknown", "parent_verdict", "new_training"])
def test_summary_cannot_change_allocation_paths_denominators_costs_or_old_verdicts(saved_fixture, target):
    cfg, parent, arrays, rows, original = saved_fixture
    summary = copy.deepcopy(original)
    record = summary["families"][0]["seeds"][0]
    path = summary["actual_paths"][0]["seeds"][0]
    if target == "component":
        record["component_endpoint_contributions"][0][0] += .1
    elif target == "diagnostic":
        record["diagnostic_means"][0][0] += .1
    elif target == "unpaired":
        record["case_counts"][1] += 1
    elif target == "endpoint":
        record["final_endpoint_difference"] += .1
    elif target == "first_divergence":
        record["first_actual_action_divergence"] = 999
    elif target == "stratum_count":
        path["source_action_reward_strata"][0]["trials"] += 1
    elif target == "stratum_null":
        next(row for row in path["source_action_reward_strata"] if row["trials"] == 0)["source_probability_mean"] = 0.
    elif target == "cost":
        path["phases"]["post"]["head_path_l2"] += .1
    elif target == "unknown":
        path["phases"]["all"]["unknown_other_damage"] += .1
    elif target == "parent_verdict":
        summary["interpretation"]["parent_U_gain"] = True
    else:
        summary["counts"]["new_training_trials"] = 1
    with pytest.raises(ValueError):
        audit.verify_science(cfg, parent, arrays, rows, summary, allow_fixture=True)


def test_saved_arithmetic_does_not_modify_inputs_or_call_randomness_or_optimization(saved_fixture, monkeypatch):
    cfg, parent, arrays, rows, summary = saved_fixture
    original = {name: value.copy() for name, value in parent.items()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Saved accounting cannot generate a new policy, random draw or solver")
    for name in ("default_rng", "SeedSequence", "random", "rand"):
        monkeypatch.setattr(np.random, name, forbidden)
    for name in ("svd", "pinv", "lstsq", "inv"):
        monkeypatch.setattr(np.linalg, name, forbidden)
    audit.verify_science(cfg, parent, arrays, rows, summary, allow_fixture=True)
    assert all(np.array_equal(value, parent[name]) for name, value in original.items())


def test_never_opened_gate_keeps_null_means_zero_cells_and_historical_nogo_diagnostics(saved_fixture):
    from drosolearn import objective_pathways as producer
    cfg, original, _, _, _ = saved_fixture
    parent = copy.deepcopy(original)
    parent["uniforms"][:] = .75
    for name in ("trial_actions", "trial_rewards", "trial_update_factors", "trial_directions", "trial_known_before", "trial_known_after",
            "trial_correct_probability_delta", "trial_aligned_margin_delta", "trial_head_steps", "trial_head_step_squared_l2",
            "trial_cumulative_head_path_l2", "trial_cumulative_head_squared_l2", "trial_unknown_other_negative_correct_probability_sum", "checkpoint_heads"):
        parent[name][:] = 0
    parent["trial_probabilities"][:] = .5
    parent["checkpoint_probabilities"][:] = .5
    other = np.arange(4) != parent["schedules"][..., None]
    for name in ("trial_unknown_other_mask_before", "trial_unknown_other_mask_after"):
        parent[name][:] = other[None, None, None]
    for name in ("first_balanced_mode_trials", "first_blend_mode_trials", "first_uniform_mode_trials", "first_probability_mode_trials"):
        parent[name][:] = cfg["training_trials"] + 1
    # Old shared/isolation/selective methods recorded an available diagnostic
    # direction on NoGo; with g=0 it is not an actual or crossed update.
    parent["trial_directions"][:, :, :3, ..., 0] = 1.
    arrays, rows = producer.analyze(cfg, parent)
    summary = producer.summarize_arrays(cfg, arrays, rows, parent)
    audit.verify_science(cfg, parent, arrays, rows, summary, allow_fixture=True)
    assert not arrays["bridge_eligible"].any() and not arrays["bridge_q_cells"].any()
    assert np.all(arrays["first_post_trials"] == cfg["training_trials"] + 1)
    assert all(family["means"]["diagnostic_means"] == [[None] * 12] * 2 for family in summary["families"])
    assert all(family["means"]["first_actual_action_divergence"] is None for family in summary["families"])
