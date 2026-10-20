"""Exact post-gate factors on handwritten saved paths, with no fresh learner."""
from copy import deepcopy

import numpy as np
import pytest


def study():
    import drosolearn.gated_factors as module
    return module


def sigmoid(z):
    return 1. / (1. + np.exp(-np.clip(z, -40., 40.)))


def invented_saved_paths(gate_never=False):
    """Twelve explicit saved steps; no actor, RNG, graph or optimizer call."""
    n = 12
    shape = (3, 2, 5, 1, n)
    signs = np.array([1, 1, -1, -1])
    cfg = deepcopy(study().SCIENCE)
    cfg.update(scope="integration_test", model_seeds=[90091], training_trials=n)
    parent = {"model_seeds": np.array([90091]), "targets": np.array([1, 1, 0, 0]),
        "schedules": np.array([[0, 1, 2, 3] * 3], np.int8), "checkpoints": np.array([0, 4, 8, 12])}
    for name in ["trial_update_factors", "trial_head_steps", "trial_head_step_squared_l2",
            "trial_cumulative_head_path_l2", "trial_cumulative_head_squared_l2", "trial_probabilities"]:
        parent[name] = np.zeros(shape)
    for name in ["trial_actions", "trial_rewards"]:
        parent[name] = np.zeros(shape, np.int8)
    for name in ["trial_directions", "trial_actual_influence", "trial_correct_probability_delta", "trial_aligned_margin_delta"]:
        parent[name] = np.zeros(shape + (4,))
    for name in ["trial_known_before", "trial_known_after"]:
        parent[name] = np.zeros(shape + (4,), np.int8)
    parent["checkpoint_heads"] = np.zeros(shape[:-1] + (4, 4))
    parent["checkpoint_logits"] = np.zeros(shape[:-1] + (4, 4))
    parent["checkpoint_probabilities"] = np.full(shape[:-1] + (4, 4), .5)
    parent["first_balanced_mode_trials"] = np.full(shape[:-1], n + 1, np.int32)
    for ti, topology in enumerate(["real", "rewired", "ann"]):
        for ri, representation in enumerate(["raw", "identity"]):
            parent["g_" + topology + "_" + representation + "_unit_rows"] = np.eye(4)[None]
            for mi in range(5):
                head = np.zeros(4)
                known = np.zeros(4, np.int8)
                path = energy = 0.
                for t, source in enumerate(parent["schedules"][0]):
                    at = (ti, ri, mi, 0, t)
                    before = head.copy()
                    memory = known.copy()
                    full = np.count_nonzero(memory) == 4
                    action = 1
                    if mi == 0:
                        action = 0
                    elif mi == 1 and t in (3, 4, 7):
                        action = 0
                    elif mi == 2 and t in (5, 7):
                        action = 0
                    elif mi == 3 and t in (5, 9):
                        action = 0
                    elif mi == 4 and t in (5, 7):
                        action = 0
                    if mi == 4 and gate_never:
                        action = 0
                    p = float(sigmoid(head[source]))
                    reward = int(signs[source]) if action else 0
                    factor = .1 * (action - p) * reward
                    direction = np.eye(4)[source].copy()
                    if mi in (2, 3) or (mi == 4 and full):
                        direction *= .5 + .05 * mi
                        for j in range(4):
                            if j != source:
                                direction[j] = .07 * mi * signs[source] * signs[j]
                    if action == 0 and (mi == 3 or (mi == 4 and full)):
                        direction[:] = 0.
                    influence = direction.copy()
                    old = before.copy()
                    delta = signs * (sigmoid(old + factor * influence) - sigmoid(old))
                    margin = signs * factor * influence
                    head += factor * direction
                    if action:
                        known[source] = reward
                    step = abs(factor) * float(np.linalg.norm(direction))
                    path += step
                    energy += step ** 2
                    for name, value in [
                        ("trial_actions", action), ("trial_rewards", reward), ("trial_probabilities", p),
                        ("trial_update_factors", factor), ("trial_directions", direction),
                        ("trial_actual_influence", influence), ("trial_head_steps", step),
                        ("trial_head_step_squared_l2", step ** 2), ("trial_cumulative_head_path_l2", path),
                        ("trial_cumulative_head_squared_l2", energy), ("trial_known_before", memory),
                        ("trial_known_after", known), ("trial_correct_probability_delta", delta),
                        ("trial_aligned_margin_delta", margin)]:
                        parent[name][at] = value
                    if mi == 4 and full and parent["first_balanced_mode_trials"][ti, ri, mi, 0] == n + 1:
                        parent["first_balanced_mode_trials"][ti, ri, mi, 0] = t + 1
                    if t + 1 in (4, 8, 12):
                        ki = (4, 8, 12).index(t + 1) + 1
                        ix = (ti, ri, mi, 0, ki)
                        parent["checkpoint_heads"][ix] = head
                        parent["checkpoint_logits"][ix] = head
                        parent["checkpoint_probabilities"][ix] = sigmoid(head)
    accounting = {"probability_changes": parent["trial_correct_probability_delta"].copy(),
        "margin_changes": parent["trial_aligned_margin_delta"].copy(),
        "gated_phase_before": (np.count_nonzero(parent["trial_known_before"][:, :, 4], axis=-1) == 4).astype(np.int8)}
    return cfg, parent, accounting


@pytest.fixture
def bridge_arguments():
    return (np.array([.1, -.3, .4, -.2]), np.array([.4, -.1, .2, -.5]),
        np.asarray(.04), np.asarray(-.03), np.array([1., .2, -.1, .3]),
        np.array([.8, .3, -.2, .1]), np.array([1, 1, -1, -1]))


def test_all_eight_cells_use_state_four_scalar_two_direction_one(bridge_arguments):
    zc, zg, gc, gg, fc, fg, signs = bridge_arguments
    cells = study().bridge_cells(*bridge_arguments)
    assert cells.shape == (8, 4)
    for code in range(8):
        z = zg if code & 4 else zc
        g = gg if code & 2 else gc
        f = fg if code & 1 else fc
        assert np.allclose(cells[code], signs * (sigmoid(z + g * f) - sigmoid(z)), atol=1e-14)


def test_batched_cells_do_not_advance_or_mutate_either_saved_path(bridge_arguments):
    old = [x.copy() for x in bridge_arguments]
    zc, zg, gc, gg, fc, fg, signs = bridge_arguments
    cells = study().bridge_cells(np.stack([zc, zg]), np.stack([zg, zc]), np.array([gc, gg]),
        np.array([gg, gc]), np.stack([fc, fg]), np.stack([fg, fc]), signs)
    assert cells.shape == (2, 8, 4)
    assert np.allclose(cells[1], cells[0, ::-1], atol=1e-14)
    for value, original in zip(bridge_arguments, old):
        assert np.array_equal(value, original)


@pytest.mark.parametrize("bit,factor", [(4, 0), (2, 1), (1, 2)])
def test_single_factor_is_not_allocated_to_other_factors(bit, factor):
    cells = np.array([[1., -2., 3., -4.] if code & bit else [0., 0., 0., 0.] for code in range(8)])
    result = study().shapley3(cells)
    assert np.allclose(result[factor], cells[bit])
    assert np.allclose(np.delete(result, factor, axis=0), 0.)


def test_two_and_three_way_interactions_are_symmetric_and_integer_safe():
    pair = np.zeros((8, 4), int)
    pair[[6, 7]] = [2, -4, 6, -8]
    split = study().shapley3(pair)
    assert np.allclose(split[0], split[1]) and not split[2].any()
    assert np.allclose(split[0], pair[7] / 2.)
    triple = np.zeros((8, 4), int)
    triple[7] = [3, -6, 9, -12]
    assert np.allclose(study().shapley3(triple), np.broadcast_to(triple[7] / 3., (3, 4)))


def test_signed_positive_negative_allocations_each_reconstruct_their_corner_gap(bridge_arguments):
    cells = study().bridge_cells(*bridge_arguments)
    signed = study().shapley3(cells)
    positive = study().shapley3(np.maximum(cells, 0.))
    negative = study().shapley3(np.maximum(-cells, 0.))
    assert np.allclose(signed, positive - negative, atol=1e-14)
    assert np.allclose(signed.sum(0), cells[7] - cells[0], atol=1e-14)
    assert np.allclose(positive.sum(0), np.maximum(cells[7], 0.) - np.maximum(cells[0], 0.), atol=1e-14)
    assert np.allclose(negative.sum(0), np.maximum(-cells[7], 0.) - np.maximum(-cells[0], 0.), atol=1e-14)
    assert np.any(positive < 0.) or np.any(negative < 0.)
    assert not np.allclose(negative, np.maximum(-signed, 0.))


def test_saturation_can_reduce_gain_without_a_worse_correct_probability_state():
    cells = study().bridge_cells(np.zeros(4), np.full(4, 8.), np.asarray(.1), np.asarray(.1),
        np.ones(4), np.ones(4), np.ones(4, int))
    split = study().shapley3(cells)
    assert np.all(sigmoid(np.full(4, 8.)) > .5)
    assert np.all(split[0] < 0.) and not split[1:].any()
    extreme = study().bridge_cells(np.full(4, 50.), np.full(4, 50.), np.asarray(.1), np.asarray(.2),
        np.ones(4), np.ones(4), np.ones(4, int))
    assert not extreme.any()


def test_synthetic_common_post_keeps_discovery_pre_and_all_four_actual_action_cases():
    cfg, parent, accounting = invented_saved_paths()
    arrays, rows = study().analyze(cfg, parent, accounting)
    assert len(arrays) == 14 and len(rows) == 288
    assert arrays["first_post_trials"][0, 0, 0] == 5
    assert arrays["reference_post"][0, 0, 0].tolist() == [False] * 4 + [True] * 8
    assert arrays["pair_action_cases"][0, 0, 0, 0, 4:8].tolist() == [2, 1, 3, 0]
    assert np.array_equal(arrays["bridge_eligible"], arrays["reference_post"][:, :, None] & (arrays["pair_action_cases"] == 3))
    assert np.all(arrays["bridge_q_cells"][~arrays["bridge_eligible"]] == 0.)
    assert np.all(arrays["factual_pair_diagnostics"][~arrays["bridge_eligible"]] == 0.)


def test_post_corners_match_facts_and_own_other_six_component_totals_reconstruct():
    cfg, parent, accounting = invented_saved_paths()
    arrays, rows = study().analyze(cfg, parent, accounting)
    for ci, control in enumerate([1, 3, 2, 0]):
        mask = arrays["bridge_eligible"][:, :, ci]
        assert np.allclose(arrays["bridge_q_cells"][:, :, ci][mask, 0], arrays["probability_changes"][:, :, control][mask], atol=1e-14)
        assert np.allclose(arrays["bridge_q_cells"][:, :, ci][mask, 7], arrays["probability_changes"][:, :, 4][mask], atol=1e-14)
    summary = study().summarize_arrays(cfg, arrays, rows, parent, accounting)
    assert len(summary["families"]) == 24
    for family in summary["families"]:
        seed = family["seeds"][0]
        signed = np.asarray(seed["component_endpoint_contributions"])
        positive = np.asarray(seed["component_positive_endpoint_allocations"])
        negative = np.asarray(seed["component_negative_endpoint_allocations"])
        assert signed.shape == (2, 6) and np.allclose(signed, positive - negative, atol=1e-14)
        assert np.allclose(signed.sum(1), seed["factual_post_endpoint_difference"], atol=1e-14)
        assert signed.sum() == pytest.approx(seed["factual_post_total_difference"])
        assert sum(seed["case_counts"]) == seed["reference_post_trials"] == 8
    for row in rows:
        assert row["endpoint_contribution"] == pytest.approx(row["probability_signed_sum"] / 4.)
        assert row["positive_endpoint_allocation"] == pytest.approx(row["positive_mass_allocation"] / 4.)
        assert row["negative_endpoint_allocation"] == pytest.approx(row["negative_mass_allocation"] / 4.)
        assert row["probability_signed_sum"] == pytest.approx(row["positive_mass_allocation"] - row["negative_mass_allocation"])


def test_partial_control_knowledge_and_nonpaired_factual_remainders_are_not_deleted():
    cfg, parent, accounting = invented_saved_paths()
    arrays, rows = study().analyze(cfg, parent, accounting)
    summary = study().summarize_arrays(cfg, arrays, rows, parent, accounting)
    seed = summary["families"][0]["seeds"][0]
    assert seed["control_partial_before_on_eligible_trials"] > 0
    contributions = np.asarray(seed["component_endpoint_contributions"])
    assert np.any(contributions[:, 3] != 0.) and np.any(contributions[:, 4] != 0.)
    assert not contributions[:, 5].any()
    assert seed["both_Go_trials"] + seed["case_counts"][0] + seed["case_counts"][1] + seed["case_counts"][2] == 8
    shared = next(f for f in summary["families"] if f["control"] == "shared")
    assert shared["seeds"][0]["both_Go_trials"] == 0
    assert shared["seeds"][0]["diagnostic_means"] == [[None] * 8, [None] * 8]


def test_diagnostics_distinguish_unit_direction_actual_scalar_and_actual_length():
    cfg, parent, accounting = invented_saved_paths()
    arrays, _ = study().analyze(cfg, parent, accounting)
    at = (0, 0, 0, 0, 6)
    assert arrays["bridge_eligible"][at]
    source = 2
    for side, mi in enumerate([1, 4]):
        base = (0, 0, mi, 0)
        factors = parent["trial_update_factors"][base]
        directions = parent["trial_directions"][base]
        head = np.sum(factors[:6, None] * directions[:6], axis=0)
        direction = directions[6]
        p = float(sigmoid(head[source]))
        expected = [(-1.) * head[source], 1. - p, p * (1. - p), abs(factors[6]),
            abs(factors[6]) * np.linalg.norm(direction), direction[source],
            np.mean((-1.) * np.array([1, 1, -1, -1])[np.arange(4) != source] * direction[np.arange(4) != source]),
            np.linalg.norm(direction)]
        assert np.allclose(arrays["factual_pair_diagnostics"][at + (side,)], expected, atol=1e-14)
        assert expected[4] == pytest.approx(expected[3] * expected[7])
    assert arrays["factual_pair_diagnostics"][at + (1, 5)] < arrays["factual_pair_diagnostics"][at + (0, 5)]
    assert arrays["factual_pair_diagnostics"][at + (1, 6)] > 0.


def test_no_open_gate_keeps_zero_padded_post_and_null_diagnostics():
    cfg, parent, accounting = invented_saved_paths(gate_never=True)
    arrays, rows = study().analyze(cfg, parent, accounting)
    summary = study().summarize_arrays(cfg, arrays, rows, parent, accounting)
    assert np.all(arrays["first_post_trials"] == 13) and not arrays["reference_post"].any()
    assert not arrays["bridge_eligible"].any() and not arrays["bridge_q_cells"].any()
    assert len(rows) == 288 and all(row["component_trials"] == 0 for row in rows)
    for family in summary["families"]:
        seed = family["seeds"][0]
        assert seed["factual_post_total_difference"] == 0.
        assert seed["diagnostic_means"] == [[None] * 8, [None] * 8]


def test_saved_parent_and_accounting_are_immutable_and_analysis_makes_no_new_rng(monkeypatch):
    cfg, parent, accounting = invented_saved_paths()
    originals = [{key: value.copy() for key, value in item.items()} for item in [parent, accounting]]
    def forbidden(*args, **kwargs):
        raise AssertionError("Saved-factor arithmetic must not draw randomness or optimize a direction")
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    arrays, rows = study().analyze(cfg, parent, accounting)
    summary = study().summarize_arrays(cfg, arrays, rows, parent, accounting)
    for original, item in zip(originals, [parent, accounting]):
        for key in original:
            assert np.array_equal(original[key], item[key]), key
    for key in ["new_training_trials", "new_head_update_calls", "new_internal_update_calls", "new_value_update_calls",
            "new_random_draws", "fresh_model_constructions", "new_optimization_calls"]:
        assert summary["counts"][key] == 0
    assert summary["interpretation"]["parent_H_gain"] is False
    assert summary["interpretation"]["parent_H_acquisition"] is False
    assert summary["interpretation"]["parent_H_protection"] is True
    assert summary["validation"]["all_exact_accounting_gates_passed"]


@pytest.mark.parametrize("key,value", [("model_seeds", [17601]), ("training_trials", 800),
    ("numeric_tolerance", .1), ("bridge_eligibility", "all_trials"),
    ("phase_definition", "known_after_count_equals_four"), ("new_confirmatory_hypothesis", True),
    ("baseline_commit", "0" * 40), ("contrast_controls", ["balanced"]),
    ("bridge_bits", ["direction", "scalar", "state"]), ("destination_groups", ["other"])])
def test_frozen_descriptive_scope_cannot_select_favorable_budget_or_subsets(key, value):
    cfg = deepcopy(study().SCIENCE)
    cfg[key] = value
    with pytest.raises(ValueError):
        study().validate_config(cfg)


@pytest.mark.parametrize("mutation", ["state_shape", "direction_shape", "scalar_shape", "nan_state",
    "infinite_scalar", "complex_direction", "invalid_sign", "string_sign"])
def test_bridge_rejects_invalid_axes_nonfinite_and_nonreal_inputs(bridge_arguments, mutation):
    values = [v.copy() for v in bridge_arguments]
    if mutation == "state_shape": values[1] = values[1][:3]
    elif mutation == "direction_shape": values[5] = values[5][:3]
    elif mutation == "scalar_shape": values[2] = np.array([.1])
    elif mutation == "nan_state": values[0][0] = np.nan
    elif mutation == "infinite_scalar": values[2] = np.asarray(np.inf)
    elif mutation == "complex_direction": values[5] = values[5].astype(complex)
    elif mutation == "invalid_sign": values[6][0] = 0
    else: values[6] = values[6].astype(str)
    with pytest.raises(ValueError):
        study().bridge_cells(*values)


@pytest.mark.parametrize("cells", [np.zeros(8), np.zeros((7, 4)), np.full((8, 4), np.nan),
    np.full((8, 4), np.inf), np.zeros((8, 4), complex), np.full((8, 4), "0")])
def test_shapley_rejects_incomplete_and_nonfinite_tables(cells):
    with pytest.raises(ValueError):
        study().shapley3(cells)


@pytest.mark.parametrize("which,key", [("parent", "trial_correct_probability_delta"),
    ("parent", "trial_aligned_margin_delta"), ("parent", "checkpoint_heads"),
    ("parent", "checkpoint_probabilities"), ("accounting", "probability_changes"),
    ("accounting", "margin_changes"), ("accounting", "gated_phase_before")])
def test_changed_parent_arithmetic_and_common_phase_evidence_are_rejected(which, key):
    cfg, parent, accounting = invented_saved_paths()
    item = parent if which == "parent" else accounting
    item[key].flat[0] += 1
    with pytest.raises(ValueError):
        study().analyze(cfg, parent, accounting)
