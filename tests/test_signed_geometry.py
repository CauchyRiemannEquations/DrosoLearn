"""Invented finite matrices only: no actual result arrays, graphs or seeds."""
import itertools

import numpy as np
import pytest

from drosolearn.signed_geometry import (
    COMPONENTS, checkpoint_drift, clipped_sigmoid, component_masks,
    component_totals, signed_static, signed_step,
)


Y = np.array([1., 1., -1., -1.])
M = np.array([[1., -.2, .3, -.4], [.5, 1., -.6, .7],
              [-.8, .9, 1., -.1], [.2, -.3, .4, 1.]])


def test_static_direction_sign_depends_on_both_labels_and_influence():
    actual = signed_static(M, Y)
    expected = np.array([[1., -.2, -.3, .4], [.5, 1., .6, -.7],
                         [.8, -.9, 1., -.1], [-.2, .3, .4, 1.]])
    assert np.array_equal(actual["target_aligned"], expected)
    assert np.array_equal(actual["helpful"] - actual["harmful"], expected)
    # Same-target negative effects harm, and opposite-target negative effects help.
    assert actual["target_aligned"][0, 1] < 0.
    assert actual["target_aligned"][0, 3] > 0.


def test_matrix_axes_are_source_then_destination_and_not_transposed():
    assert not np.allclose(M, M.T)
    static = signed_static(M, Y)
    step = signed_step(M, Y, 2, -.07, [0., 0., 0., 0.])
    assert step["margin_change"][1] == pytest.approx(-.063)
    assert step["margin_change"][1] != pytest.approx(Y[1] * -.07 * M[1, 2])
    assert np.allclose(step["margin_change"], .07 * static["target_aligned"][2])


def test_component_partition_and_unions():
    masks = component_masks(Y)
    assert tuple(masks) == COMPONENTS
    assert [masks[k].sum() for k in COMPONENTS] == [4, 4, 8, 12, 16]
    assert not (masks["self"] & masks["cross_same"]).any()
    assert not (masks["self"] & masks["cross_opposite"]).any()
    assert not (masks["cross_same"] & masks["cross_opposite"]).any()
    assert np.array_equal(masks["cross_same"] | masks["cross_opposite"], masks["cross_all"])
    assert np.array_equal(masks["self"] | masks["cross_all"], masks["all"])


def test_exact_masses_preserve_small_neutral_terms():
    values = np.array([[1e-10, -2e-10], [.5, -.3]])
    actual = signed_static(values, [1., 1.])
    assert np.array_equal(actual["classification"], [[0, 0], [1, -1]])
    row = actual["components"]["all"]
    assert row["helpful"] == pytest.approx(.5 + 1e-10, abs=1e-13)
    assert row["harmful"] == pytest.approx(.3 + 2e-10, abs=1e-13)
    assert row["net"] == pytest.approx(.2 - 1e-10, abs=1e-13)
    assert row["neutral_count"] == 2
    assert row["exact_positive_count"] == row["exact_negative_count"] == 2
    assert row["neutral_net"] == pytest.approx(-1e-10)
    assert row["neutral_absolute"] == pytest.approx(3e-10)
    assert row["mass_identity_residual"] < 1e-15


def test_neutral_tolerance_is_inclusive_not_a_sign_rewrite():
    result = signed_static([[1e-8, -1e-8], [0., 1e-8 + 1e-15]], [1., 1.])
    assert np.array_equal(result["classification"], [[0, 0], [0, 1]])
    assert result["components"]["all"]["exact_negative_count"] == 1
    assert result["components"]["all"]["neutral_count"] == 3


@pytest.mark.parametrize("source,scalar", itertools.product(range(4), (0., -.09, .04)))
def test_actual_change_matches_a_direct_saved_head_step(source, scalar):
    # Invented feature coordinates, a head and directions give an independent
    # dot-product reference instead of copying the effect helper's formula.
    features = np.array([[1., .2, .1], [.1, 1., .3], [-.4, .7, 1.], [.6, -.2, 1.]])
    directions = np.array([[1., 0., .1], [.3, -.5, 1.], [-1., .2, 0.], [0., .1, -.7]])
    head = np.array([.25, -.37, .18])
    result = signed_step(directions @ features.T, Y, source, scalar, features @ head)
    new_head = head + scalar * directions[source]
    before = Y * (features @ head)
    after = Y * (features @ new_head)
    assert np.allclose(result["margin_change"], after - before, atol=1e-15)
    assert np.allclose(result["post_margins"], after, atol=1e-15)
    assert np.allclose(result["probability_change"],
                       clipped_sigmoid(after) - clipped_sigmoid(before), atol=1e-15)
    for kind in ("margin_components", "probability_components"):
        d = result[kind]
        assert d["all"]["net"] == pytest.approx(sum(d[k]["net"] for k in COMPONENTS[:3]))
        assert d["cross_all"]["net"] == pytest.approx(
            d["cross_same"]["net"] + d["cross_opposite"]["net"])


def test_wrong_sign_source_scalar_reverses_help_and_harm():
    good = signed_step(M, Y, 0, .05, np.zeros(4))
    wrong = signed_step(M, Y, 0, -.05, np.zeros(4))
    assert np.array_equal(good["margin_change"], -wrong["margin_change"])
    assert good["margin_components"]["self"]["helpful"] > 0.
    assert wrong["margin_components"]["self"]["harmful"] > 0.
    assert good["margin_components"]["cross_same"]["harmful"] > 0.


def test_reward_only_go_and_nogo_scales_are_distinct():
    logits = np.array([.4, -.7, .8, -.3])
    p = clipped_sigmoid(logits)
    for source in range(4):
        go_scalar = .1 * (1. - p[source]) * Y[source]
        go = signed_step(M, Y, source, go_scalar, logits)
        nogo = signed_step(M, Y, source, .1 * (0. - p[source]) * 0., logits)
        assert go["margin_components"]["self"]["helpful"] > 0.
        assert np.array_equal(nogo["margin_change"], np.zeros(4))
        assert np.array_equal(nogo["probability_change"], np.zeros(4))


def test_nonlinear_probability_changes_depend_on_saved_prestate():
    center = signed_step(np.eye(4), Y, 0, .1, np.zeros(4))
    saturated = signed_step(np.eye(4), Y, 0, .1, [10., 0., 0., 0.])
    assert np.array_equal(center["margin_change"], saturated["margin_change"])
    assert center["probability_change"][0] > 1000 * saturated["probability_change"][0]


def test_clipping_uses_post_raw_margin_not_clipped_pre_margin():
    step = signed_step(np.eye(2), [1., -1.], 0, -5., [43., -43.])
    expected = clipped_sigmoid(38.) - clipped_sigmoid(43.)
    wrong = clipped_sigmoid(35.) - clipped_sigmoid(40.)
    assert step["probability_change"][0] == expected
    assert step["probability_change"][0] != wrong


def test_probability_and_margin_effects_both_telescope_on_fixed_path():
    logits = np.array([.2, -.1, .4, -.3])
    start_logits = logits.copy()
    margin_sum = np.zeros(4)
    probability_sum = np.zeros(4)
    split_margin = {k: 0. for k in COMPONENTS[:3]}
    split_probability = {k: 0. for k in COMPONENTS[:3]}
    for source, scalar in [(0, .05), (3, -.08), (1, .02), (2, 0.), (2, -.04)]:
        step = signed_step(M, Y, source, scalar, logits)
        logits += scalar * M[source]
        margin_sum += step["margin_change"]
        probability_sum += step["probability_change"]
        for k in COMPONENTS[:3]:
            split_margin[k] += step["margin_components"][k]["net"]
            split_probability[k] += step["probability_components"][k]["net"]
    assert np.allclose(margin_sum, Y * (logits - start_logits), atol=1e-15)
    assert np.allclose(probability_sum, clipped_sigmoid(Y * logits) - clipped_sigmoid(Y * start_logits), atol=1e-15)
    assert sum(split_margin.values()) == pytest.approx(margin_sum.sum())
    assert sum(split_probability.values()) == pytest.approx(probability_sum.sum())


def test_saved_path_probability_partition_is_not_removed_event_rollout():
    # At a different head state, precisely the same later score change can have
    # a different probability effect. Removing an earlier update changes state.
    first = signed_step(M, Y, 0, 3., np.zeros(4))
    after_first = first["post_margins"] * Y
    later_on_saved = signed_step(M, Y, 1, 2., after_first)
    later_without_first = signed_step(M, Y, 1, 2., np.zeros(4))
    assert np.array_equal(later_on_saved["margin_change"], later_without_first["margin_change"])
    assert not np.allclose(later_on_saved["probability_change"], later_without_first["probability_change"])


def test_checkpoint_expectations_match_all_eight_enumerated_possibilities():
    logits = np.array([.4, -.7, .8, -.3])
    eta = .3
    result = checkpoint_drift(M, Y, logits, eta)
    p = clipped_sigmoid(logits)
    expected_margin, expected_probability = np.zeros(4), np.zeros(4)
    for source, action in itertools.product(range(4), (0, 1)):
        weight = (p[source] if action else 1. - p[source]) / 4
        reward = Y[source] if action else 0.
        scalar = eta * (action - p[source]) * reward
        new_logits = logits + scalar * M[source]
        expected_margin += weight * Y * (new_logits - logits)
        expected_probability += weight * (
            clipped_sigmoid(Y * new_logits) - clipped_sigmoid(Y * logits))
    assert np.allclose(result["expected_margin_drift"], expected_margin, atol=1e-15)
    assert np.allclose(result["expected_probability_drift"], expected_probability, atol=1e-15)
    assert np.allclose(result["margin_contributions"].sum(axis=0), expected_margin)
    assert np.allclose(result["probability_contributions"].sum(axis=0), expected_probability)
    assert np.allclose(result["expected_scalars_given_source"], eta * p * (1-p) * Y)


def test_expected_probability_is_not_sigmoid_of_expected_margin():
    logits = np.array([.9, -.7, .6, -.3])
    result = checkpoint_drift(M * 8., Y, logits, .7)
    invalid_substitute = clipped_sigmoid(Y * logits + result["expected_margin_drift"]) - clipped_sigmoid(Y * logits)
    assert not np.allclose(result["expected_probability_drift"], invalid_substitute, atol=1e-5)
    assert result["source_probability"] == .25
    assert not result["complete_future_rollout"]


def test_checkpoint_zero_eta_and_orthogonal_no_cross_effect():
    zero = checkpoint_drift(np.eye(4), Y, np.zeros(4), 0.)
    assert np.array_equal(zero["expected_margin_drift"], np.zeros(4))
    assert np.array_equal(zero["expected_probability_drift"], np.zeros(4))
    active = checkpoint_drift(np.eye(4), Y, np.zeros(4), .1)
    assert np.allclose(active["expected_margin_drift"], np.full(4, .00625))
    assert active["margin_components"]["cross_all"]["absolute"] == 0.
    assert active["probability_components"]["cross_all"]["absolute"] == 0.


def test_functions_do_not_mutate_saved_inputs():
    matrix, labels, logits = M.copy(), Y.copy(), np.array([.1, -.2, .3, -.4])
    original = [v.copy() for v in (matrix, labels, logits)]
    signed_static(matrix, labels)
    signed_step(matrix, labels, 0, .05, logits)
    checkpoint_drift(matrix, labels, logits, .1)
    for actual, before in zip((matrix, labels, logits), original):
        assert np.array_equal(actual, before)


@pytest.mark.parametrize("matrix,labels", [([], []), ([[1., 0.]], [1.]),
    ([[1., np.nan], [0., 1.]], [1., -1.]), (np.eye(2), [0., 1.]),
    (np.eye(2), [1.]), (np.eye(2), [1., np.inf]), (np.eye(2), [[1., -1.]])])
def test_bad_matrix_or_labels_rejected(matrix, labels):
    with pytest.raises(ValueError):
        signed_static(matrix, labels)


@pytest.mark.parametrize("tolerance", [0., -1., np.nan, np.inf, True, [1e-8]])
def test_bad_tolerance_rejected(tolerance):
    with pytest.raises(ValueError):
        signed_static(M, Y, tolerance)


@pytest.mark.parametrize("source", [-1, 4, 1.5, True])
def test_bad_source_rejected(source):
    with pytest.raises(ValueError):
        signed_step(M, Y, source, .1, np.zeros(4))


@pytest.mark.parametrize("scalar", [np.nan, np.inf, True, [.1]])
def test_bad_saved_scalar_rejected(scalar):
    with pytest.raises(ValueError):
        signed_step(M, Y, 0, scalar, np.zeros(4))


@pytest.mark.parametrize("logits", [[0., 0.], [0., 0., 0., np.nan], [[0., 0., 0., 0.]]])
def test_bad_logits_rejected(logits):
    with pytest.raises(ValueError):
        signed_step(M, Y, 0, .1, logits)
    with pytest.raises(ValueError):
        checkpoint_drift(M, Y, logits, .1)


@pytest.mark.parametrize("eta", [-.1, np.nan, np.inf, True, [.1]])
def test_bad_eta_rejected(eta):
    with pytest.raises(ValueError):
        checkpoint_drift(M, Y, np.zeros(4), eta)


def test_bad_component_masks_and_nonfinite_effects_rejected():
    masks = component_masks(Y)
    with pytest.raises(ValueError):
        component_totals(M, {k: v for k, v in masks.items() if k != "all"})
    masks["self"] = masks["self"].astype(int)
    with pytest.raises(ValueError):
        component_totals(M, masks)
    with pytest.raises(ValueError):
        component_totals(np.full((4, 4), np.nan), component_masks(Y))


def test_probability_representation_agrees_with_parent_within_tolerance():
    logits = np.array([-100., -3.2, .4, 100.])
    parent = np.where(Y == 1., clipped_sigmoid(logits), 1. - clipped_sigmoid(logits))
    assert np.allclose(clipped_sigmoid(Y * logits), parent, atol=1e-8, rtol=0.)
