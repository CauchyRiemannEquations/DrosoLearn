"""Invented algebra only; no actual study graphs, seeds, schedules or outcomes."""
import numpy as np
import pytest

from drosolearn.update_budget_geometry import UPDATES, update_budget_geometry


def test_registered_three_direction_order_and_array_axes():
    result = update_budget_geometry(np.eye(4) + .3)
    assert UPDATES == ("shared", "projected_own", "projected_l2")
    assert result["updates"] == UPDATES
    assert result["directions"].shape == (3, 4, 4)
    assert result["influences"].shape == (3, 4, 4)
    assert result["direction_norms"].shape == (3, 4)
    assert result["own_influence"].shape == (3, 4)
    assert result["spillover_squared"].shape == (3, 4)


def test_common_scalar_matches_l2_but_not_own_score_for_correlated_rows():
    result = update_budget_geometry(np.eye(4) + .7)
    assert result["numeric_budget_qualified"]
    assert result["projected_l2_exact_isolation_qualified"]
    assert np.allclose(result["direction_norms"][[0, 2]], 1.)
    assert np.all(result["direction_norms"][1] > 1.)
    assert np.allclose(result["own_influence"][:2], 1.)
    assert np.all(result["own_influence"][2] < 1.)
    assert np.allclose(result["own_influence"][2], 1. / result["direction_norms"][1])
    for scalar in (0., .05, -.03):
        movement = np.linalg.norm(scalar * result["directions"], axis=2)
        assert np.allclose(movement[[0, 2]], abs(scalar))


def test_projected_l2_keeps_zero_off_input_effect_in_full_rank_case():
    result = update_budget_geometry(np.array([[1., .4, 0., 0.], [0., 1., .6, 0.],
                                             [0., 0., 1., .2], [.3, 0., 0., 1.]]))
    assert result["projected_l2_exact_isolation_qualified"]
    assert np.allclose(result["spillover_squared"][1:], 0., atol=1e-24)
    assert np.allclose(result["influences"][1], np.eye(4), atol=1e-12)
    expected = np.diag(1. / result["direction_norms"][1])
    assert np.allclose(result["influences"][2], expected, atol=1e-12)


def test_orthogonal_features_make_all_three_updates_identical():
    result = update_budget_geometry(np.eye(4))
    for direction in result["directions"]:
        assert np.array_equal(direction, np.eye(4))
    assert np.array_equal(result["own_influence"], np.ones((3, 4)))
    assert result["conditional_l2_match_residual"] == 0.


def test_rank_deficiency_has_explicit_residual_spillover_not_exact_isolation():
    result = update_budget_geometry(np.array([[1., 0.], [1., 1.], [0., 1.], [2., 1.]]))
    assert result["numeric_budget_qualified"]
    assert not result["projected_l2_exact_isolation_qualified"]
    assert np.any(result["spillover_squared"][2] > .1)
    assert np.allclose(result["direction_norms"][[0, 2]], 1.)
    assert np.allclose(result["own_influence"][1], 1.)


def test_influence_axes_are_updated_source_then_probed_target():
    result = update_budget_geometry(np.array([[1., 0., 0.], [1., 1., 0.],
                                             [0., 1., 1.], [2., 0., 1.]]))
    d = result["base_geometry"]["unit_rows"]
    for update in range(3):
        for source in range(4):
            assert np.allclose(d @ result["directions"][update, source],
                               result["influences"][update, source])
    assert not np.allclose(result["influences"][2], result["influences"][2].T)


def test_autonomous_scalars_do_not_match_actual_realized_cost():
    result = update_budget_geometry(np.eye(4) + .2)
    # The same action/reward with different learned probabilities changes g.
    scalar_shared = .1 * (1. - .2) * 1.
    scalar_l2 = .1 * (1. - .8) * 1.
    shared_cost = np.linalg.norm(scalar_shared * result["directions"][0, 0])
    projected_cost = np.linalg.norm(scalar_l2 * result["directions"][2, 0])
    assert shared_cost == pytest.approx(.08)
    assert projected_cost == pytest.approx(.02)
    assert result["equal_conditional_scalar_l2_only"]
    assert not result["equal_realized_cumulative_l2_claimed"]


def test_unit_directions_give_eta_per_trial_and_n_eta_cumulative_upper_bound():
    result = update_budget_geometry(np.eye(4) + .5)
    # Finite hand-picked hypothetical trials; no task RNG or actual seed.
    probabilities = np.array([0., .2, .5, .9, 1.])
    actions = np.array([1., 1., 0., 1., 0.])
    rewards = np.array([-1., 1., 0., -1., 0.])
    eta = .1
    scalars = eta * (actions - probabilities) * rewards
    for update in (0, 2):
        costs = np.array([np.linalg.norm(scalar * result["directions"][update, i % 4])
                          for i, scalar in enumerate(scalars)])
        assert np.all(costs <= eta + 1e-12)
        assert costs.sum() <= len(scalars) * eta + 1e-12


def test_response_per_unit_cost_is_invariant_to_own_vs_l2_scaling():
    result = update_budget_geometry(np.eye(4) + .8)
    assert np.allclose(result["own_response_per_unit_l2"][1],
                       result["own_response_per_unit_l2"][2])
    assert np.allclose(result["own_response_per_unit_l2"][0], 1.)
    assert np.all(result["own_response_per_unit_l2"][2] < 1.)


def test_row_scaling_does_not_change_budget_directions():
    rows = np.eye(4) + .3
    initial = update_budget_geometry(rows)
    scaled = update_budget_geometry(rows * np.array([.2, .7, 2., 9.])[:, None])
    for key in ("directions", "influences", "direction_norms", "own_influence"):
        assert np.allclose(initial[key], scaled[key])


def test_condition_gate_changes_claim_only_not_predefined_direction():
    rows = np.array([[1., 0., 0., 0.], [1., 1e-4, 0., 0.],
                     [0., 0., 1., 0.], [0., 0., 1., 1e-4]])
    loose = update_budget_geometry(rows, exact_condition_max=1e6)
    strict = update_budget_geometry(rows, exact_condition_max=100.)
    assert loose["projected_l2_exact_isolation_qualified"]
    assert not strict["projected_l2_exact_isolation_qualified"]
    assert strict["numeric_budget_qualified"]
    assert np.array_equal(loose["directions"], strict["directions"])


def test_invalid_and_zero_input_rows_rejected_before_budget_construction():
    for rows in ([], [[0., 0.]], [[1., np.nan]], [[1., 0.], [0., 0.]]):
        with pytest.raises(ValueError):
            update_budget_geometry(rows)


def test_helper_preserves_input_and_exposes_base_frozen_geometry():
    rows = np.eye(4) + .4
    original = rows.copy()
    result = update_budget_geometry(rows)
    assert np.array_equal(rows, original)
    assert np.array_equal(result["base_geometry"]["directions"], result["directions"][1])
    assert np.array_equal(result["base_geometry"]["unit_rows"], result["directions"][0])
    for key in ("unit_norm_residual", "conditional_l2_match_residual",
                "l2_normalization_reconstruction_residual", "own_scaled_match_residual",
                "projected_own_norm_lower_bound_residual", "projected_l2_own_cap_residual"):
        assert result[key] <= 1e-12
