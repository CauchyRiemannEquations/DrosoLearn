"""Invented finite matrices only: no actual scientific graph, seeds or draws."""
import numpy as np
import pytest

from drosolearn.representation_geometry import (geometry, hard_margin_witness,
                                               matrix_diagnostics, raw_parallelogram_certificate)


def test_full_rank_dual_matches_own_and_eliminates_other_influence():
    d = np.eye(4) + .3 * np.ones((4, 4))
    result = geometry(d)
    assert np.allclose(np.linalg.norm(result["unit_rows"], axis=1), 1.)
    assert result["exact_isolation_qualified"]
    assert np.allclose(result["influence_projected"], np.eye(4), atol=1e-12)
    assert np.allclose(np.diag(result["influence_shared"]), 1.)
    assert result["raw_diagnostics"]["full_row_condition"] == pytest.approx(2.2)


def test_influence_orientation_is_update_source_by_probe_target():
    result = geometry(np.array([[1., 0.], [1., 1.], [0., 1.], [2., 1.]]))
    for updated in range(4):
        response = result["unit_rows"] @ result["directions"][updated]
        assert np.allclose(response, result["influence_projected"][updated])
    assert not np.allclose(result["influence_projected"], result["influence_projected"].T)


def test_rank_three_asymmetric_projected_influence_catches_transpose_errors():
    result = geometry(np.array([[1., 0., 0.], [1., 1., 0.], [0., 1., 1.], [2., 0., 1.]]))
    assert result["unit_diagnostics"]["retained_rank"] == 3
    assert not np.allclose(result["influence_projected"], result["influence_projected"].T)
    for updated in range(4):
        assert np.allclose(result["influence_projected"][updated],
                           result["unit_rows"] @ result["directions"][updated])


def test_exact_raw_certificate_uses_integer_arithmetic_and_preserves_identity():
    result = raw_parallelogram_certificate()
    rows = result["scaled_integer_rows"]
    assert np.array_equal(result["integer_relation_residual"], np.zeros(7, dtype=int))
    assert len({tuple(row) for row in rows}) == 4
    assert result["rank_minor_determinant"] == -1
    assert result["affine_minor_determinant"] == 1
    assert result["exact_row_rank"] == 3 and result["exact_affine_dimension"] == 2
    assert np.sum(np.sign(result["three_correct_example_scaled_logits"]) == np.array([1, 1, -1, -1])) == 3
    assert result["distinct_inputs_preserve_identity"]
    assert result["nonlinear_feature_separation_not_ruled_out"]


def test_rank_deficient_projection_has_matched_own_but_unavoidable_spillover():
    result = geometry(np.ones((4, 3)))
    assert result["unit_diagnostics"]["retained_rank"] == 1
    assert result["unit_diagnostics"]["full_row_condition"] is None
    assert result["numeric_update_qualified"]
    assert not result["exact_isolation_qualified"]
    assert np.allclose(result["influence_projected"], np.ones((4, 4)))
    assert np.allclose(result["projected_spillover_squared"], 3.)
    assert np.allclose(result["minimum_spillover_squared_analytic"], 3.)


def test_projection_minimizes_squared_spillover_among_feasible_responses():
    d = np.array([[1., 0., 0.], [1., 1., 0.], [0., 1., 0.], [2., 1., 0.]])
    result = geometry(d)
    unit, directions = result["unit_rows"], result["directions"]
    for i in range(4):
        for perturbation in (np.array([.3, -.2, .7]), np.array([-1., 2., 0.])):
            # Remove own-input effect; any remaining direction is feasible.
            perturbation -= (unit[i] @ perturbation) * unit[i]
            alternate = unit @ (directions[i] + perturbation)
            assert alternate[i] == pytest.approx(1.)
            assert np.sum(alternate ** 2) >= np.sum(result["influence_projected"][i] ** 2) - 1e-12


def test_row_normalization_removes_per_input_scale_and_keeps_directions():
    d = np.eye(4) + .2
    base = geometry(d)
    scaled = geometry(d * np.array([.2, 3., 9., .1])[:, None])
    for key in ("unit_rows", "directions", "influence_shared", "influence_projected"):
        assert np.allclose(base[key], scaled[key])


def test_preregistered_truncation_changes_rank_without_undefined_inverse():
    d = np.array([[1., 0., 0., 0.], [1., 1e-14, 0., 0.],
                  [0., 0., 1., 0.], [0., 0., 1., 1e-14]])
    result = geometry(d)
    assert result["unit_diagnostics"]["strict_rank"] == 4
    assert result["unit_diagnostics"]["retained_rank"] == 2
    assert np.all(np.isfinite(result["directions"]))
    assert result["numeric_update_qualified"] and not result["exact_isolation_qualified"]
    assert result["off_diagonal_max"] == pytest.approx(1.)


def test_condition_bound_closes_exact_claim_without_changing_predefined_update():
    d = np.array([[1., 0., 0., 0.], [1., 1e-4, 0., 0.],
                  [0., 0., 1., 0.], [0., 0., 1., 1e-4]])
    loose = geometry(d, exact_condition_max=1e6)
    strict = geometry(d, exact_condition_max=100.)
    assert loose["exact_isolation_qualified"]
    assert not strict["exact_isolation_qualified"]
    assert np.array_equal(loose["directions"], strict["directions"])


@pytest.mark.parametrize("rows", [[], [[0., 0.]], [[1., 0.], [0., 0.]], [[1., np.nan]], [[1., np.inf]]])
def test_invalid_or_zero_feature_norm_rejected(rows):
    with pytest.raises(ValueError):
        geometry(rows)


@pytest.mark.parametrize("rcond", [0., -1., 1., np.nan])
def test_invalid_cutoffs_rejected(rcond):
    with pytest.raises(ValueError):
        geometry(np.eye(4), rcond=rcond)


def test_zero_matrix_rank_diagnostics_are_finite_and_explicitly_undefined():
    result = matrix_diagnostics(np.zeros((4, 7)))
    assert result["strict_rank"] == result["retained_rank"] == 0
    assert result["full_row_condition"] is result["retained_condition"] is None
    assert np.all(result["singular_values"] == 0)


def test_interpolation_capacity_witness_is_analytic_not_reward_learning():
    result = hard_margin_witness(np.eye(4), [1, 1, -1, -1])
    assert result["feasible"]
    assert np.allclose(result["head"], [1, 1, -1, -1])
    assert result["head_l2"] == pytest.approx(2.)
    assert result["examined_subset_count"] == 15
    assert result["analytic_probe_not_reward_learning"]


def test_rank_deficient_but_separable_labels_still_get_a_witness():
    d = np.array([[1., 0.], [2., 0.], [-1., 0.], [-2., 0.]])
    result = hard_margin_witness(d, [1, 1, -1, -1])
    assert result["feasible"]
    assert result["head_l2"] == pytest.approx(1.)
    assert np.all(result["signed_margins"] >= 1. - 1e-12)


def test_invented_intersecting_class_midpoint_gives_no_float_certificate():
    d = np.array([[1., 1., 1.], [-1., -1., 1.], [1., -1., 1.], [-1., 1., 1.]])
    result = hard_margin_witness(d, [1, 1, -1, -1])
    assert not result["feasible"]
    assert result["status"] == "no_feasible_candidate_under_tolerance"
    assert not result["failed_search_is_impossibility_proof"]
    assert result["head_l2"] is None
    assert result["active_subset_mask"] is None


def test_witness_minimum_norm_can_use_a_proper_subset():
    d = np.array([[1., 0.], [2., 0.], [-1., 1.], [-2., 2.]])
    result = hard_margin_witness(d, [1, 1, -1, -1])
    assert result["feasible"]
    assert result["head_l2"] == pytest.approx(1.)
    assert np.allclose(result["head"], [1., 0.])
    assert result["active_subset_mask"] != 15


@pytest.mark.parametrize("labels", [[0, 1, -1, -1], [1, -1], [1, 1, -1, np.nan]])
def test_margin_probe_requires_exact_signed_four_label_contract(labels):
    with pytest.raises(ValueError):
        hard_margin_witness(np.eye(4), labels)


def test_all_diagnostics_do_not_mutate_caller_inputs():
    d = np.eye(4) + .2
    before = d.copy()
    geometry(d)
    hard_margin_witness(d, [1, 1, -1, -1])
    matrix_diagnostics(d)
    assert np.array_equal(d, before)
