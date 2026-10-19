"""Small analytic cones only; no task records, graph files or scientific seeds."""
import numpy as np
import pytest

from drosolearn.cone_geometry import cone_direction


def certify(a, c, info, atol=1e-8):
    """Check the support upper bound directly from the returned dual witness."""
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)
    p = info["projected_objective"]
    w = info["direction"]
    dual = info["multipliers"]
    assert info["numeric_qualified"]
    assert np.isfinite(p).all() and np.isfinite(w).all()
    assert np.isfinite(dual).all()
    assert np.linalg.norm(w) <= 1. + atol
    assert np.all(a @ w >= -atol)
    assert np.all(a @ p >= -atol)
    assert np.all(dual >= -atol)
    np.testing.assert_allclose(p - c, a.T @ dual, atol=atol)
    np.testing.assert_allclose(dual * (a @ p), 0., atol=atol)
    np.testing.assert_allclose(info["objective_upper_bound"], np.linalg.norm(p), atol=atol)
    np.testing.assert_allclose(info["dual_vector"], c + a.T @ dual, atol=atol)
    np.testing.assert_allclose(info["dual_upper_bound"], np.linalg.norm(c + a.T @ dual), atol=atol)
    np.testing.assert_allclose(info["achieved_objective"], c @ w, atol=atol)
    np.testing.assert_allclose(c @ w, np.linalg.norm(p), atol=atol)
    assert abs(info["optimality_gap"]) <= atol
    assert abs(info["dual_optimality_gap"]) <= atol
    assert info["active_subsets_evaluated"] == 2 ** len(a)
    assert all(np.isfinite(x) and x <= atol for x in info["residuals"].values())


def test_empty_constraints_gives_unconstrained_unit_support():
    a = np.empty((0, 3))
    c = np.array([3., 4., 0.])
    info = cone_direction(a, c)
    np.testing.assert_allclose(info["direction"], [.6, .8, 0.])
    np.testing.assert_allclose(info["projected_objective"], c)
    assert info["objective_upper_bound"] == 5.
    assert info["multipliers"].shape == (0,)
    certify(a, c, info)


@pytest.mark.parametrize("c", ([3., -4.], [-3., 4.], [-3., -4.], [3., 4.]))
def test_orthant_has_known_analytic_projection(c):
    c = np.array(c)
    a = np.eye(2)
    expected = np.maximum(c, 0.)
    info = cone_direction(a, c)
    np.testing.assert_allclose(info["projected_objective"], expected, atol=1e-12)
    if np.linalg.norm(expected):
        np.testing.assert_allclose(info["direction"], expected / np.linalg.norm(expected))
    else:
        np.testing.assert_array_equal(info["direction"], [0., 0.])
        assert info["zero_projection"]
    certify(a, c, info)


@pytest.mark.parametrize("c", ([2., 3.], [-2., 3.], [2., -3.], [-2., -3.]))
def test_opposite_halfspaces_force_one_coordinate_to_zero(c):
    a = np.array([[1., 0.], [-1., 0.]])
    c = np.array(c)
    info = cone_direction(a, c)
    np.testing.assert_allclose(info["projected_objective"], [0., c[1]], atol=1e-12)
    np.testing.assert_allclose(info["direction"], [0., np.sign(c[1])], atol=1e-12)
    certify(a, c, info)


def test_oblique_wedge_projection_and_support_are_analytic():
    # x >= |y|, so projecting (0, 2) lands on the y=x boundary at (1, 1).
    a = np.array([[1., -1.], [1., 1.]])
    c = np.array([0., 2.])
    info = cone_direction(a, c)
    np.testing.assert_allclose(info["projected_objective"], [1., 1.], atol=1e-12)
    np.testing.assert_allclose(info["direction"], np.ones(2) / np.sqrt(2), atol=1e-12)
    assert info["objective_upper_bound"] == pytest.approx(np.sqrt(2))
    certify(a, c, info)


def test_four_constraints_force_zero_with_zero_support():
    a = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    c = np.array([2., -3.])
    info = cone_direction(a, c)
    np.testing.assert_allclose(info["projected_objective"], 0., atol=1e-12)
    np.testing.assert_allclose(info["direction"], 0., atol=1e-12)
    assert info["zero_projection"]
    certify(a, c, info)


def test_zero_objective_uses_zero_direction_without_dividing_by_zero():
    a = np.array([[1., 2., 3.], [-1., 0., 2.]])
    info = cone_direction(a, np.zeros(3))
    np.testing.assert_array_equal(info["direction"], np.zeros(3))
    assert info["objective_upper_bound"] == 0.
    assert info["zero_projection"]
    certify(a, np.zeros(3), info)


def test_zero_constraint_row_does_not_change_the_feasible_set():
    a = np.array([[0., 0.], [1., 0.]])
    c = np.array([-3., 4.])
    info = cone_direction(a, c)
    np.testing.assert_allclose(info["direction"], [0., 1.], atol=1e-12)
    certify(a, c, info)


def test_dependent_duplicate_constraints_preserve_projection():
    a = np.array([[1., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 3., 0.]])
    c = np.array([-2., -3., 4.])
    info = cone_direction(a, c)
    np.testing.assert_allclose(info["direction"], [0., 0., 1.], atol=1e-12)
    np.testing.assert_allclose(info["projected_objective"], [0., 0., 4.], atol=1e-12)
    certify(a, c, info)


def test_nearly_opposite_independent_rows_keep_the_small_singular_direction():
    # The row singular ratio is about 5e-7; squaring it for a Gram cutoff
    # would discard the second controllable constraint at rcond=1e-12.
    a = np.array([[1., 0.], [-1., 1e-6]])
    c = np.array([-1., -1.])
    info = cone_direction(a, c)
    np.testing.assert_allclose(info["projected_objective"], 0., atol=1e-10)
    assert info["selected_active_rank"] == 2
    assert info["rank_cutoff_applies_to_active_rows"]
    certify(a, c, info)


def test_nonbinding_constraints_leave_original_objective_direction():
    a = np.array([[1., 0., 0.], [1., 1., 0.], [1., 0., 1.]])
    c = np.array([2., 1., 1.])
    info = cone_direction(a, c)
    np.testing.assert_allclose(info["direction"], c / np.linalg.norm(c), atol=1e-12)
    np.testing.assert_allclose(info["multipliers"], 0., atol=1e-12)
    certify(a, c, info)


def test_feasible_objective_on_boundary_is_not_forced_to_intersection_vertex():
    a = np.array([[1., 0.], [0., 1.]])
    c = np.array([0., 2.])
    info = cone_direction(a, c)
    np.testing.assert_allclose(info["direction"], [0., 1.], atol=1e-12)
    certify(a, c, info)


def test_positive_constraint_rescaling_preserves_direction():
    a = np.array([[1., -1., 0.], [0., 1., -1.], [1., 0., 0.]])
    c = np.array([-2., 1., 3.])
    first = cone_direction(a, c)
    scaled = a * np.array([.2, 7., 3.])[:, None]
    second = cone_direction(scaled, c)
    np.testing.assert_allclose(first["direction"], second["direction"], atol=1e-12)
    np.testing.assert_allclose(first["objective_upper_bound"], second["objective_upper_bound"], atol=1e-12)
    certify(a, c, first)
    certify(scaled, c, second)


def test_constraint_permutation_preserves_projection_and_dual_witness():
    a = np.array([[1., -1., 0.], [0., 1., -1.], [1., 0., 0.]])
    c = np.array([-2., 1., 3.])
    order = [2, 0, 1]
    first = cone_direction(a, c)
    second = cone_direction(a[order], c)
    np.testing.assert_allclose(first["direction"], second["direction"], atol=1e-12)
    certify(a[order], c, second)


def test_orthogonal_output_coordinate_change_preserves_effects():
    a = np.array([[1., -1., 0.], [0., 1., -1.], [1., 0., 0.]])
    c = np.array([-2., 1., 3.])
    q, _ = np.linalg.qr(np.array([[1., 2., 0.], [0., 1., 3.], [2., 0., 1.]]))
    first = cone_direction(a, c)
    second = cone_direction(a @ q, c @ q)
    np.testing.assert_allclose(second["direction"], first["direction"] @ q, atol=1e-12)
    np.testing.assert_allclose((a @ q) @ second["direction"], a @ first["direction"], atol=1e-12)
    certify(a @ q, c @ q, second)


def test_positive_objective_scale_preserves_direction_and_scales_support():
    a = np.array([[1., 0., 0.], [0., 1., 0.]])
    c = np.array([-2., 3., 4.])
    first = cone_direction(a, c)
    second = cone_direction(a, c * 7.)
    np.testing.assert_allclose(first["direction"], second["direction"], atol=1e-12)
    assert second["objective_upper_bound"] == pytest.approx(7 * first["objective_upper_bound"])
    certify(a, c * 7., second)


def test_independent_feasible_candidates_never_exceed_the_dual_bound():
    a = np.array([[1., -.3, .1, 0.], [.2, 1., -.4, .1],
                  [0., .1, 1., -.3], [0., 0., .1, 1.]])
    c = np.array([-1., .7, 1., -.5])
    info = cone_direction(a, c)
    # This fixed development generator is unrelated to any scientific seed.
    rng = np.random.default_rng(203)
    candidates = rng.normal(size=(1000, 4))
    candidates /= np.maximum(1., np.linalg.norm(candidates, axis=1))[:, None]
    feasible = candidates[(candidates @ a.T >= 0.).all(axis=1)]
    assert len(feasible) > 10
    assert np.all(feasible @ c <= info["objective_upper_bound"] + 1e-12)
    assert np.all(feasible @ c <= info["dual_upper_bound"] + 1e-12)
    certify(a, c, info)


def test_own_objective_has_positive_support_with_full_rank_signed_rows():
    d = np.array([[1., 0., 0., 0.], [.6, .8, 0., 0.],
                  [.3, .4, np.sqrt(.75), 0.], [.2, .1, .2, np.sqrt(.91)]])
    signs = np.array([1., 1., -1., -1.])
    a = signs[:, None] * d
    info = cone_direction(a, d[0])
    assert d[0] @ info["direction"] > 0.
    assert info["objective_upper_bound"] > 0.
    certify(a, d[0], info)


def test_including_help_in_objective_can_trade_own_gain_for_help():
    # Both axes are safe. Own-only makes no second-axis progress, balanced does.
    a = np.eye(2)
    own = cone_direction(a, np.array([1., 0.]))
    balanced = cone_direction(a, np.array([.5, .5]))
    assert own["direction"][0] > balanced["direction"][0]
    assert own["direction"][1] < balanced["direction"][1]
    certify(a, [.5, .5], balanced)


def test_unknown_destination_need_not_be_safe_without_its_constraint():
    a = np.array([[1., 0.]])
    c = np.array([1., -1.])
    info = cone_direction(a, c)
    unknown_row = np.array([0., 1.])
    assert unknown_row @ info["direction"] < 0.
    certify(a, c, info)


def test_arguments_are_not_modified_and_outputs_are_own_copies():
    a = np.array([[1., 0.], [0., 1.]])
    c = np.array([-1., 2.])
    before_a, before_c = a.copy(), c.copy()
    info = cone_direction(a, c)
    info["direction"][0] = 42.
    info["projected_objective"][0] = 42.
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(c, before_c)


@pytest.mark.parametrize("a,c", (
    ([1., 0.], [1., 0.]),
    (np.ones((2, 2, 2)), [1., 0.]),
    (np.ones((5, 2)), [1., 0.]),
    (np.ones((1, 0)), []),
    (np.eye(2), [[1., 0.]]),
    (np.eye(2), [1.]),
    ([[1., np.nan]], [1., 0.]),
    ([[1., np.inf]], [1., 0.]),
    (np.eye(2), [np.nan, 0.]),
    (np.eye(2), [np.inf, 0.]),
    ([["a", "b"]], [1., 0.]),
))
def test_invalid_dimensions_or_nonfinite_values_fail_closed(a, c):
    with pytest.raises(ValueError):
        cone_direction(a, c)


@pytest.mark.parametrize("name,value", (
    ("rcond", 0.), ("rcond", -1.), ("rcond", 1.), ("rcond", np.nan),
    ("tolerance", 0.), ("tolerance", -1.), ("tolerance", np.nan),
    ("tolerance", np.inf),
))
def test_invalid_numeric_controls_fail_closed(name, value):
    with pytest.raises(ValueError):
        cone_direction(np.eye(2), [1., 1.], **{name: value})
