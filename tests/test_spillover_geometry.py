"""Finite invented matrices only; no actual graphs, study seeds or task RNG."""
import numpy as np
import pytest

from drosolearn.spillover_geometry import ALPHAS, spillover_geometry
from drosolearn.update_budget_geometry import update_budget_geometry


def test_predeclared_alphas_and_axes():
    result = spillover_geometry(np.eye(4) + .3)
    assert ALPHAS == (0., .25, .5, .75, 1.)
    assert np.array_equal(result["alphas"], ALPHAS)
    assert result["directions"].shape == (5, 4, 4)
    assert result["influences"].shape == (5, 4, 4)
    for key in ("direction_norms", "own_influence", "spillover_squared",
                "own_response_per_unit_l2", "mixture_norms",
                "analytic_mixture_norms", "analytic_own_influence"):
        assert result[key].shape == (5, 4)


def test_exact_frozen_endpoints():
    rows = np.eye(4) + .7
    old = update_budget_geometry(rows)
    result = spillover_geometry(rows)
    assert np.array_equal(result["directions"][0], old["directions"][0])
    assert np.array_equal(result["directions"][-1], old["directions"][2])
    assert np.allclose(result["influences"][[0, -1]], old["influences"][[0, 2]])


def test_each_direction_has_unit_length_and_same_conditional_scalar_cost():
    result = spillover_geometry(np.eye(4) + .8)
    assert result["numeric_spillover_qualified"]
    for scalar in (0., -.05, .03, .1):
        movement = np.linalg.norm(scalar * result["directions"], axis=2)
        assert np.allclose(movement, abs(scalar))


def test_central_direction_is_normalized_endpoint_bisector():
    result = spillover_geometry(np.eye(4) + .6)
    midpoint = result["directions"][0] + result["directions"][-1]
    midpoint /= np.linalg.norm(midpoint, axis=1)[:, None]
    assert np.allclose(result["directions"][2], midpoint)


def test_analytic_norm_and_own_response():
    result = spillover_geometry(np.eye(4) + .3)
    alpha = np.array(ALPHAS)[:, None]
    c = result["endpoint_cosines"][None, :]
    q = np.sqrt((1. - alpha) ** 2 + alpha ** 2 + 2 * alpha * (1. - alpha) * c)
    assert np.allclose(result["mixture_norms"], q)
    assert np.allclose(result["own_influence"], ((1 - alpha) + alpha * c) / q)
    assert np.all(q ** 2 >= .5)
    assert np.all(c > 0.)
    assert np.all(np.diff(result["own_influence"], axis=0) <= 1e-12)


def test_full_rank_spillover_has_exact_endpoint_scaling():
    result = spillover_geometry(np.eye(4) + .4)
    assert result["projected_l2_exact_isolation_qualified"]
    alpha = np.array(ALPHAS)[:, None]
    expected = (((1. - alpha) / result["mixture_norms"]) ** 2
                * result["spillover_squared"][0:1])
    assert np.allclose(result["spillover_squared"], expected, atol=1e-12)
    assert np.all(np.diff(result["spillover_squared"], axis=0) <= 1e-12)
    assert np.allclose(result["spillover_squared"][-1], 0., atol=1e-24)


def test_rank_deficiency_preserves_its_residual_spillover():
    result = spillover_geometry(np.array([[1., 0.], [1., 1.], [0., 1.], [2., 1.]]))
    assert result["numeric_spillover_qualified"]
    assert not result["projected_l2_exact_isolation_qualified"]
    assert np.any(result["spillover_squared"][-1] > .1)
    assert not result["alpha_is_fraction_of_spillover_removed"]
    assert not result["all_unit_directions_optimized"]


def test_orthogonal_features_have_identical_five_directions():
    result = spillover_geometry(np.eye(4))
    for direction in result["directions"]:
        assert np.array_equal(direction, np.eye(4))
    assert np.array_equal(result["spillover_squared"], np.zeros((5, 4)))
    assert np.array_equal(result["own_influence"], np.ones((5, 4)))


def test_influence_axes_are_source_then_probed_input():
    rows = np.array([[1., 0., 0.], [1., 1., 0.], [0., 1., 1.], [2., 0., 1.]])
    result = spillover_geometry(rows)
    d = result["base_budget"]["base_geometry"]["unit_rows"]
    for a in range(5):
        for source in range(4):
            assert np.allclose(d @ result["directions"][a, source],
                               result["influences"][a, source])
    assert not np.allclose(result["influences"][-1], result["influences"][-1].T)


def test_common_bounds_do_not_equalize_autonomous_total_costs():
    result = spillover_geometry(np.eye(4) + .2)
    own_direction = result["directions"][0, 0]
    interior_direction = result["directions"][2, 0]
    first = np.linalg.norm(.1 * (1 - .2) * own_direction)
    second = np.linalg.norm(.1 * (1 - .8) * interior_direction)
    assert first == pytest.approx(.08)
    assert second == pytest.approx(.02)
    assert result["equal_conditional_scalar_l2_only"]
    assert not result["equal_realized_cumulative_l2_claimed"]


def test_fixed_eta_step_and_path_upper_bounds():
    result = spillover_geometry(np.eye(4) + .5)
    probabilities = np.array([0., .2, .5, .9, 1.])
    actions = np.array([1., 1., 0., 1., 0.])
    rewards = np.array([-1., 1., 0., -1., 0.])
    scalars = .1 * (actions - probabilities) * rewards
    for a in range(5):
        costs = [np.linalg.norm(g * result["directions"][a, i % 4])
                 for i, g in enumerate(scalars)]
        assert max(costs) <= .1 + 1e-12
        assert sum(costs) <= .1 * len(scalars) + 1e-12


def test_row_scaling_preserves_all_direction_and_influence_outputs():
    rows = np.eye(4) + .25
    initial = spillover_geometry(rows)
    scaled = spillover_geometry(rows * np.array([.2, .7, 2., 9.])[:, None])
    for key in ("directions", "influences", "direction_norms", "own_influence",
                "spillover_squared", "mixture_norms"):
        assert np.allclose(initial[key], scaled[key])


def test_stricter_exact_condition_gate_does_not_change_predefined_directions():
    rows = np.array([[1., 0., 0., 0.], [1., 1e-4, 0., 0.],
                     [0., 0., 1., 0.], [0., 0., 1., 1e-4]])
    loose = spillover_geometry(rows, exact_condition_max=1e6)
    strict = spillover_geometry(rows, exact_condition_max=100.)
    assert loose["projected_l2_exact_isolation_qualified"]
    assert not strict["projected_l2_exact_isolation_qualified"]
    assert strict["numeric_spillover_qualified"]
    assert np.array_equal(loose["directions"], strict["directions"])


@pytest.mark.parametrize("alphas", [[], [0., np.nan], [-.1, .5], [.5, 1.1],
                                   [.5, .5], [.5, 0.], [[0., 1.]]])
def test_invalid_alpha_family_rejected(alphas):
    with pytest.raises(ValueError):
        spillover_geometry(np.eye(4), alphas=alphas)


@pytest.mark.parametrize("rows", [[], [[0., 0.]], [[1., np.nan]],
                                 [[1., 0.], [0., 0.]]])
def test_invalid_or_zero_features_rejected(rows):
    with pytest.raises(ValueError):
        spillover_geometry(rows)


def test_inputs_unchanged_and_all_registered_static_gates_pass():
    rows = np.eye(4) + .4
    original = rows.copy()
    result = spillover_geometry(rows)
    assert np.array_equal(rows, original)
    for key, value in result.items():
        if key.endswith("_residual"):
            assert value <= 1e-12
