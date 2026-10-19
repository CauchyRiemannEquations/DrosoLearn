"""A predeclared, label-free family between two unit-L2 update endpoints.

Alpha is a direction mixture coefficient, not a fraction of interference removed.
Every resulting direction has unit length. Thus a common update scalar gives
equal movement, but autonomous learning need not use equal total path lengths.
No graph construction, reward labels, scientific seed or task RNG occurs here.
"""
from __future__ import annotations

import numpy as np

from drosolearn.update_budget_geometry import update_budget_geometry


ALPHAS = (0., .25, .5, .75, 1.)


def spillover_geometry(feature_rows, alphas=ALPHAS, rcond=1e-12,
                       tolerance=1e-8, exact_condition_max=1e6):
    """Normalize convex combinations of shared and projected-L2 directions.

    Arrays have axes ``(alpha, source input, head coordinate)`` for directions
    and ``(alpha, source input, probed input)`` for influences. An influence
    omits the scalar ``eta * (action - probability) * obtained_reward``.

    If d is a shared unit row, u its own-score-matched projected direction,
    and v=u/||u||, then c=d dot v=1/||u||>0. The mixture norm squared is
    (1-alpha)^2 + alpha^2 + 2*alpha*(1-alpha)*c >= 1/2. No opposing-vector
    cancellation or outcome-dependent fallback is therefore required. This
    family is not asserted to optimize all possible unit-L2 directions.
    """
    alpha = np.asarray(alphas, dtype=float)
    if (alpha.ndim != 1 or len(alpha) == 0 or not np.isfinite(alpha).all()
            or np.any(alpha < 0.) or np.any(alpha > 1.)
            or np.any(np.diff(alpha) <= 0.)):
        raise ValueError("Alphas must be finite, strictly increasing values in [0, 1]")
    base = update_budget_geometry(feature_rows, rcond=rcond, tolerance=tolerance,
                                  exact_condition_max=exact_condition_max)
    shared = base["directions"][0]
    projected = base["directions"][2]
    endpoint_cosines = np.sum(shared * projected, axis=1)
    mixture = ((1. - alpha[:, None, None]) * shared[None, :, :]
               + alpha[:, None, None] * projected[None, :, :])
    mixture_norms = np.linalg.norm(mixture, axis=2)
    if not np.isfinite(mixture_norms).all() or not np.all(mixture_norms > 0.):
        raise ValueError("Mixture directions require finite positive L2 lengths")
    directions = mixture / mixture_norms[:, :, None]
    # Keep the exact frozen endpoint direction arrays, including their floating
    # representation, rather than introducing a redundant normalization there.
    directions[alpha == 0.] = shared
    directions[alpha == 1.] = projected
    influences = directions @ shared.T
    norms = np.linalg.norm(directions, axis=2)
    own = np.diagonal(influences, axis1=1, axis2=2).copy()
    off = influences.copy()
    indices = np.arange(shared.shape[0])
    off[:, indices, indices] = 0.
    analytic_norms = np.sqrt((1. - alpha[:, None]) ** 2 + alpha[:, None] ** 2
                             + 2. * alpha[:, None] * (1. - alpha[:, None])
                             * endpoint_cosines[None, :])
    analytic_own = ((1. - alpha[:, None])
                    + alpha[:, None] * endpoint_cosines[None, :]) / analytic_norms
    unit_residual = float(np.max(np.abs(norms - 1.)))
    conditional_residual = float(np.max(np.abs(norms - norms[0:1])))
    reconstruction_residual = float(np.max(np.abs(
        directions * mixture_norms[:, :, None] - mixture)))
    norm_formula_residual = float(np.max(np.abs(mixture_norms - analytic_norms)))
    own_formula_residual = float(np.max(np.abs(own - analytic_own)))
    cosine_formula_residual = float(np.max(np.abs(
        endpoint_cosines - 1. / base["direction_norms"][1])))
    lower_bound_residual = float(np.max(np.maximum(
        .5 - mixture_norms ** 2, 0.)))
    own_monotonic_residual = (float(np.max(np.maximum(np.diff(own, axis=0), 0.)))
                              if len(alpha) > 1 else 0.)
    cosine_range_residual = float(np.max(np.maximum(
        endpoint_cosines - 1., 0.)))
    residuals = (unit_residual, conditional_residual, reconstruction_residual,
                 norm_formula_residual, own_formula_residual,
                 cosine_formula_residual, lower_bound_residual,
                 own_monotonic_residual, cosine_range_residual)
    finite = all(np.isfinite(array).all() for array in (
        directions, influences, norms, own, analytic_norms, analytic_own))
    qualified = bool(base["numeric_budget_qualified"] and finite
                     and np.all(endpoint_cosines > 0.)
                     and all(value <= tolerance for value in residuals))
    return {
        "base_budget": base, "alphas": alpha,
        "directions": directions, "influences": influences,
        "direction_norms": norms, "own_influence": own,
        "spillover_squared": np.sum(off ** 2, axis=2),
        "own_response_per_unit_l2": own / norms,
        "mixture_norms": mixture_norms,
        "endpoint_cosines": endpoint_cosines,
        "analytic_mixture_norms": analytic_norms,
        "analytic_own_influence": analytic_own,
        "unit_norm_residual": unit_residual,
        "conditional_l2_match_residual": conditional_residual,
        "normalization_reconstruction_residual": reconstruction_residual,
        "mixture_norm_formula_residual": norm_formula_residual,
        "own_formula_residual": own_formula_residual,
        "endpoint_cosine_formula_residual": cosine_formula_residual,
        "mixture_norm_squared_lower_bound_residual": lower_bound_residual,
        "own_monotonic_residual": own_monotonic_residual,
        "endpoint_cosine_upper_bound_residual": cosine_range_residual,
        "numeric_spillover_qualified": qualified,
        "projected_l2_exact_isolation_qualified": bool(
            base["projected_l2_exact_isolation_qualified"]),
        "alpha_is_fraction_of_spillover_removed": False,
        "all_unit_directions_optimized": False,
        "equal_conditional_scalar_l2_only": True,
        "equal_realized_cumulative_l2_claimed": False,
        "tolerance": float(tolerance),
    }
