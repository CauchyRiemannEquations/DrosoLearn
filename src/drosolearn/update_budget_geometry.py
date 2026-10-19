"""Label-free output-update directions under separate resource conventions.

The own-score-matched projected direction is a frozen computational control.
Normalizing it to unit L2 length limits its output-weight movement, while
reducing its own-input score effect. Equal conditional movement is distinct
from equal realized cumulative movement on autonomous reward trajectories.
No graph construction, task labels, scientific seeds or reward RNG occur here.
"""
from __future__ import annotations

import numpy as np

from drosolearn.representation_geometry import geometry


UPDATES = ("shared", "projected_own", "projected_l2")


def update_budget_geometry(feature_rows, rcond=1e-12, tolerance=1e-8,
                           exact_condition_max=1e6):
    """Return shared, own-score-matched and L2-matched directions.

    ``directions`` has axes (update, source input, head coordinate).
    ``influences`` has axes (update, source input, probed input). All static
    influences omit the scalar ``eta * (action - probability) * reward``.

    Shared and projected_l2 directions each have unit L2 length, so a COMMON
    scalar gives equal head-step lengths. A reward-only actor's probabilities
    and obtained rewards need not agree across conditions. Neither the helper
    nor the protocol thereby equalizes actual cumulative costs.

    Unit-normalizing the already specified projected direction preserves its
    zero off-input effects when exact isolation is qualified. It does not
    preserve the unit own-input effect, and is not claimed to solve a new
    minimum-spillover optimization under a free-own-response constraint.
    """
    base = geometry(feature_rows, rcond=rcond, tolerance=tolerance,
                    exact_condition_max=exact_condition_max)
    shared = base["unit_rows"]
    own = base["directions"]
    own_norms = np.linalg.norm(own, axis=1)
    if not np.isfinite(own_norms).all() or not np.all(own_norms > 0):
        raise ValueError("Every own-score-matched direction needs finite positive L2 norm")
    l2 = own / own_norms[:, None]
    directions = np.stack((shared, own, l2))
    influences = directions @ shared.T
    if not np.isfinite(directions).all() or not np.isfinite(influences).all():
        raise ValueError("Nonfinite output-budget direction or input influence")
    norms = np.linalg.norm(directions, axis=2)
    diagonal = np.diagonal(influences, axis1=1, axis2=2).copy()
    off_diagonal = influences.copy()
    indices = np.arange(shared.shape[0])
    off_diagonal[:, indices, indices] = 0.
    unit_residual = float(np.max(np.abs(norms[[0, 2]] - 1.)))
    conditional_match_residual = float(np.max(np.abs(norms[0] - norms[2])))
    reconstruction_residual = float(np.max(np.abs(l2 * own_norms[:, None] - own)))
    own_scaled_residual = float(np.max(np.abs(diagonal[2] * own_norms - diagonal[1])))
    # Cauchy-Schwarz: D_i has unit norm and D_i dot U_i = 1, so ||U_i|| >= 1.
    norm_lower_residual = float(np.max(np.maximum(1. - own_norms, 0.)))
    own_cap_residual = float(np.max(np.maximum(diagonal[2] - diagonal[0], 0.)))
    budget_qualified = bool(base["numeric_update_qualified"]
                            and unit_residual <= tolerance
                            and conditional_match_residual <= tolerance
                            and reconstruction_residual <= tolerance
                            and own_scaled_residual <= tolerance
                            and norm_lower_residual <= tolerance
                            and own_cap_residual <= tolerance)
    return {
        "base_geometry": base, "updates": UPDATES,
        "directions": directions, "influences": influences,
        "direction_norms": norms, "own_influence": diagonal,
        "spillover_squared": np.sum(off_diagonal ** 2, axis=2),
        "own_response_per_unit_l2": diagonal / norms,
        "unit_norm_residual": unit_residual,
        "conditional_l2_match_residual": conditional_match_residual,
        "l2_normalization_reconstruction_residual": reconstruction_residual,
        "own_scaled_match_residual": own_scaled_residual,
        "projected_own_norm_lower_bound_residual": norm_lower_residual,
        "projected_l2_own_cap_residual": own_cap_residual,
        "projected_l2_exact_isolation_qualified": bool(base["exact_isolation_qualified"]),
        "numeric_budget_qualified": budget_qualified,
        "equal_conditional_scalar_l2_only": True,
        "equal_realized_cumulative_l2_claimed": False,
        "tolerance": float(tolerance),
    }
