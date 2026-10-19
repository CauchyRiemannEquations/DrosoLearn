"""Unit-length output directions that suppress observed harmful score effects.

The geometry constructor is label-free. A direction call receives only its
explicit caller-supplied record of observed reward signs; unknown signs are
zero and never inferred. Full labels may be passed only by a separate oracle
diagnostic. This simplified computation is not a biological learning rule.
"""
from __future__ import annotations

import numpy as np

from drosolearn.update_budget_geometry import update_budget_geometry


def selective_geometry(feature_rows, rcond=1e-12, tolerance=1e-8,
                       exact_condition_max=1e6):
    """Freeze label-free normalization, pseudoinverse and endpoint directions.

Exactly four finite, independently controllable input rows are required. A
rank/condition/numeric failure raises before any behavioral RNG is consumed.
The reused constructor uses three SVD calls; dynamic directions use none.
"""
    rows = np.asarray(feature_rows, dtype=float)
    if rows.ndim != 2 or rows.shape[0] != 4:
        raise ValueError("Exactly four input feature rows required")
    base = update_budget_geometry(rows, rcond=rcond, tolerance=tolerance,
                                  exact_condition_max=exact_condition_max)
    underlying = base["base_geometry"]
    diagnostic = underlying["unit_diagnostics"]
    d = underlying["unit_rows"].copy()
    inverse = underlying["pseudoinverse"].copy()
    shared = d @ d.T
    projector_residual = float(np.max(np.abs(d @ inverse - np.eye(4))))
    finite = all(np.isfinite(item).all() for item in (d, inverse, shared))
    qualified = bool(base["numeric_budget_qualified"]
                     and base["projected_l2_exact_isolation_qualified"]
                     and diagnostic["retained_rank"] == 4
                     and finite and projector_residual <= tolerance)
    if not qualified:
        raise ValueError("Selective directions require qualified full-rank exact input control")
    return {
        "base_budget": base,
        "unit_rows": d,
        "base_influences": shared,
        "pseudoinverse": inverse,
        "shared_directions": base["directions"][0].copy(),
        "isolated_directions": base["directions"][2].copy(),
        "full_row_rank": bool(diagnostic["full_row_rank"]),
        "retained_rank": int(diagnostic["retained_rank"]),
        "full_row_condition": float(diagnostic["full_row_condition"]),
        "projector_identity_residual": projector_residual,
        "numeric_selective_qualified": qualified,
        "rcond": float(rcond),
        "tolerance": float(tolerance),
        "exact_condition_max": float(exact_condition_max),
        "direction_uses_task_labels": False,
        "observed_sign_memory_is_separate_learning_resource": True,
    }


def selective_direction(geometry, source, observed_signs):
    """Remove known harmful off-input effects, then normalize movement to one.

For base row m_i=D D^T[i], set only entries with both signs known and
sign_i*sign_j*m_ij < -tolerance to zero. Preserve all other raw effects b_j.
The closest raw correction is u=d_i+pinv(D)@(b-m_i). The final direction u/||u||
retains nonzero-effect ratios, not their original magnitudes. Its own effect
also shrinks. Axes are source row then destination column throughout.

The caller must record only actual obtained Go reward +/-1 before calling.
NoGo reward 0 supplies no new target sign. This pure helper does not record,
infer, cache or correct observations and never reads an environment label.
"""
    if (not isinstance(source, (int, np.integer))
            or isinstance(source, (bool, np.bool_)) or not 0 <= source < 4):
        raise ValueError("Source must be an integer in [0, 3]")
    given = np.asarray(observed_signs)
    if (given.shape != (4,) or given.dtype.kind not in "iuf"
            or not np.isfinite(given).all() or not np.isin(given, (-1, 0, 1)).all()):
        raise ValueError("Observed signs must be four exact values from {-1, 0, 1}")
    signs = given.astype(np.int8, copy=True)
    if not geometry.get("numeric_selective_qualified", False):
        raise ValueError("Unqualified selective geometry")
    d = np.asarray(geometry["unit_rows"], dtype=float)
    inverse = np.asarray(geometry["pseudoinverse"], dtype=float)
    base = np.asarray(geometry["base_influences"], dtype=float)[source].copy()
    tolerance = float(geometry["tolerance"])
    known = (signs[source] != 0) & (signs != 0)
    harm = known & (signs[source] * signs * base < -tolerance)
    harm[source] = False
    desired = base.copy()
    desired[harm] = 0.
    raw = d[source].copy() + inverse @ (desired - base)
    raw_norm = float(np.linalg.norm(raw))
    if not np.isfinite(raw_norm) or raw_norm <= 0.:
        raise ValueError("Selective correction needs finite positive length")
    # Preserve the exact shared endpoint when no suppression is called for.
    scale = 1. / raw_norm if np.any(harm) else 1.
    direction = raw * scale
    raw_influence = raw @ d.T
    influence = direction @ d.T
    residuals = {
        "raw_target_residual": float(np.max(np.abs(raw_influence - desired))),
        "normalization_residual": float(np.max(np.abs(direction - raw * scale))),
        "unit_norm_residual": float(abs(np.linalg.norm(direction) - 1.)),
        "scaled_target_residual": float(np.max(np.abs(influence - desired * scale))),
        "own_raw_preservation_residual": float(abs(raw_influence[source] - base[source])),
        "raw_norm_lower_bound_residual": float(max(1. - raw_norm, 0.)),
        "harm_zero_residual": float(np.max(np.abs(influence[harm]))) if np.any(harm) else 0.,
    }
    finite = all(np.isfinite(item).all() for item in (
        direction, raw, desired, raw_influence, influence))
    qualified = bool(finite and all(value <= tolerance for value in residuals.values()))
    if not qualified:
        raise ValueError("Selective correction failed its frozen numeric checks")
    return {"source": int(source), "observed_signs": signs,
            "harm_mask": harm, "direction": direction, "raw_direction": raw,
            "desired_influence": desired, "raw_influence": raw_influence,
            "influence": influence, "raw_norm": raw_norm,
            "normalization_scale": scale, **residuals,
            "numeric_direction_qualified": qualified,
            "original_helpful_magnitudes_retained_after_normalization": False}
