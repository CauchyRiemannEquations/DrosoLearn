"""Label-free, certified directions inside a polyhedral cone and a unit ball.

This is a small computational optimization helper, not a biological plasticity
rule. Its caller supplies an objective and at most four homogeneous linear
constraints. No labels, observations, model weights or task state are read.
"""
from __future__ import annotations

import itertools

import numpy as np


def _norm(value):
    """Euclidean norm with scaling to avoid avoidable overflow/underflow."""
    value = np.asarray(value, dtype=float)
    scale = float(np.max(np.abs(value))) if value.size else 0.
    if scale == 0.:
        return 0.
    result = scale * float(np.linalg.norm(value / scale))
    if not np.isfinite(result):
        raise ValueError("The Euclidean norm must remain finite")
    return result


def cone_direction(constraints, objective, rcond=1e-12, tolerance=1e-8):
    """Maximize ``objective @ w`` with ``constraints @ w >= 0, ||w|| <= 1``.

All active subsets are enumerated, including the empty subset. For a subset S,
the projection p of c onto its equality space and its candidate multipliers
are computed by an SVD of A_S. This computes
``lambda_S = -pinv(A_S A_S.T) @ A_S @ c`` while retaining rank according to
the singular values of A_S itself. Applying the cutoff to its squared Gram
matrix would prematurely discard controllable constraints.

A candidate is accepted only if its primal feasibility, nonnegative dual
multipliers, stationarity and complementarity pass the supplied tolerance.
Only negative multipliers within the tolerance are rounded up to zero; the
resulting stationarity residual is independently checked and reported. No
constraint or failed certificate is silently relaxed.

The minimum-distance qualified candidate is the cone projection p. If p is
nonzero, its normalized direction attains the support value ||p||. The dual
certificate gives, for every feasible unit-ball direction w,
``c @ w = (c + A.T @ lambda) @ w - lambda @ (A @ w)
<= ||c + A.T @ lambda||``. The returned ``dual_upper_bound`` uses this
explicit norm and nonnegative multipliers, so its algebraic upper bound does
not depend on approximate projection stationarity. The separate
``objective_upper_bound = ||p||`` agrees within the checked residuals.
A numerical zero projection
returns a zero direction, without attempting normalization. This zero cleanup
is restricted to 128 machine epsilons times ||c|| and is reported explicitly.

The inputs are never modified. Invalid inputs or a failed numerical
certificate raise ValueError; no optimizer fallback changes the constraints.
"""
    given_a = np.asarray(constraints)
    given_c = np.asarray(objective)
    if (given_a.ndim != 2 or given_c.ndim != 1
            or given_c.size < 1 or given_a.shape[1] != given_c.size
            or given_a.shape[0] > 4
            or given_a.dtype.kind not in "iuf"
            or given_c.dtype.kind not in "iuf"
            or not np.isfinite(given_a).all()
            or not np.isfinite(given_c).all()):
        raise ValueError("Finite real A[m,n] and c[n], n >= 1 and 0 <= m <= 4 required")
    given_rcond = np.asarray(rcond)
    given_tolerance = np.asarray(tolerance)
    if (given_rcond.ndim != 0 or given_rcond.dtype.kind not in "iuf"
            or not np.isfinite(given_rcond) or not 0. < float(given_rcond) < 1.
            or given_tolerance.ndim != 0 or given_tolerance.dtype.kind not in "iuf"
            or not np.isfinite(given_tolerance) or float(given_tolerance) <= 0.):
        raise ValueError("Finite 0 < rcond < 1 and positive tolerance required")
    a = given_a.astype(float, copy=True)
    c = given_c.astype(float, copy=True)
    rcond = float(rcond)
    tolerance = float(tolerance)
    m = a.shape[0]
    c_norm = _norm(c)
    zero_threshold = 128. * np.finfo(float).eps * c_norm
    candidates = []
    subsets_evaluated = 0

    for size in range(m + 1):
        for active in itertools.combinations(range(m), size):
            subsets_evaluated += 1
            multipliers = np.zeros(m, dtype=float)
            p = c.copy()
            active_rank = 0
            cleanup = 0.
            if active:
                selected = a[list(active)]
                try:
                    left, singular, right = np.linalg.svd(selected, full_matrices=False)
                except np.linalg.LinAlgError as error:
                    raise ValueError("Active-set SVD did not converge") from error
                if singular.size and singular[0] > 0.:
                    keep = singular > rcond * singular[0]
                    active_rank = int(np.sum(keep))
                    components = right[keep] @ c
                    selected_multipliers = -left[:, keep] @ (components / singular[keep])
                    p = c - right[keep].T @ components
                    if not np.isfinite(selected_multipliers).all() or not np.isfinite(p).all():
                        continue
                    if np.any(selected_multipliers < -tolerance):
                        continue
                    cleanup = float(np.max(np.maximum(-selected_multipliers, 0.)))
                    multipliers[list(active)] = np.maximum(selected_multipliers, 0.)
            if not np.isfinite(p).all():
                continue
            raw_projection_norm = _norm(p)
            zero_projection = bool(raw_projection_norm <= zero_threshold)
            if zero_projection:
                p = np.zeros_like(c)
            p_norm = _norm(p)
            direction = p / p_norm if p_norm > 0. else np.zeros_like(c)
            projection_values = a @ p
            constraint_values = a @ direction
            dual_vector = c + a.T @ multipliers
            stationarity = p - dual_vector
            if not all(np.isfinite(value).all() for value in (
                    projection_values, constraint_values, dual_vector, stationarity)):
                continue
            upper = p_norm
            dual_upper = _norm(dual_vector)
            achieved = float(c @ direction)
            direction_norm = _norm(direction)
            residuals = {
                "primal_violation": float(np.max(np.maximum(-constraint_values, 0.))) if m else 0.,
                "projection_primal_violation": float(np.max(np.maximum(-projection_values, 0.))) if m else 0.,
                "dual_violation": float(np.max(np.maximum(-multipliers, 0.))) if m else 0.,
                "stationarity": float(np.max(np.abs(stationarity))),
                "complementarity": float(np.max(np.abs(multipliers * projection_values))) if m else 0.,
                "active_equality": float(np.max(np.abs(projection_values[list(active)]))) if active else 0.,
                "norm_violation": float(max(direction_norm - 1., 0.)),
                "support_gap": float(abs(upper - achieved)),
                "upper_bound_violation": float(max(achieved - upper, 0.)),
                "dual_bound_gap": float(abs(dual_upper - achieved)),
                "roundoff_dual_cleanup_max": cleanup,
                "zero_projection_cleanup_norm": raw_projection_norm if zero_projection else 0.,
            }
            if not np.isfinite(achieved) or not all(
                    np.isfinite(value) and value <= tolerance for value in residuals.values()):
                continue
            active_mask = np.zeros(m, dtype=bool)
            active_mask[list(active)] = True
            candidates.append({
                "direction": direction,
                "projected_objective": p,
                "multipliers": multipliers,
                "active_mask": active_mask,
                "constraint_values": constraint_values,
                "projection_constraint_values": projection_values,
                "objective_upper_bound": upper,
                "dual_upper_bound": dual_upper,
                "dual_vector": dual_vector,
                "dual_optimality_gap": dual_upper - achieved,
                "achieved_objective": achieved,
                "optimality_gap": upper - achieved,
                "direction_norm": direction_norm,
                "projection_norm": p_norm,
                "objective_norm": c_norm,
                "selected_active_set": list(active),
                "selected_active_rank": active_rank,
                "projection_distance": _norm(p - c),
                "zero_projection": zero_projection,
                "numerical_zero_threshold": zero_threshold,
                "residuals": residuals,
            })
    if not candidates:
        raise ValueError("No active set passed the frozen cone certificate tolerance")
    result = min(candidates, key=lambda item: item["projection_distance"])
    result.update({
        "candidate_count": len(candidates),
        "active_subsets_evaluated": subsets_evaluated,
        "numeric_qualified": True,
        "rcond": rcond,
        "tolerance": tolerance,
        "rank_cutoff_applies_to_active_rows": True,
        "direction_uses_task_labels": False,
    })
    return result
