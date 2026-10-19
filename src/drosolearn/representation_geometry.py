"""Label-free finite feature geometry; no graph, scientific seed or task RNG.

Projected updates minimize other-input squared raw-logit influence within the
prospectively retained SVD response subspace, subject to unit own-input effect.
They are a computational control, not a biological plasticity rule.
"""
from __future__ import annotations

import numpy as np


def raw_parallelogram_certificate():
    """Exact integer certificate for the frozen four raw-string inputs.

    Multiplication by three gives integer inputs; the length coordinate is two.
    The first three rows/columns have determinant -1, proving rank >=3, while
    the stated row relation proves rank <=3. The two independent differences
    together with the parallelogram relation give affine dimension two.
    Labels are used here only to explain the analytic obstruction.
    """
    integer_rows = np.array([[0, 1, 0, 1, 0, 0, 2], [1, 0, 1, 0, 0, 0, 2],
                             [0, 1, 1, 0, 0, 0, 2], [1, 0, 0, 1, 0, 0, 2]], dtype=np.int64)
    relation = np.array([1, 1, -1, -1], dtype=np.int64)
    residual = relation @ integer_rows
    minor = integer_rows[:3, :3]
    # The explicit 3x3 integer determinant avoids a float determinant claim.
    a, b, c = minor
    determinant = int(a[0] * (b[1] * c[2] - b[2] * c[1])
                      - a[1] * (b[0] * c[2] - b[2] * c[0])
                      + a[2] * (b[0] * c[1] - b[1] * c[0]))
    differences = integer_rows[1:3] - integer_rows[0]
    affine_minor_determinant = int(differences[0, 0] * differences[1, 2]
                                   - differences[0, 2] * differences[1, 0])
    three_correct_head = np.array([1, -1, 0, 2, 0, 0, 0], dtype=np.int64)
    return {"scaled_integer_rows": integer_rows, "scale_denominator": 3,
            "row_relation_coefficients": relation, "integer_relation_residual": residual,
            "rank_minor_rows": [0, 1, 2], "rank_minor_columns": [0, 1, 2],
            "rank_minor_determinant": determinant,
            "affine_difference_rows": [1, 2], "affine_minor_columns": [0, 2],
            "affine_minor_determinant": affine_minor_determinant,
            "exact_row_rank": 3, "exact_affine_dimension": 2,
            "distinct_inputs_preserve_identity": True,
            "positive_negative_midpoint_identical": True,
            "strict_affine_separation_impossible_for_labels": relation.copy(),
            "three_correct_example_head": three_correct_head,
            "three_correct_example_scaled_logits": integer_rows @ three_correct_head,
            "maximum_strict_correct_decisions": 3, "four_training_examples_only": True,
            "nonlinear_feature_separation_not_ruled_out": True}


def _matrix(rows):
    value = np.asarray(rows, dtype=float)
    if value.ndim != 2 or not value.shape[0] or not value.shape[1] or not np.isfinite(value).all():
        raise ValueError("Nonempty finite two-dimensional feature matrix required")
    return value


def _settings(rcond, tolerance=None):
    if not np.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError("Finite SVD relative cutoff strictly between zero and one required")
    if tolerance is not None and (not np.isfinite(tolerance) or not 0 < tolerance < 1):
        raise ValueError("Finite numeric tolerance strictly between zero and one required")


def _svd(rows, rcond):
    u, singular, vh = np.linalg.svd(rows, full_matrices=False)
    cutoff = float(rcond * singular[0])
    keep = singular > cutoff
    inverse = ((vh[keep].T / singular[keep]) @ u[:, keep].T
               if np.any(keep) else np.zeros((rows.shape[1], rows.shape[0])))
    if not np.isfinite(inverse).all():
        raise ValueError("Nonfinite retained pseudoinverse")
    return u, singular, vh, keep, inverse


def matrix_diagnostics(rows, rcond=1e-12):
    """Report ranks separately: floating roundoff rank and registered SVD rank.

    A missing full-row condition is represented by None, never infinity or a
    deceptively small finite sentinel. A rank failure is not a label-specific
    nonseparability certificate.
    """
    _settings(rcond)
    value = _matrix(rows)
    _, singular, _, keep, _ = _svd(value, rcond)
    strict_cutoff = float(max(value.shape) * np.finfo(value.dtype).eps * singular[0])
    strict_rank = int(np.count_nonzero(singular > strict_cutoff))
    retained_rank = int(np.count_nonzero(keep))
    full_row_rank = retained_rank == value.shape[0]
    return {"shape": list(value.shape), "singular_values": singular.copy(),
            "strict_rank": strict_rank, "strict_rank_cutoff": strict_cutoff,
            "retained_rank": retained_rank, "retained_rank_cutoff": float(rcond * singular[0]),
            "rcond": float(rcond), "full_row_rank": bool(full_row_rank),
            "full_row_condition": float(singular[0] / singular[-1]) if full_row_rank else None,
            "retained_condition": float(singular[0] / singular[keep][-1]) if retained_rank else None}


def geometry(feature_rows, rcond=1e-12, tolerance=1e-8, exact_condition_max=1e6):
    """Construct shared and own-score-matched minimum-spillover directions.

    For unit feature rows D and retained pseudoinverse C, P=D C. Its ith
    diagonal is positive for any retained input component. Among response
    vectors v in col(D_retained) with v_i=1, Cauchy-Schwarz gives
    ||v||^2 >= 1/P_ii, attained by P[:,i]/P_ii. Thus directions[i]=C[:,i]/P_ii
    minimize the sum of squared other-input influences. Discarded singular
    directions are explicitly excluded from this claim.

    Influence matrices use ROW=updated input and COLUMN=probed input. They
    omit the common eta*(action-p)*obtained_reward multiplier.
    """
    _settings(rcond, tolerance)
    if not np.isfinite(exact_condition_max) or exact_condition_max < 1:
        raise ValueError("Finite exact-isolation condition bound >=1 required")
    raw = _matrix(feature_rows)
    norms = np.linalg.norm(raw, axis=1)
    if not np.isfinite(norms).all() or not np.all(norms > 0):
        raise ValueError("Every feature difference must have finite positive L2 norm")
    d = raw / norms[:, None]
    left, _, _, keep, inverse = _svd(d, rcond)
    projector = d @ inverse
    diagonal = np.diag(projector).copy()
    if not np.isfinite(projector).all() or not np.all(diagonal > 0):
        raise ValueError("Every input needs finite positive retained projection diagonal")
    directions = inverse.T / diagonal[:, None]
    shared = d @ d.T
    projected = directions @ d.T
    if not np.isfinite(directions).all() or not np.isfinite(projected).all():
        raise ValueError("Nonfinite own-score-matched projected direction")
    own_residual = float(np.max(np.abs(np.diag(projected) - np.diag(shared))))
    off_diagonal = projected.copy()
    np.fill_diagonal(off_diagonal, 0.)
    ideal_projector = left[:, keep] @ left[:, keep].T
    diagnostic = matrix_diagnostics(d, rcond)
    full_condition = diagnostic["full_row_condition"]
    update_qualified = own_residual <= tolerance
    off_residual = float(np.max(np.abs(off_diagonal)))
    exact_qualified = bool(update_qualified and diagnostic["full_row_rank"]
                           and full_condition <= exact_condition_max and off_residual <= tolerance)
    return {"raw_rows": raw.copy(), "row_norms": norms, "unit_rows": d,
            "raw_diagnostics": matrix_diagnostics(raw, rcond), "unit_diagnostics": diagnostic,
            "pseudoinverse": inverse, "projector": projector, "projector_diagonal": diagonal,
            "directions": directions, "influence_shared": shared, "influence_projected": projected,
            "own_step_residual": own_residual, "off_diagonal_max": off_residual,
            "projector_symmetry_residual": float(np.max(np.abs(projector - projector.T))),
            "projector_idempotence_residual": float(np.max(np.abs(projector @ projector - projector))),
            "projector_svd_residual": float(np.max(np.abs(projector - ideal_projector))),
            "direction_norms": np.linalg.norm(directions, axis=1),
            "projected_spillover_squared": np.sum(off_diagonal ** 2, axis=1),
            "minimum_spillover_squared_analytic": 1. / diagonal - 1.,
            "numeric_update_qualified": bool(update_qualified),
            "exact_isolation_qualified": exact_qualified,
            "retained_subspace_only": True, "tolerance": float(tolerance),
            "exact_condition_max": float(exact_condition_max)}


def hard_margin_witness(rows, labels, rcond=1e-12, tolerance=1e-8):
    """Enumerate all 15 nonempty active sets for four homogeneous constraints.

    At a minimum-norm feasible solution, the head lies in the span of active
    signed rows and is their minimum-norm equality solution. Enumerating every
    subset therefore contains that solution in exact arithmetic. Floating
    failure is reported only as no feasible candidate under the fixed cutoff
    and tolerance; it is not an exact mathematical impossibility statement.
    This receives labels solely as an analytic capacity probe, never to train
    or normalize a reward-only actor.
    """
    _settings(rcond, tolerance)
    d = _matrix(rows)
    y = np.asarray(labels, dtype=float)
    if d.shape[0] != 4 or y.shape != (4,) or not np.isin(y, (-1., 1.)).all():
        raise ValueError("Exactly four rows and four +/-1 labels required")
    signed = y[:, None] * d
    heads, residuals, margins, feasible = [], [], [], []
    for mask in range(1, 16):
        indices = np.asarray([index for index in range(4) if mask & (1 << index)])
        active = signed[indices]
        *_, inverse = _svd(active, rcond)
        head = inverse @ np.ones(len(indices))
        signed_margins = signed @ head
        residual = float(np.max(np.abs(active @ head - 1.)))
        heads.append(head)
        residuals.append(residual)
        margins.append(float(np.min(signed_margins)))
        feasible.append(bool(residual <= tolerance and np.all(signed_margins >= 1. - tolerance)))
    candidate_heads = np.asarray(heads)
    flags = np.asarray(feasible, dtype=bool)
    norms = np.linalg.norm(candidate_heads, axis=1)
    valid = np.flatnonzero(flags)
    best = int(valid[np.argmin(norms[valid])]) if len(valid) else None
    head = candidate_heads[best].copy() if best is not None else np.zeros(d.shape[1])
    return {"status": "feasible_under_tolerance" if best is not None else "no_feasible_candidate_under_tolerance",
            "feasible": bool(best is not None), "head": head,
            "signed_margins": signed @ head, "head_l2": float(norms[best]) if best is not None else None,
            "active_subset_mask": best + 1 if best is not None else None,
            "examined_subset_count": 15, "feasible_candidate_count": int(flags.sum()),
            "candidate_heads": candidate_heads, "candidate_l2": norms,
            "candidate_equality_residuals": np.asarray(residuals),
            "candidate_min_margins": np.asarray(margins), "candidate_feasible": flags,
            "rcond": float(rcond), "tolerance": float(tolerance),
            "failed_search_is_impossibility_proof": False, "analytic_probe_not_reward_learning": True}
