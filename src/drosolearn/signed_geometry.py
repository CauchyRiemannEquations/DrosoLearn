"""Signed, retrospective descriptions of fixed saved output updates.

This module performs no training, random draws, graph construction or label
dependent choice of an update direction. Labels enter only the analysis of
which way a *saved* change moves a destination's correct-action margin.
"""
from __future__ import annotations

import numpy as np


COMPONENTS = ("self", "cross_same", "cross_opposite", "cross_all", "all")


def _validate(influences, labels, tolerance):
    matrix = np.asarray(influences, dtype=float)
    targets = np.asarray(labels, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] == 0
            or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all()):
        raise ValueError("Influences must be a finite nonempty source-by-destination square matrix")
    if (targets.shape != (matrix.shape[0],) or not np.isfinite(targets).all()
            or not np.isin(targets, (-1., 1.)).all()):
        raise ValueError("Signed labels must contain exactly one -1 or +1 per input")
    if (isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance)
            or not np.isfinite(tolerance) or tolerance <= 0.):
        raise ValueError("Tolerance must be a finite positive scalar")
    return matrix, targets, float(tolerance)


def _logits(logits, count):
    logits = np.asarray(logits, dtype=float)
    if logits.shape != (count,) or not np.isfinite(logits).all():
        raise ValueError("Logits must contain one finite raw Go score per input")
    return logits


def clipped_sigmoid(value):
    """The parent's logistic(clip(raw score, -40, 40)), without model state."""
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError("Scores must be finite")
    return 1. / (1. + np.exp(-np.clip(value, -40., 40.)))


def component_masks(labels):
    """Disjoint self/same-target/opposite-target masks, then their unions.

    Rows denote the source input whose saved update occurred; columns denote
    the destination input being probed. Equal labels do not guarantee help:
    a negative cross influence can reverse its direction.
    """
    targets = np.asarray(labels, dtype=float)
    if (targets.ndim != 1 or targets.size == 0
            or not np.isfinite(targets).all()
            or not np.isin(targets, (-1., 1.)).all()):
        raise ValueError("Signed labels must be a finite nonempty vector of -1 or +1")
    own = np.eye(len(targets), dtype=bool)
    equal = targets[:, None] == targets[None, :]
    return {"self": own, "cross_same": ~own & equal,
            "cross_opposite": ~equal, "cross_all": ~own,
            "all": np.ones(own.shape, dtype=bool)}


def component_totals(values, masks, tolerance=1e-8):
    """Exact signed masses and separate tolerance-based event counts.

    No small term is discarded from the masses. The neutral event class is
    ``abs(value) <= tolerance``; its signed and absolute masses remain visible.
    The helper works on a full source/destination matrix or one source row.
    """
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Effect values must be finite")
    if (isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance)
            or not np.isfinite(tolerance) or tolerance <= 0.):
        raise ValueError("Tolerance must be a finite positive scalar")
    if tuple(masks) != COMPONENTS:
        raise ValueError("All five ordered component masks are required")
    result = {}
    for key in COMPONENTS:
        mask = np.asarray(masks[key])
        if mask.shape != values.shape or mask.dtype != bool:
            raise ValueError("Component masks must be boolean and match the effect shape")
        selected = values[mask]
        helpful, harmful = np.maximum(selected, 0.), np.maximum(-selected, 0.)
        neutral = np.abs(selected) <= tolerance
        result[key] = {
            "helpful": float(helpful.sum()), "harmful": float(harmful.sum()),
            "net": float(selected.sum()), "absolute": float(np.abs(selected).sum()),
            "entries": int(selected.size),
            "helpful_count": int(np.count_nonzero(selected > tolerance)),
            "harmful_count": int(np.count_nonzero(selected < -tolerance)),
            "neutral_count": int(np.count_nonzero(neutral)),
            "exact_positive_count": int(np.count_nonzero(selected > 0.)),
            "exact_negative_count": int(np.count_nonzero(selected < 0.)),
            "exact_zero_count": int(np.count_nonzero(selected == 0.)),
            "neutral_net": float(selected[neutral].sum()),
            "neutral_absolute": float(np.abs(selected[neutral]).sum()),
            "mass_identity_residual": float(abs(helpful.sum() - harmful.sum() - selected.sum())),
        }
    return result


def signed_static(influences, labels, tolerance=1e-8):
    """Describe unit source-correct-direction influence B_ij=y_i*y_j*M_ij.

    These are raw score effects before any reward scalar. They are neither
    probability changes nor the expected effect of a future training run.
    """
    matrix, targets, tolerance = _validate(influences, labels, tolerance)
    signed = targets[:, None] * targets[None, :] * matrix
    masks = component_masks(targets)
    return {
        "target_aligned": signed,
        "helpful": np.maximum(signed, 0.), "harmful": np.maximum(-signed, 0.),
        "classification": np.where(signed > tolerance, 1,
                                   np.where(signed < -tolerance, -1, 0)).astype(np.int8),
        "masks": masks, "components": component_totals(signed, masks, tolerance),
        "source_axis": 0, "destination_axis": 1, "tolerance": tolerance,
        "labels_used_only_in_analysis": True,
    }


def signed_step(influences, labels, source_index, scalar, pre_logits, tolerance=1e-8):
    """Describe one *saved* update, without changing a head or drawing actions.

    C_j=y_j*g*M_ij is its additive correct-margin contribution. Probability
    effects compare the same saved pre-step state with and without this one
    full saved update. Their sigmoid is clipped exactly as in the parent.
    Successive effects telescope along the saved path. Splitting these effects
    by the source event does not predict a new trajectory with events removed
    and is not an order-free causal attribution of the final probability.
    """
    matrix, targets, tolerance = _validate(influences, labels, tolerance)
    if (isinstance(source_index, (bool, np.bool_))
            or not isinstance(source_index, (int, np.integer))
            or not 0 <= source_index < len(targets)):
        raise ValueError("Source index must be an input index")
    if (isinstance(scalar, (bool, np.bool_)) or not np.isscalar(scalar)
            or not np.isfinite(scalar)):
        raise ValueError("Saved update scalar must be finite")
    logits = _logits(pre_logits, len(targets))
    margins = targets * logits
    change = targets * float(scalar) * matrix[source_index]
    post_margins = margins + change
    if not np.isfinite(post_margins).all():
        raise ValueError("Post-update margins must remain finite")
    probability = clipped_sigmoid(post_margins) - clipped_sigmoid(margins)
    masks = {key: mask[source_index] for key, mask in component_masks(targets).items()}
    return {
        "pre_margins": margins, "post_margins": post_margins,
        "margin_change": change, "probability_change": probability,
        "margin_components": component_totals(change, masks, tolerance),
        "probability_components": component_totals(probability, masks, tolerance),
        "source_index": int(source_index), "scalar": float(scalar),
        "tolerance": tolerance, "probability_clip": 40.,
    }


def checkpoint_drift(influences, labels, logits, eta, tolerance=1e-8):
    """Exact conditional one-step expectations at a fixed saved checkpoint.

    Sources are uniformly mixed and a~Bernoulli(p_i), Go obtains y_i and NoGo
    obtains zero. Each returned matrix is *already weighted* by source
    probability 1/n. The expected probability effect enumerates the Go/NoGo
    possibilities; it is not sigmoid of the expected margin. Nothing here is
    a rollout or a prediction of the complete later learning trajectory.
    """
    matrix, targets, tolerance = _validate(influences, labels, tolerance)
    logits = _logits(logits, len(targets))
    if (isinstance(eta, (bool, np.bool_)) or not np.isscalar(eta)
            or not np.isfinite(eta) or eta < 0.):
        raise ValueError("Eta must be finite and nonnegative")
    p = clipped_sigmoid(logits)
    n = len(targets)
    expected_scalars = float(eta) * p * (1. - p) * targets
    go_scalars = float(eta) * (1. - p) * targets
    go_margins = go_scalars[:, None] * targets[None, :] * matrix
    contributions = (expected_scalars[:, None] * targets[None, :] * matrix) / n
    margins = targets * logits
    go_post_margins = margins[None, :] + go_margins
    if not np.isfinite(go_post_margins).all():
        raise ValueError("Conditional Go margins must remain finite")
    probability_contributions = (p[:, None] / n) * (
        clipped_sigmoid(go_post_margins) - clipped_sigmoid(margins)[None, :])
    masks = component_masks(targets)
    return {
        "go_probabilities": p, "expected_scalars_given_source": expected_scalars,
        "go_scalars_given_source": go_scalars, "conditional_go_margin_changes": go_margins,
        "margin_contributions": contributions,
        "probability_contributions": probability_contributions,
        "expected_margin_drift": contributions.sum(axis=0),
        "expected_probability_drift": probability_contributions.sum(axis=0),
        "margin_components": component_totals(contributions, masks, tolerance),
        "probability_components": component_totals(probability_contributions, masks, tolerance),
        "source_probability": 1. / n, "tolerance": tolerance,
        "probability_clip": 40., "complete_future_rollout": False,
    }
