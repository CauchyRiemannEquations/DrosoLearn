"""Independent active-face check for a saved-path unit-cap geometry analysis.

The frontier producer and its solver are never imported. The optimization is
recomputed in the original output-feature space using small least-squares
systems. Recorded update scalars are references, never a new learning policy.
"""
from __future__ import annotations

from itertools import combinations
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


TOPOLOGIES = ("real", "rewired", "ann")
REPRESENTATIONS = ("raw", "identity")
CANDIDATES = ("parent_feedback", "shared_reference", "isolated_reference", "observed_own",
              "observed_balanced", "oracle_own", "oracle_balanced")
PHASES = ("all", "partial", "full")
SIGNS = np.asarray([1., 1., -1., -1.])
METRICS = ("trial_count", "updated_trials", "own_margin", "helpful_margin", "harmful_margin",
           "unknown_harmful_margin", "total_margin", "cumulative_step_l2", "max_step_l2",
           "own_retention", "help_retention", "joint_nondegrading_updates",
           "joint_strict_improvement_updates", "known_safety_violations")
FIELDS = ("case_keys", "case_directions", "case_projections", "case_multipliers", "case_objectives",
          "case_constraints", "case_certificates", "case_indices", "candidate_influences",
          "candidate_margin_delta", "candidate_step_lengths", "known_after", "sources",
          "update_factors", "full_known_after")
INTERPRETATION = {"prior_outcomes_known": True, "confirmatory_behavioral_hypothesis": False,
    "same_saved_scalar_not_autonomous_policy": True, "no_unknown_signs_in_observed_constraints": True,
    "oracle_is_full_information_safety_reference": True, "oracle_not_universal_observed_upper_bound": True,
    "own_optimum_not_joint_improvement_guarantee": True, "balanced_not_complete_Pareto_frontier": True,
    "score_not_accuracy": True, "unknown_harm_evaluator_only": True,
    "unit_cap_not_equal_processing_cost": True, "generalization_tested": False,
    "new_human_experiment": False, "shared_neural_mechanism_claim": False, "education_effect_claim": False}


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, tolerance=1e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    demand(actual.shape == expected.shape and actual.dtype.kind in "biuf"
           and np.isfinite(actual).all(), label + " finite axes")
    if expected.dtype.kind in "biu":
        demand(actual.dtype.kind in "biu" and np.array_equal(actual, expected), label + " exact")
    else:
        demand(np.allclose(actual, expected, atol=tolerance, rtol=1e-9), label + " arithmetic")


def compare_tree(actual, expected, label="summary"):
    if isinstance(expected, dict):
        demand(isinstance(actual, dict) and actual.keys() == expected.keys(), label + " keys")
        for key, value in expected.items():
            compare_tree(actual[key], value, label + "." + key)
    elif isinstance(expected, list):
        demand(isinstance(actual, list) and len(actual) == len(expected), label + " length")
        for i, value in enumerate(expected):
            compare_tree(actual[i], value, label + "." + str(i))
    elif type(expected) is bool:
        demand(type(actual) is bool and actual == expected, label + " flag")
    elif type(expected) is int:
        demand(type(actual) is int and actual == expected, label + " integer")
    elif type(expected) is float:
        demand(type(actual) in (float, int) and np.isfinite(actual)
               and np.isclose(actual, expected, atol=1e-9, rtol=1e-9), label + " numeric")
    else:
        demand(actual == expected, label + " value")


def validate_science(cfg, allow_fixture=False):
    choices = {"study_id": "cone_frontier_v1", "scope": "retrospective_descriptive_cone_frontier",
        "module": "drosolearn.cone_frontier", "protocol": "docs/cone-frontier-protocol.md",
        "source_result": "results/selective_spillover_v1", "topologies": list(TOPOLOGIES),
        "representations": list(REPRESENTATIONS), "model_seeds": list(range(17401, 17411)),
        "training_trials": 400, "head_lr": .1, "numeric_tolerance": 1e-8,
        "neutral_tolerance": 1e-8, "svd_rcond": 1e-12, "labels": [1, 1, -1, -1],
        "parent_method": "feedback_selective", "phases": list(PHASES),
        "candidates": list(CANDIDATES), "external_manifest": None}
    demand(isinstance(cfg, dict), "Independent fixed configuration object")
    fixture = cfg.get("scope") == "integration_test"
    demand(not fixture or allow_fixture, "Invented fixtures cannot enter actual analysis")
    exemptions = {"scope", "model_seeds", "training_trials"} if fixture else set()
    for key, value in choices.items():
        if key not in exemptions:
            demand(cfg.get(key) == value, "Independent fixed scientific choice: " + key)
    if fixture:
        seeds, n = cfg.get("model_seeds"), cfg.get("training_trials")
        demand(type(seeds) is list and 1 <= len(seeds) <= 2 and len(set(seeds)) == len(seeds)
               and all(type(x) is int and 90001 <= x <= 90100 for x in seeds), "Invented fixture seeds")
        demand(type(n) is int and 4 <= n <= 16 and n % 4 == 0, "Bounded invented saved records")


def independent_problem(d, source, signs, objective, tolerance):
    """Only the supplied observation signs enter the proposed direction problem."""
    d, k = np.asarray(d, dtype=float), np.asarray(signs, dtype=int)
    demand(d.ndim == 2 and d.shape[0] == 4 and k.shape == (4,)
           and np.isin(k, [-1, 0, 1]).all() and source in range(4) and k[source] != 0
           and objective in (0, 1), "Independent actual Go optimization problem")
    known = np.flatnonzero(k)
    oriented = k[source] * k[:, None] * d
    gram_row = d @ d[source]
    useful = np.flatnonzero((np.arange(4) != source) & (k != 0)
                           & (k[source] * k * gram_row > tolerance))
    c = d[source].copy()
    if objective == 1 and useful.size:
        c = .5 * c + .5 * oriented[useful].mean(axis=0)
    return oriented[known], c, known, useful


def independent_unit_cone(objective, constraints, tolerance=1e-9):
    """Maximize c.w under ||w|| <= 1 and A.w >= 0 by face enumeration.

    Every possible active subset is considered. On its null space the optimum
    of the remaining linear objective is its normalized orthogonal projection.
    Feasible candidates plus zero include a global solution. This routine uses
    ambient feature vectors directly, independently of a reduced-space solver.
    """
    c = np.asarray(objective, dtype=float)
    a = np.asarray(constraints, dtype=float)
    demand(c.ndim == 1 and a.ndim == 2 and a.shape[1] == c.size,
           "Independent cone objective and constraints axes")
    demand(np.isfinite(c).all() and np.isfinite(a).all() and np.isfinite(tolerance)
           and tolerance > 0, "Independent cone finite inputs")
    demand(a.shape[0] <= 4, "At most four destination inequalities")
    best = np.zeros_like(c)
    value = 0.
    for count in range(a.shape[0] + 1):
        for subset in combinations(range(a.shape[0]), count):
            face = a[list(subset)]
            if count:
                # Solve the ambient rectangular system directly. Forming a
                # Gram matrix would square its condition number and can alter
                # the retained numerical rank at the registered cutoff.
                coefficients = np.linalg.lstsq(face.T, c, rcond=1e-12)[0]
                projection = c - face.T @ coefficients
            else:
                projection = c.copy()
            length = float(np.linalg.norm(projection))
            if length <= 1e-12:
                continue
            candidate = projection / length
            if np.any(a @ candidate < -tolerance):
                continue
            candidate_value = float(c @ candidate)
            if candidate_value > value:
                best, value = candidate, candidate_value
    return best, value


def independent_support_bound(objective, constraints, tolerance=1e-8):
    """Independently minimize certified dual bounds over every active face.

    A slightly infeasible primal direction can overstate a support value when
    nearly parallel constraints amplify roundoff. Every candidate here instead
    has exactly nonnegative multipliers, including the all-zero candidate. Its
    norm(c+A.T.mu) is a valid upper bound regardless of primal roundoff. The
    true optimum's active subset is included in this finite enumeration.
    """
    c, a = np.asarray(objective, dtype=float), np.asarray(constraints, dtype=float)
    demand(c.ndim == 1 and a.ndim == 2 and a.shape[1] == c.size and a.shape[0] <= 4
           and np.isfinite(c).all() and np.isfinite(a).all() and np.isfinite(tolerance)
           and tolerance > 0., "Independent dual enumeration finite axes")
    best_mu = np.zeros(a.shape[0])
    best_upper = float(np.linalg.norm(c))
    for count in range(1, a.shape[0] + 1):
        for subset in combinations(range(a.shape[0]), count):
            face = a[list(subset)]
            selected = -np.linalg.lstsq(face.T, c, rcond=1e-12)[0]
            if np.any(selected < -tolerance):
                continue
            mu = np.zeros(a.shape[0])
            mu[list(subset)] = np.maximum(selected, 0.)
            upper = float(np.linalg.norm(c + a.T @ mu))
            if upper < best_upper:
                best_upper, best_mu = upper, mu
    demand(np.isfinite(best_upper) and np.isfinite(best_mu).all() and np.all(best_mu >= 0.),
           "Independent nonnegative dual support bound")
    return best_upper, best_mu


def independent_certificate(objective, constraints, direction, tolerance=1e-8):
    """A nonnegative dual witness makes a feasible unit-cap point globally optimal."""
    c, a, w = (np.asarray(x, dtype=float) for x in (objective, constraints, direction))
    demand(c.ndim == w.ndim == 1 and c.shape == w.shape and a.ndim == 2
           and a.shape[1] == c.size and np.isfinite(c).all() and np.isfinite(a).all()
           and np.isfinite(w).all(), "Independent KKT finite axes")
    norm, values, objective_value = float(np.linalg.norm(w)), a @ w, float(c @ w)
    active = np.flatnonzero(np.abs(values) <= tolerance)
    lam = max(objective_value, 0.) if norm > 1e-12 else 0.
    rhs = lam * w - c
    multipliers = np.zeros(a.shape[0])
    if active.size:
        multipliers[active] = np.linalg.lstsq(a[active].T, rhs, rcond=1e-12)[0]
    residuals = np.asarray([
        max(norm - 1., 0.), max(-float(values.min()), 0.) if values.size else 0.,
        max(-float(multipliers.min()), 0.) if multipliers.size else 0.,
        float(np.max(np.abs(c + a.T @ multipliers - lam * w))),
        float(np.max(np.abs(multipliers * values))) if values.size else 0.,
        abs(lam * (norm - 1.)), max(-objective_value, 0.),
    ])
    demand(np.isfinite(residuals).all() and residuals.max() <= tolerance,
           "Independent global KKT certificate")
    return {"lambda": lam, "multipliers": multipliers, "residuals": residuals}


def validate_dual_witness(objective, constraints, direction, multipliers, tolerance=1e-8):
    """Check the algebraic unit-ball upper bound, independently of stationarity.

    For nonnegative mu and A.w >= 0, c.w <= (c+A.T.mu).w <=
    ||c+A.T.mu||. A matching primal value therefore certifies the support
    optimum even if a solver's projection has small stationarity roundoff.
    """
    c, a, w, mu = (np.asarray(x, dtype=float) for x in
                    (objective, constraints, direction, multipliers))
    demand(c.ndim == w.ndim == mu.ndim == 1 and c.shape == w.shape
           and a.ndim == 2 and a.shape == (mu.size, c.size)
           and all(np.isfinite(x).all() for x in (c, a, w, mu)),
           "Independent dual witness finite axes")
    demand(np.all(mu >= 0.), "Independent dual multipliers exactly nonnegative")
    vector = c + a.T @ mu
    upper, achieved = float(np.linalg.norm(vector)), float(c @ w)
    gap = upper - achieved
    residuals = np.asarray([
        max(float(np.linalg.norm(w)) - 1., 0.),
        max(-float(np.min(a @ w)), 0.) if mu.size else 0.,
        abs(gap), max(-gap, 0.),
    ])
    demand(np.isfinite(residuals).all() and residuals.max() <= tolerance,
           "Independent primal/dual support bound")
    return {"dual_vector": vector, "dual_upper_bound": upper,
            "achieved_objective": achieved, "dual_optimality_gap": gap,
            "residuals": residuals}


def saved_feedback(cfg, parent):
    """Check the fixed parent slice; its complete prior replay is hash-bound upstream."""
    s, n, tol = len(cfg["model_seeds"]), cfg["training_trials"], cfg["numeric_tolerance"]
    shape = (3, 2, 4, s, n)
    close(parent["model_seeds"], np.asarray(cfg["model_seeds"]), "Saved seed identity")
    close(parent["targets"], (SIGNS > 0).astype(int), "Saved target identity")
    sources = parent["schedules"]
    demand(sources.shape == (s, n) and sources.dtype.kind in "iu"
           and np.isin(sources, range(4)).all(), "Saved source axes")
    demand(all(np.array_equal(np.sort(block), np.arange(4))
               for schedule in sources for block in schedule.reshape(-1, 4)), "Saved balanced source blocks")
    scalars = {}
    for field in ("probabilities", "actions", "rewards", "update_factors", "head_steps"):
        demand(parent["trial_" + field].shape == shape
               and np.isfinite(parent["trial_" + field]).all(), "Saved scalar axes: " + field)
        scalars[field] = parent["trial_" + field][:, :, 3]
    p, actions, rewards, g = [scalars[key] for key in ("probabilities", "actions", "rewards", "update_factors")]
    demand(np.all((p > 0.) & (p < 1.)) and np.isin(actions, [0, 1]).all(), "Saved probabilities and actions")
    close(rewards, actions * SIGNS[sources][None, None], "Actual obtained rewards")
    close(g, cfg["head_lr"] * (actions - p) * rewards, "Actual saved scalar")
    demand(np.array_equal(g != 0., actions == 1), "Only Go nonzero update scalars")
    demand(np.all(np.abs(g) <= cfg["head_lr"] + tol), "Original per-trial scalar ceiling")
    close(scalars["head_steps"], np.abs(g), "Saved unit direction step bound")
    for field in ("known_before", "known_after", "routing_signs"):
        value = parent["trial_" + field]
        demand(value.shape == shape + (4,) and value.dtype.kind in "iu"
               and np.isin(value, [-1, 0, 1]).all(), "Saved categorical state axes")
    known = parent["trial_known_after"][:, :, 3]
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            prefix = "g_" + topology + "_" + representation + "_"
            d_all = parent[prefix + "unit_rows"]
            demand(d_all.ndim == 3 and d_all.shape[:2] == (s, 4) and np.isfinite(d_all).all(),
                   "Saved feature axes")
            close(np.linalg.norm(d_all, axis=-1), np.ones((s, 4)), "Saved feature unit norms")
            for si in range(s):
                d, order = d_all[si], sources[si]
                gram, inverse = d @ d.T, np.linalg.pinv(d, rcond=1e-12)
                close(d @ inverse, np.eye(4), "Saved full-row-rank features")
                close(parent[prefix + "base_influences"][si], gram, "Saved Gram orientation")
                close(parent[prefix + "pseudoinverse"][si], inverse, "Saved pseudoinverse")
                at = (ti, ri, si)
                seen = np.logical_or.accumulate((np.arange(4)[None, :] == order[:, None])
                                               & (actions[at][:, None] == 1), axis=0)
                observed = seen.astype(np.int8) * SIGNS.astype(np.int8)
                close(known[at], observed, "Actual obtained memory only")
                close(parent["trial_known_before"][ti, ri, 3, si],
                      np.concatenate([np.zeros((1, 4), np.int8), observed[:-1]]), "Memory before continuity")
                close(parent["trial_routing_signs"][ti, ri, 3, si], observed, "Feedback routing has no oracle")
                route_source = observed[np.arange(n), order]
                harmful = (route_source[:, None] * observed * gram[order] < -tol)
                harmful[np.arange(n), order] = False
                desired = gram[order].copy()
                desired[harmful] = 0.
                raw = d[order] + (desired - gram[order]) @ inverse.T
                scales = np.where(harmful.any(axis=1), 1. / np.linalg.norm(raw, axis=1), 1.)
                directions = raw * scales[:, None]
                saved_direction = parent["trial_directions"][ti, ri, 3, si]
                close(saved_direction, directions, "Saved feedback direction from observed signs")
                close(parent["trial_actual_influence"][ti, ri, 3, si], directions @ d.T,
                      "Saved feedback score influences")
                close(parent["trial_desired_influence"][ti, ri, 3, si], desired, "Saved raw desired influences")
    for field in ("internal_steps", "value_steps", "internal_update_calls", "value_update_calls"):
        demand(not np.any(parent["trial_" + field]), "Parent fixed internal/value state")
    return sources, g, known


def expected_artifacts(cfg, parent, actual):
    """Independently bind every case and derive effects from certified directions."""
    s, n, tol = len(cfg["model_seeds"]), cfg["training_trials"], cfg["numeric_tolerance"]
    sources, g, known = saved_feedback(cfg, parent)
    demand(set(actual) == set(FIELDS), "Independent exact fifteen-array closure")
    for key in FIELDS:
        demand(np.asarray(actual[key]).dtype.kind in "biuf" and np.isfinite(actual[key]).all(),
               "Finite new analysis array: " + key)
    full = np.all(known != 0, axis=-1)
    influences = np.zeros((3, 2, s, n, 7, 4))
    steps = np.zeros((3, 2, s, n, 7))
    indices = np.full((3, 2, s, n, 4), -1, int)
    keys, objects, constraints = [], [], []
    cache = {}
    case_directions = np.asarray(actual["case_directions"])
    demand(case_directions.ndim == 2, "Case direction axes")
    count, dimension = case_directions.shape
    demand(actual["case_keys"].shape == (count, 10)
           and actual["case_keys"].dtype.kind in "iu", "Unique optimization case key axes")
    for field in ("case_projections", "case_objectives"):
        demand(actual[field].shape == (count, dimension), "Case vector axes")
    demand(actual["case_multipliers"].shape == (count, 4)
           and actual["case_constraints"].shape == (count, 4, dimension)
           and actual["case_certificates"].shape == (count, 8), "Padded case certificate axes")
    certificate_values = []
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            d_all = parent["g_" + topology + "_" + representation + "_unit_rows"]
            demand(d_all.shape[-1] == dimension, "Common saved feature dimension")
            for si in range(s):
                d = d_all[si]
                isolated = np.stack([np.linalg.lstsq(d, np.eye(4)[:, i], rcond=1e-12)[0]
                                     for i in range(4)])
                isolated /= np.linalg.norm(isolated, axis=1, keepdims=True)
                close(parent["g_" + topology + "_" + representation + "_isolated_directions"][si],
                      isolated, "Saved isolated reference computed independently")
                for t in np.flatnonzero(g[ti, ri, si] != 0.):
                    i, at = int(sources[si, t]), (ti, ri, si, int(t))
                    vectors = [parent["trial_directions"][ti, ri, 3, si, t], d[i], isolated[i]]
                    for mode in range(4):
                        scope, objective = divmod(mode, 2)
                        k = SIGNS.astype(int) if scope else known[at]
                        key = (ti, ri, si, i, scope, objective, *map(int, k))
                        if key not in cache:
                            index = len(keys)
                            demand(index < count, "Complete optimization case cache")
                            cache[key] = index
                            keys.append(key)
                            a, c, active_destinations, _ = independent_problem(d, i, k, objective, tol)
                            padded_a = np.zeros((4, dimension))
                            padded_a[active_destinations] = a
                            objects.append(c)
                            constraints.append(padded_a)
                            close(actual["case_keys"][index], np.asarray(key), "Case deterministic observed key")
                            close(actual["case_objectives"][index], c, "Case observation-only objective")
                            close(actual["case_constraints"][index], padded_a, "Case observation-only safety constraints")
                            w, p = case_directions[index], actual["case_projections"][index]
                            mu = actual["case_multipliers"][index]
                            demand(np.all(mu >= 0.) and np.all(mu[k == 0] == 0.),
                                   "Padded dual multipliers nonnegative and unknown-free")
                            witness = validate_dual_witness(c, padded_a, w, mu, tol)
                            optimum, _ = independent_support_bound(c, a, tolerance=tol)
                            demand(abs(float(c @ w) - optimum) <= tol
                                   and abs(witness["dual_upper_bound"] - optimum) <= tol,
                                   "Independent active-face dual support optimum")
                            ap = padded_a @ p
                            certificate = np.asarray([float(c @ w), np.linalg.norm(p),
                                witness["dual_upper_bound"], np.linalg.norm(w),
                                max(-float(np.min(padded_a @ w)), 0.),
                                np.max(np.abs(p - c - padded_a.T @ mu)),
                                np.max(np.abs(mu * ap)), abs(np.linalg.norm(p) - float(c @ w))])
                            demand(np.all(certificate[4:] <= tol), "Independent stored case KKT certificate")
                            close(actual["case_certificates"][index], certificate, "Stored case certificate arithmetic")
                            certificate_values.append(certificate)
                        index = cache[key]
                        indices[at + (mode,)] = index
                        vectors.append(case_directions[index])
                    vectors = np.asarray(vectors)
                    influences[at] = vectors @ d.T
                    steps[at] = abs(g[at]) * np.linalg.norm(vectors, axis=1)
    demand(len(keys) == count, "No unused or duplicated optimization cases")
    wanted = {"case_keys": np.asarray(keys, int).reshape(count, 10),
        "case_directions": case_directions, "case_projections": actual["case_projections"],
        "case_multipliers": actual["case_multipliers"],
        "case_objectives": np.asarray(objects).reshape(count, dimension),
        "case_constraints": np.asarray(constraints).reshape(count, 4, dimension),
        "case_certificates": np.asarray(certificate_values).reshape(count, 8),
        "case_indices": indices, "candidate_influences": influences,
        "candidate_margin_delta": influences * g[..., None, None] * SIGNS,
        "candidate_step_lengths": steps, "known_after": known, "sources": sources,
        "update_factors": g, "full_known_after": full}
    for key, value in wanted.items():
        close(actual[key], value, "Independent derived array: " + key)
    demand(np.all(steps <= np.abs(g)[..., None] + tol), "All candidate per-trial unit caps")
    return wanted


def expected_rows(cfg, arrays, parent):
    rows, n = [], cfg["training_trials"]
    tol = cfg["numeric_tolerance"]
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            for ci, candidate in enumerate(CANDIDATES):
                for si, seed in enumerate(cfg["model_seeds"]):
                    at = (ti, ri, si)
                    order, known = arrays["sources"][si], arrays["known_after"][at]
                    d = parent["g_" + topology + "_" + representation + "_unit_rows"][si]
                    gram = d @ d.T
                    own = np.eye(4, dtype=bool)[order]
                    cross = ~own
                    helpful = (SIGNS[order, None] * SIGNS * gram[order] > cfg["neutral_tolerance"]) & cross
                    delta = arrays["candidate_margin_delta"][at][:, ci]
                    baseline = arrays["candidate_margin_delta"][at][:, 0]
                    influence = arrays["candidate_influences"][at][:, ci]
                    self_change = delta[np.arange(n), order]
                    old_self = baseline[np.arange(n), order]
                    helpful_change = np.where(helpful, np.maximum(delta, 0.), 0.).sum(axis=1)
                    old_helpful = np.where(helpful, np.maximum(baseline, 0.), 0.).sum(axis=1)
                    updated = arrays["update_factors"][at] != 0.
                    safe_influence = known[np.arange(n), order, None] * known * influence
                    safety_violation = updated & np.any((safe_influence < -tol) & (known != 0), axis=1)
                    joint = updated & (self_change >= old_self - tol) & (helpful_change >= old_helpful - tol)
                    strict = joint & ((self_change > old_self + tol) | (helpful_change > old_helpful + tol))
                    length = arrays["candidate_step_lengths"][at][:, ci]
                    for phase in PHASES:
                        mask = np.ones(n, bool) if phase == "all" else arrays["full_known_after"][at]
                        if phase == "partial":
                            mask = ~mask
                        own_mass, useful_mass = float(self_change[mask].sum()), float(helpful_change[mask].sum())
                        own_denominator, helpful_denominator = float(old_self[mask].sum()), float(old_helpful[mask].sum())
                        row = {"topology": topology, "representation": representation, "candidate": candidate,
                            "seed": seed, "phase": phase,
                            "trial_count": int(mask.sum()), "updated_trials": int((updated & mask).sum()),
                            "own_margin": own_mass, "helpful_margin": useful_mass,
                            "harmful_margin": float(np.maximum(-delta, 0.)[mask[:, None] & cross].sum()),
                            "unknown_harmful_margin": float(np.maximum(-delta, 0.)[
                                mask[:, None] & cross & (known == 0)].sum()),
                            "total_margin": float(delta[mask].sum()),
                            "cumulative_step_l2": float(length[mask].sum()),
                            "max_step_l2": float(length[mask].max()) if mask.any() else 0.,
                            "own_retention": own_mass / own_denominator if own_denominator > 0. else None,
                            "help_retention": useful_mass / helpful_denominator if helpful_denominator > 0. else None,
                            "joint_nondegrading_updates": int((joint & mask).sum()),
                            "joint_strict_improvement_updates": int((strict & mask).sum()),
                            "known_safety_violations": int((safety_violation & mask).sum())}
                        demand(tuple(row)[5:] == METRICS, "Independent fixed measurement metric order")
                        rows.append(row)
    return rows


def stats(values):
    valid = np.asarray([value for value in values if value is not None], dtype=float)
    return {"mean": float(valid.mean()) if valid.size else None,
            "sd": float(valid.std(ddof=1)) if valid.size > 1 else 0. if valid.size else None,
            "valid_count": int(valid.size), "total_count": len(values)}


def expected_summary(cfg, arrays, rows):
    indexed = {}
    for row in rows:
        key = (row["topology"], row["representation"], row["candidate"], row["seed"], row["phase"])
        demand(key not in indexed, "Unique expected candidate/seed/phase row")
        indexed[key] = row
    s, n, tol = len(cfg["model_seeds"]), cfg["training_trials"], cfg["numeric_tolerance"]
    demand(len(rows) == 6 * 7 * s * 3, "Complete independent measurement rows")
    families, balanced, safe = [], True, True
    additive = [metric for metric in METRICS if metric not in ("max_step_l2", "own_retention", "help_retention")]
    for topology in TOPOLOGIES:
        for representation in REPRESENTATIONS:
            for candidate in CANDIDATES:
                phases = {}
                for phase in PHASES:
                    group = [indexed[topology, representation, candidate, seed, phase] for seed in cfg["model_seeds"]]
                    ratios = {}
                    for name, field in (("own_retention", "own_margin"), ("help_retention", "helpful_margin")):
                        denominator = sum(indexed[topology, representation, "parent_feedback", seed, phase][field]
                                          for seed in cfg["model_seeds"])
                        ratios[name] = sum(row[field] for row in group) / denominator if denominator > 0. else None
                    phases[phase] = {"metrics": {metric: stats([row[metric] for row in group]) for metric in METRICS},
                                     "pooled": ratios}
                    if candidate not in ("shared_reference", "isolated_reference"):
                        safe &= all(row["known_safety_violations"] == 0 for row in group)
                for seed in cfg["model_seeds"]:
                    for metric in additive:
                        values = [indexed[topology, representation, candidate, seed, phase][metric] for phase in PHASES]
                        balanced &= abs(values[0] - values[1] - values[2]) <= tol
                role = "primary_descriptive" if (topology, representation, candidate) == (
                    "real", "raw", "observed_balanced") else "prespecified_complement" if candidate == "observed_own" else (
                    "privileged_diagnostic" if candidate.startswith("oracle_") else "descriptive_control")
                families.append({"topology": topology, "representation": representation, "candidate": candidate,
                                 "role": role, "phases": phases})
    cert = arrays["case_certificates"]
    error = float(max(np.max(np.abs(cert[:, 4:])), np.max(np.abs(cert[:, 2] - cert[:, 0]))))
    violation = float(np.maximum(arrays["candidate_step_lengths"] - np.abs(arrays["update_factors"])[..., None], 0.).max())
    unit = violation <= tol and np.max(cert[:, 3]) <= 1. + tol
    return {"study_id": cfg["study_id"], "status": "completed", "scope": cfg["scope"],
        "candidates": list(CANDIDATES), "phases": list(PHASES), "families": families,
        "validation": {"all_unit_caps_passed": bool(unit), "all_observed_and_oracle_constraints_passed": bool(safe),
            "all_case_certificates_passed": bool(error <= tol), "all_phase_balances_passed": bool(balanced),
            "max_certificate_residual": error, "max_step_bound_violation": violation},
        "counts": {"analyzed_parent_trial_records": 6 * s * n,
            "nonzero_parent_updates": int(np.count_nonzero(arrays["update_factors"])),
            "candidate_trial_records": 6 * s * n * 7, "candidate_destination_effects": 6 * s * n * 7 * 4,
            "unique_optimization_cases": len(arrays["case_keys"]), "measurement_records": len(rows),
            "conditions": 6, "model_seeds": s, "new_training_trials": 0, "new_head_update_calls": 0,
            "new_internal_update_calls": 0, "new_value_update_calls": 0,
            "new_random_draws": 0, "fresh_model_constructions": 0}, "interpretation": dict(INTERPRETATION)}


def check_rows(actual, wanted):
    demand(len(actual) == len(wanted), "Complete ordered measurement table")
    for row, expected in zip(actual, wanted):
        demand(tuple(row) == tuple(expected), "Exact measurement column order and closure")
        for key, value in expected.items():
            if value is None:
                demand(row[key] is None or row[key] == "", "Undefined ratio is empty: " + key)
            elif type(value) is str:
                demand(row[key] == value, "Measurement identity: " + key)
            else:
                number = float(row[key])
                correct = number.is_integer() and int(number) == value if type(value) is int else np.isclose(
                    number, value, atol=1e-9, rtol=1e-9)
                demand(np.isfinite(number) and correct, "Measurement arithmetic: " + key)


def verify_science(cfg, parent, arrays, rows, summary, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    wanted = expected_artifacts(cfg, parent, arrays)
    wanted_rows = expected_rows(cfg, wanted, parent)
    check_rows(rows, wanted_rows)
    wanted_summary = expected_summary(cfg, wanted, wanted_rows)
    demand(all(value for key, value in wanted_summary["validation"].items() if key.startswith("all_")),
           "Independent unit-cap/safety/certificate/partition gates")
    compare_tree(summary, wanted_summary, "Independent complete cone summary")
    count = 6 * len(cfg["model_seeds"]) * cfg["training_trials"]
    return {"arrays": len(FIELDS), "parent_trial_records_checked": count,
        "candidate_trial_records_checked": count * 7, "candidate_destination_effects_checked": count * 7 * 4,
        "unique_optimization_cases_checked": len(wanted["case_keys"]), "measurement_records_checked": len(wanted_rows),
        "active_faces_reoptimized_in_original_feature_space": True,
        "nonnegative_dual_upper_bound_independently_checked": True,
        "certified_direction_not_exact_degenerate_vector_identity": True,
        "observed_objectives_constraints_use_only_actual_observations": True,
        "oracle_full_information_separate_from_observed_mode": True,
        "no_go_cases_skipped_and_all_zero_effects": True,
        "same_saved_scalar_not_autonomous_policy": True,
        "parent_full_trajectory_replay_reused_by_frozen_hashes": True,
        "no_rng_learner_or_graph_construction_calls": True,
        "new_training_trials": 0, "new_random_draws": 0}


def load_rows(path):
    with Path(path).open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        demand(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), "Unique CSV columns")
        return list(reader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--publication-proof", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve() / "src"))
    from drosolearn.cone_envelope import verify_envelope, sha, write_new
    cfg, _, meta = verify_envelope(args.root, args.config, args.registration, args.result_dir, args.publication_proof)
    with np.load(args.root / cfg["source_result"] / "arrays.npz", allow_pickle=False) as saved:
        parent = {key: saved[key] for key in saved.files}
    with np.load(args.result_dir / "arrays.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    rows = load_rows(args.result_dir / "measurements.csv")
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf8"))
    checks = verify_science(cfg, parent, arrays, rows, summary)
    demand(meta["counts"] == summary["counts"], "Verified metadata/summary count binding")
    report = {"status": "passed", "study_id": cfg["study_id"],
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_sha256": sha(args.registration), "summary_sha256": sha(args.result_dir / "summary.json"),
        "scientific_producer_imported": False, "new_helper_imported": False, "counts": checks}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
