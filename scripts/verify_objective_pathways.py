"""Independent saved-only direction/size/choice accounting for U and P.

The scientific producer is never imported. Every mixed cell is arithmetic
on saved facts, not an executed policy or future reward path.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

TOPS = ("real", "rewired", "ann")
REPS = ("raw", "identity")
METHODS = ("shared", "isolated", "feedback_selective", "balanced", "gated_balanced", "gated_self_blend", "gated_uniform", "gated_probability")
SIGNS = np.asarray([1., 1., -1., -1.])
FACTORS = ("state", "scalar", "direction")
COMPARISONS = (("gated_uniform", "gated_balanced"), ("gated_probability", "gated_balanced"), ("gated_probability", "gated_uniform"))
GROUPS = ("own", "other")
COMPONENTS = FACTORS + ("control_only_Go", "candidate_only_Go", "neither_Go")
DIAGNOSTICS = ("aligned_source_logit_before", "correct_source_probability", "logistic_sensitivity_value",
    "abs_update_factor", "head_step_l2", "own_unit_direction_influence", "other_unit_aligned_influence_mean", "direction_l2",
    "observed_reward", "source_Go_probability", "own_aligned_margin_step", "other_aligned_margin_step_mean")
FIELDS = ("probability_changes", "margin_changes", "source_logits_before", "reference_post", "pair_action_cases", "bridge_eligible",
          "bridge_q_cells", "bridge_q_shapley", "bridge_positive_shapley", "bridge_negative_shapley", "factual_pair_diagnostics",
          "first_post_trials", "model_seeds", "schedules", "targets", "uniforms")


def demand(condition, label):
    if not condition:
        raise ValueError(label)


def close(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    demand(actual.shape == expected.shape and actual.dtype.kind in "biuf" and np.isfinite(actual).all(), label + " finite axes")
    if expected.dtype.kind in "biu":
        demand(actual.dtype.kind in "biu" and np.array_equal(actual, expected), label + " exact")
    else:
        demand(np.allclose(actual, expected, atol=1e-9, rtol=1e-9), label + " arithmetic")


def compare_tree(actual, expected, label="summary"):
    if isinstance(expected, dict):
        demand(isinstance(actual, dict) and actual.keys() == expected.keys(), label + " keys")
        for key in expected:
            compare_tree(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, list):
        demand(isinstance(actual, list) and len(actual) == len(expected), label + " length")
        for i, value in enumerate(expected):
            compare_tree(actual[i], value, label + "." + str(i))
    elif type(expected) is bool:
        demand(type(actual) is bool and actual == expected, label + " flag")
    elif type(expected) is int:
        demand(type(actual) is int and actual == expected, label + " integer")
    elif type(expected) is float:
        demand(type(actual) in (float, int) and np.isfinite(actual) and np.isclose(actual, expected, atol=1e-9, rtol=1e-9), label + " numeric")
    else:
        demand(actual == expected, label + " value")


def sigmoid(values):
    return 1. / (1. + np.exp(-np.clip(values, -40., 40.)))


def independent_bridge(z_control, z_new, g_control, g_new, f_control, f_new):
    zc, zn, gc, gn, fc, fn = map(np.asarray, (z_control, z_new, g_control, g_new, f_control, f_new))
    demand(zc.shape == zn.shape == fc.shape == fn.shape == gc.shape + (4,) and gc.shape == gn.shape
           and all(value.dtype.kind in "fiu" and np.isfinite(value).all() for value in (zc, zn, gc, gn, fc, fn)), "Recorded eight-corner finite axes")
    cells = np.empty(gc.shape + (8, 4))
    for code in range(8):
        z, g, f = (zn if code & 4 else zc), (gn if code & 2 else gc), (fn if code & 1 else fc)
        cells[..., code, :] = SIGNS * (sigmoid(z + g[..., None] * f) - sigmoid(z))
    return cells


def independent_shapley(cells):
    value = np.asarray(cells)
    demand(value.ndim >= 2 and value.shape[-2:] == (8, 4) and value.dtype.kind in "fiu"
           and np.isfinite(value).all(), "Eight finite exact probability cells")
    value = value.astype(float)
    output = []
    for bit in (4, 2, 1):
        other = [item for item in (4, 2, 1) if item != bit]
        allocation = (value[..., bit, :] - value[..., 0, :]) / 3.
        allocation += ((value[..., bit | other[0], :] - value[..., other[0], :])
                      + (value[..., bit | other[1], :] - value[..., other[1], :])) / 6.
        allocation += (value[..., 7, :] - value[..., 7 ^ bit, :]) / 3.
        output.append(allocation)
    result = np.stack(output, axis=-2)
    close(result.sum(axis=-2), value[..., 7, :] - value[..., 0, :], "Symmetric Shapley efficiency")
    return result


def factual_states(cfg, parent):
    """Reconstruct saved head prefixes, not a new actor or reward process."""
    s, n = len(cfg["model_seeds"]), cfg["training_trials"]
    shape = (3, 2, 8, s, n)
    close(parent["model_seeds"], np.asarray(cfg["model_seeds"]), "Frozen original computational seeds")
    close(parent["targets"], (SIGNS > 0).astype(int), "Frozen four reward target identities")
    source, uniforms = parent["schedules"], parent["uniforms"]
    demand(source.shape == (s, n) and source.dtype.kind in "iu" and np.isin(source, range(4)).all(), "Recorded integer source schedules")
    demand(uniforms.shape == (s, n) and uniforms.dtype.kind in "fiu" and np.isfinite(uniforms).all()
           and np.all((uniforms >= 0.) & (uniforms < 1.)), "Recorded common choice uniforms")
    actions = parent["trial_actions"]
    demand(actions.shape == shape and actions.dtype.kind in "biu" and np.isin(actions, (0, 1)).all(), "Recorded binary actual choices")
    before, after = parent["trial_known_before"], parent["trial_known_after"]
    demand(before.shape == after.shape == shape + (4,) and before.dtype.kind in "biu" and after.dtype.kind in "biu"
           and np.isin(before, (-1, 0, 1)).all() and np.isin(after, (-1, 0, 1)).all(), "Recorded actual reward sign memory")
    full_before = np.all(before != 0, axis=-1)
    post = full_before[:, :, 4]
    for mi in (5, 6, 7):
        close(full_before[:, :, mi], post, "Common gated full-before boundary")
    grid = parent["checkpoints"]
    demand(grid.ndim == 1 and grid.dtype.kind in "iu" and grid.tolist() == sorted(set(grid.tolist()))
           and grid[0] == 0 and grid[-1] == n, "Frozen complete checkpoint grid")
    g, w = parent["trial_update_factors"], parent["trial_directions"]
    demand(g.shape == shape and w.shape[:-1] == shape and w.shape[-1] > 0 and g.dtype.kind in "fiu" and w.dtype.kind in "fiu"
           and np.isfinite(g).all() and np.isfinite(w).all(), "Recorded finite actual signed scalar and direction")
    q, margin, logits, influence = (np.empty(shape + (4,)) for _ in range(4))
    residuals = {name: 0. for name in ("factual_parent_effects", "choice_probability", "signed_update_equation", "head_costs", "unknown_harm", "endpoint_telescoping")}

    def check(actual, expected, family):
        close(actual, expected, "Independent factual " + family)
        residuals[family] = max(residuals[family], float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float)), initial=0.)))

    for ti, top in enumerate(TOPS):
        for ri, rep in enumerate(REPS):
            features = parent["g_" + top + "_" + rep + "_unit_rows"]
            demand(features.shape == (s, 4, w.shape[-1]) and features.dtype.kind in "fiu" and np.isfinite(features).all(), "Common recorded fixed feature rows")
            close(np.linalg.norm(features, axis=-1), np.ones((s, 4)), "Frozen unit feature normalization")
            for si in range(s):
                d, src = features[si], source[si]
                own = np.eye(4, dtype=bool)[src]
                for mi in range(8):
                    at = (ti, ri, mi, si)
                    steps = g[at][:, None] * w[at]
                    heads = np.concatenate((np.zeros((1, d.shape[1])), np.cumsum(steps, axis=0)))
                    logits[at] = heads[:-1] @ d.T
                    influence[at] = w[at] @ d.T
                    delta = g[at][:, None] * influence[at]
                    q[at] = SIGNS * (sigmoid(logits[at] + delta) - sigmoid(logits[at]))
                    margin[at] = SIGNS * delta
                    p = sigmoid(logits[at][np.arange(n), src])
                    check(parent["trial_probabilities"][at], p, "choice_probability")
                    close(actions[at], (uniforms[si] < p).astype(int), "Actual saved uniform threshold choices")
                    reward = actions[at] * SIGNS[src].astype(int)
                    close(parent["trial_rewards"][at], reward, "Only actually obtained reward")
                    check(g[at], .1 * (actions[at] - p) * reward, "signed_update_equation")
                    demand(np.all(g[at][actions[at] == 0] == 0.), "NoGo has no actual signed update")
                    optimized_mode = np.ones(n, bool) if mi == 3 else full_before[at] if mi >= 4 else np.zeros(n, bool)
                    demand(np.all(w[at][optimized_mode & (actions[at] == 0)] == 0.), "Actual NoGo in optimized mode has zero recorded direction")
                    expected_before, expected_after = np.zeros((n, 4), np.int8), np.zeros((n, 4), np.int8)
                    memory = np.zeros(4, np.int8)
                    for t, index in enumerate(src):
                        expected_before[t] = memory
                        if actions[at][t]:
                            memory[index] = reward[t]
                        expected_after[t] = memory
                    close(before[at], expected_before, "Actual feedback memory before update")
                    close(after[at], expected_after, "Actual feedback memory after update")
                    for actual, expected in ((parent["trial_correct_probability_delta"][at], q[at]),
                            (parent["trial_aligned_margin_delta"][at], margin[at]),
                            (parent["checkpoint_heads"][at], heads[grid]),
                            (parent["checkpoint_probabilities"][at], sigmoid(heads[grid] @ d.T))):
                        check(actual, expected, "factual_parent_effects")
                    costs = np.linalg.norm(steps, axis=-1)
                    for name, expected in (("trial_head_steps", costs), ("trial_head_step_squared_l2", costs ** 2),
                            ("trial_cumulative_head_path_l2", costs.cumsum()), ("trial_cumulative_head_squared_l2", (costs ** 2).cumsum())):
                        check(parent[name][at], expected, "head_costs")
                    unknown = (before[at] == 0) & ~own
                    close(parent["trial_unknown_other_mask_before"][at], unknown, "Unknown other before actual feedback")
                    close(parent["trial_unknown_other_mask_after"][at], (after[at] == 0) & ~own, "Unknown other after actual feedback")
                    check(parent["trial_unknown_other_negative_correct_probability_sum"][at], np.where(unknown, np.maximum(-q[at], 0.), 0.).sum(axis=-1), "unknown_harm")
                    final = SIGNS * (sigmoid(heads[-1] @ d.T) - .5)
                    check(q[at].sum(axis=0), final, "endpoint_telescoping")
                pre = ~post[ti, ri, si]
                for mi in (5, 6, 7):
                    for name in ("trial_actions", "trial_rewards", "trial_probabilities", "trial_update_factors", "trial_directions", "trial_known_before", "trial_known_after"):
                        demand(np.array_equal(parent[name][ti, ri, mi, si][pre], parent[name][ti, ri, 4, si][pre]), "Exact common isolated prefix")
    first = np.where(post.any(axis=-1), post.argmax(axis=-1) + 1, n + 1).astype(np.int32)
    for name, mi in (("first_balanced_mode_trials", 4), ("first_blend_mode_trials", 5), ("first_uniform_mode_trials", 6), ("first_probability_mode_trials", 7)):
        close(parent[name][:, :, mi], first, "Original first gated mode boundary")
    return {"q": q, "margin": margin, "logits": logits, "influence": influence, "post": post, "first": first,
            "full_before": full_before, "residuals": residuals}


def validate_science(cfg, allow_fixture=False):
    fixed = {"study_id": "objective_pathways_v1", "scope": "retrospective_descriptive_objective_pathways",
        "module": "drosolearn.objective_pathways", "protocol": "docs/objective-pathways-protocol.md",
        "learning_result": "results/objective_learning_v1", "baseline_tag": "objective-learning-v1",
        "baseline_commit": "b96a6c9d88a9f83cbe4e093272eea0ca77f245fe", "topologies": list(TOPS),
        "representations": list(REPS), "methods": list(METHODS), "comparisons": [list(item) for item in COMPARISONS],
        "model_seeds": list(range(17801, 17811)), "training_trials": 400, "numeric_tolerance": 1e-8,
        "phase_definition": "all_three_gated_known_before_count_equals_four", "bridge_eligibility": "common_post_and_both_actual_Go",
        "bridge_bits": list(FACTORS), "bridge_allocation": "three_factor_symmetric_Shapley_exact_sigmoid",
        "destination_groups": list(GROUPS), "choice_reward_phases": ["all", "pre", "post"], "choice_reward_actions": [0, 1],
        "choice_reward_values": [-1, 0, 1], "new_confirmatory_hypothesis": False, "external_manifest": None}
    demand(isinstance(cfg, dict), "Independent fixed saved-path configuration")
    fixture = cfg.get("scope") == "integration_test"
    demand(not fixture or allow_fixture, "Invented fixture cannot enter actual registered analysis")
    for key, value in fixed.items():
        if fixture and key in ("scope", "model_seeds", "training_trials"):
            continue
        demand(cfg.get(key) == value, "Independent fixed objective pathway choice: " + key)
    if fixture:
        seeds, n = cfg.get("model_seeds"), cfg.get("training_trials")
        demand(type(seeds) is list and 1 <= len(seeds) <= 2 and len(set(seeds)) == len(seeds)
               and all(type(seed) is int and 90001 <= seed <= 90100 for seed in seeds), "Invented diagnostic seeds only")
        demand(type(n) is int and 4 <= n <= 16, "Bounded invented diagnostic path")


def independent_diagnostics(state, parent, path, mask, order):
    source = order[mask]
    z, f = state["logits"][path][mask], state["influence"][path][mask]
    factors = parent["trial_update_factors"][path][mask]
    norms = np.linalg.norm(parent["trial_directions"][path][mask], axis=-1)
    own = np.eye(4, dtype=bool)[source]
    p_go = sigmoid(z[np.arange(len(source)), source])
    aligned = SIGNS[source, None] * SIGNS * f
    actual_margin = SIGNS * factors[:, None] * f
    return np.column_stack((SIGNS[source] * z[np.arange(len(source)), source],
        np.where(SIGNS[source] == 1, p_go, 1. - p_go), p_go * (1. - p_go), np.abs(factors), np.abs(factors) * norms,
        f[np.arange(len(source)), source], np.where(own, 0., aligned).sum(axis=-1) / 3., norms,
        parent["trial_rewards"][path][mask], p_go, actual_margin[np.arange(len(source)), source],
        np.where(own, 0., actual_margin).sum(axis=-1) / 3.))


def expected_artifacts(cfg, parent):
    state = factual_states(cfg, parent)
    s, n = len(cfg["model_seeds"]), cfg["training_trials"]
    pair_shape = (3, 2, 3, s, n)
    cases, eligible = np.empty(pair_shape, np.int8), np.empty(pair_shape, bool)
    cells = np.zeros(pair_shape + (8, 4))
    diagnostics = np.zeros(pair_shape + (2, 12))
    for ti in range(3):
        for ri in range(2):
            for ci, (candidate, control) in enumerate(COMPARISONS):
                cm, nm = METHODS.index(control), METHODS.index(candidate)
                for si, order in enumerate(parent["schedules"]):
                    at, c, new = (ti, ri, ci, si), (ti, ri, cm, si), (ti, ri, nm, si)
                    cases[at] = parent["trial_actions"][c] + 2 * parent["trial_actions"][new]
                    mask = state["post"][ti, ri, si] & (cases[at] == 3)
                    eligible[at] = mask
                    cells[at][mask] = independent_bridge(state["logits"][c][mask], state["logits"][new][mask],
                        parent["trial_update_factors"][c][mask], parent["trial_update_factors"][new][mask],
                        state["influence"][c][mask], state["influence"][new][mask])
                    close(cells[at][mask, 0], state["q"][c][mask], "Control actual corner")
                    close(cells[at][mask, 7], state["q"][new][mask], "Candidate actual corner")
                    for side, path in enumerate((c, new)):
                        diagnostics[at][mask, side] = independent_diagnostics(state, parent, path, mask, order)
    signed = independent_shapley(cells)
    positive = independent_shapley(np.maximum(cells, 0.))
    negative = independent_shapley(np.maximum(-cells, 0.))
    source = parent["schedules"][None, None, None, ..., None]
    source_logits = np.take_along_axis(state["logits"], np.broadcast_to(source, state["logits"].shape[:-1] + (1,)), axis=-1)[..., 0]
    arrays = {"probability_changes": state["q"], "margin_changes": state["margin"], "source_logits_before": source_logits,
        "reference_post": state["post"], "pair_action_cases": cases, "bridge_eligible": eligible, "bridge_q_cells": cells,
        "bridge_q_shapley": signed, "bridge_positive_shapley": positive, "bridge_negative_shapley": negative,
        "factual_pair_diagnostics": diagnostics, "first_post_trials": state["first"],
        "model_seeds": parent["model_seeds"].copy(), "schedules": parent["schedules"].copy(), "targets": parent["targets"].copy(),
        "uniforms": parent["uniforms"].copy()}
    demand(tuple(arrays) == FIELDS, "Complete sixteen-array scientific closure")
    return arrays, expected_rows(cfg, arrays)


def expected_rows(cfg, arrays):
    rows = []
    for ti, top in enumerate(TOPS):
        for ri, rep in enumerate(REPS):
            for ci, (candidate, control) in enumerate(COMPARISONS):
                cm, nm = METHODS.index(control), METHODS.index(candidate)
                comparison = candidate + "_minus_" + control
                for si, seed in enumerate(cfg["model_seeds"]):
                    at = (ti, ri, ci, si)
                    c, new = (ti, ri, cm, si), (ti, ri, nm, si)
                    post, eligible = arrays["reference_post"][ti, ri, si], arrays["bridge_eligible"][at]
                    cases = arrays["pair_action_cases"][at]
                    own = np.eye(4, dtype=bool)[arrays["schedules"][si]]
                    for group, destination in zip(GROUPS, (own, ~own)):
                        for ki, component in enumerate(COMPONENTS):
                            if ki < 3:
                                signed = float(arrays["bridge_q_shapley"][at][:, ki][destination].sum())
                                positive = float(arrays["bridge_positive_shapley"][at][:, ki][destination].sum())
                                negative = float(arrays["bridge_negative_shapley"][at][:, ki][destination].sum())
                                count = int(eligible.sum())
                            else:
                                code = (1, 2, 0)[ki - 3]
                                mask = (post & (cases == code))[:, None] & destination
                                value = arrays["probability_changes"][c if code == 1 else new]
                                sign = -1. if code == 1 else 1. if code == 2 else 0.
                                signed = float(sign * value[mask].sum())
                                positive = float(sign * np.maximum(value[mask], 0.).sum())
                                negative = float(sign * np.maximum(-value[mask], 0.).sum())
                                count = int(np.count_nonzero(post & (cases == code)))
                            rows.append({"topology": top, "representation": rep, "comparison": comparison, "seed": seed,
                                "destination_group": group, "component": component, "reference_post_trials": int(post.sum()),
                                "both_Go_trials": int(eligible.sum()), "component_trials": count, "probability_signed_sum": signed,
                                "positive_mass_allocation": positive, "negative_mass_allocation": negative,
                                "endpoint_contribution": signed / 4., "positive_endpoint_allocation": positive / 4.,
                                "negative_endpoint_allocation": negative / 4.})
    return rows


def mean_nested(values):
    if isinstance(values[0], list):
        return [mean_nested(list(column)) for column in zip(*values)]
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def independent_paths(cfg, arrays, parent):
    paths, residual = [], 0.
    for ti, top in enumerate(TOPS):
        for ri, rep in enumerate(REPS):
            for mi, method in enumerate(METHODS):
                records = []
                for si, seed in enumerate(cfg["model_seeds"]):
                    at = ti, ri, mi, si
                    src = arrays["schedules"][si]
                    action, reward = parent["trial_actions"][at], parent["trial_rewards"][at]
                    factors, directions = parent["trial_update_factors"][at], parent["trial_directions"][at]
                    changes = arrays["probability_changes"][at]
                    p = sigmoid(arrays["source_logits_before"][at])
                    step = np.abs(factors) * np.linalg.norm(directions, axis=-1)
                    own = np.eye(4, dtype=bool)[src]
                    unknown = (parent["trial_known_before"][at] == 0) & ~own
                    checkpoints = parent["checkpoint_probabilities"][at]
                    initial = np.where(SIGNS > 0, checkpoints[0], 1. - checkpoints[0])
                    final = np.where(SIGNS > 0, checkpoints[-1], 1. - checkpoints[-1])
                    residual = max(residual, float(np.max(np.abs(changes.sum(axis=0) - final + initial))))
                    phases, strata = {}, []
                    masks = (np.ones(cfg["training_trials"], bool), ~arrays["reference_post"][ti, ri, si], arrays["reference_post"][ti, ri, si])
                    for phase, mask in zip(("all", "pre", "post"), masks):
                        size = int(np.count_nonzero(mask))
                        correct = np.where(SIGNS[src] > 0, p, 1. - p)
                        phases[phase] = {"trials": size, "Go": int(action[mask].sum()), "NoGo": int(np.count_nonzero(action[mask] == 0)),
                            "reward_positive": int(np.count_nonzero(reward[mask] > 0)), "reward_negative": int(np.count_nonzero(reward[mask] < 0)),
                            "obtained_reward_sum": int(reward[mask].sum()), "expected_obtained_reward_sum": float((p * SIGNS[src])[mask].sum()),
                            "source_Go_probability_mean": float(p[mask].mean()) if size else None,
                            "correct_source_probability_mean": float(correct[mask].mean()) if size else None,
                            "head_path_l2": float(step[mask].sum()), "head_squared_l2": float(np.square(step[mask]).sum()),
                            "unknown_other_damage": float(np.maximum(-changes, 0.)[unknown & mask[:, None]].sum()),
                            "own_endpoint_contribution": float(changes[own & mask[:, None]].sum() / 4.),
                            "other_endpoint_contribution": float(changes[~own & mask[:, None]].sum() / 4.)}
                        for source in range(4):
                            for chosen in (0, 1):
                                for obtained in (-1, 0, 1):
                                    selected = mask & (src == source) & (action == chosen) & (reward == obtained)
                                    count = int(np.count_nonzero(selected))
                                    strata.append({"phase": phase, "source": source, "action": chosen, "reward": obtained, "trials": count,
                                        "obtained_reward_sum": int(reward[selected].sum()), "absolute_scalar_sum": float(np.abs(factors[selected]).sum()),
                                        "head_path_l2": float(step[selected].sum()), "source_probability_mean": float(p[selected].mean()) if count else None,
                                        "own_endpoint_contribution": float(changes[own & selected[:, None]].sum() / 4.),
                                        "other_endpoint_contribution": float(changes[~own & selected[:, None]].sum() / 4.)})
                    for phase in ("all", "pre", "post"):
                        demand(sum(row["trials"] for row in strata if row["phase"] == phase) == phases[phase]["trials"], "Complete 24 actual source/action/reward cells per phase")
                    records.append({"seed": seed, "initial_correct_probabilities": initial.tolist(), "final_correct_probabilities": final.tolist(),
                        "final_mean": float(final.mean()), "phases": phases, "source_action_reward_strata": strata})
                paths.append({"topology": top, "representation": rep, "method": method, "seeds": records,
                    "final_mean": float(np.mean([record["final_mean"] for record in records]))})
    return paths, residual


def expected_summary(cfg, arrays, rows, parent):
    state = factual_states(cfg, parent)
    indexed = {}
    for row in rows:
        key = tuple(row[name] for name in ("topology", "representation", "comparison", "seed", "destination_group", "component"))
        demand(key not in indexed, "Unique objective pathway component measurement")
        indexed[key] = row
    families, accounting_residual = [], 0.
    q = arrays["probability_changes"]
    for ti, top in enumerate(TOPS):
        for ri, rep in enumerate(REPS):
            for ci, (candidate, control) in enumerate(COMPARISONS):
                comparison = candidate + "_minus_" + control
                cm, nm = METHODS.index(control), METHODS.index(candidate)
                records = []
                for si, seed in enumerate(cfg["model_seeds"]):
                    at, c, new = (ti, ri, ci, si), (ti, ri, cm, si), (ti, ri, nm, si)
                    post = arrays["reference_post"][ti, ri, si]
                    own = np.eye(4, dtype=bool)[arrays["schedules"][si]]
                    components, positive, negative, factual, pre, factual_pos, factual_neg = [], [], [], [], [], [], []
                    for group, destination in zip(GROUPS, (own, ~own)):
                        group_rows = [indexed[top, rep, comparison, seed, group, name] for name in COMPONENTS]
                        parts = [[row[field] for row in group_rows] for field in ("endpoint_contribution", "positive_endpoint_allocation", "negative_endpoint_allocation")]
                        components.append(parts[0]); positive.append(parts[1]); negative.append(parts[2])
                        selected = post[:, None] & destination
                        factual.append(float((q[new] - q[c])[selected].sum() / 4.))
                        pre.append(float((q[new] - q[c])[~post[:, None] & destination].sum() / 4.))
                        factual_pos.append(float((np.maximum(q[new], 0.) - np.maximum(q[c], 0.))[selected].sum() / 4.))
                        factual_neg.append(float((np.maximum(-q[new], 0.) - np.maximum(-q[c], 0.))[selected].sum() / 4.))
                        accounting_residual = max(accounting_residual, abs(sum(parts[0]) - factual[-1]),
                            abs(sum(parts[1]) - factual_pos[-1]), abs(sum(parts[2]) - factual_neg[-1]))
                    eligible = arrays["bridge_eligible"][at]
                    diagnostics = arrays["factual_pair_diagnostics"][at][eligible]
                    difference = parent["checkpoint_probabilities"][new][-1] - parent["checkpoint_probabilities"][c][-1]
                    endpoint = float((SIGNS * difference).mean())
                    accounting_residual = max(accounting_residual, abs(sum(pre) + sum(factual) - endpoint))
                    divergence = np.flatnonzero(parent["trial_actions"][new] != parent["trial_actions"][c])
                    gc, gn = parent["trial_update_factors"][c][eligible], parent["trial_update_factors"][new][eligible]
                    rc, rn = parent["trial_rewards"][c][eligible], parent["trial_rewards"][new][eligible]
                    close(rc, rn, "Identical obtained reward on actual paired Go")
                    pc, pn = parent["trial_probabilities"][c][eligible], parent["trial_probabilities"][new][eligible]
                    scalar_residual = float(np.max(np.abs(gn - gc + .1 * rc * (pn - pc)), initial=0.))
                    accounting_residual = max(accounting_residual, scalar_residual)
                    records.append({"seed": seed, "reference_post_trials": int(post.sum()), "both_Go_trials": int(eligible.sum()),
                        "case_counts": [int(np.count_nonzero(post & (arrays["pair_action_cases"][at] == case))) for case in range(4)],
                        "component_endpoint_contributions": components, "component_positive_endpoint_allocations": positive,
                        "component_negative_endpoint_allocations": negative, "factual_pre_endpoint_difference": pre,
                        "factual_post_endpoint_difference": factual, "factual_post_positive_endpoint_difference": factual_pos,
                        "factual_post_negative_endpoint_difference": factual_neg, "final_endpoint_difference": endpoint,
                        "diagnostic_means": diagnostics.mean(axis=0).tolist() if len(diagnostics) else [[None] * 12 for _ in range(2)],
                        "first_actual_action_divergence": int(divergence[0] + 1) if len(divergence) else None,
                        "both_Go_scalar_probability_identity_residual": scalar_residual})
                means = {name: mean_nested([record[name] for record in records]) for name in records[0] if name != "seed"}
                families.append({"topology": top, "representation": rep, "candidate": candidate, "control": control,
                    "comparison": comparison, "role": "primary" if (ti, ri) == (0, 0) else "descriptive_control", "seeds": records, "means": means})
    paths, telescope = independent_paths(cfg, arrays, parent)
    cells, eligible = arrays["bridge_q_cells"], arrays["bridge_eligible"]
    residuals = {"factual_parent_reconstruction": max(state["residuals"].values()), "complete_endpoint_accounting": accounting_residual,
        "all_path_telescoping": telescope,
        "shapley_efficiency": float(np.max(np.abs(arrays["bridge_q_shapley"].sum(axis=-2) - cells[..., 7, :] + cells[..., 0, :]))),
        "signed_positive_negative_identity": float(np.max(np.abs(arrays["bridge_q_shapley"] - arrays["bridge_positive_shapley"] + arrays["bridge_negative_shapley"]))),
        "ineligible_zero_padding": float(max(np.max(np.abs(arrays[name][~eligible]), initial=0.) for name in (
            "bridge_q_cells", "bridge_q_shapley", "bridge_positive_shapley", "bridge_negative_shapley", "factual_pair_diagnostics")))}
    demand(all(np.isfinite(value) and value <= cfg["numeric_tolerance"] for value in residuals.values()), "Complete independent saved objective accounting gates")
    demand(len(arrays) == 16 and len(rows) == 6 * 3 * len(cfg["model_seeds"]) * 2 * 6, "Complete sixteen arrays and factor rows")
    return {"study_id": cfg["study_id"], "status": "completed", "scope": cfg["scope"], "families": families, "actual_paths": paths,
        "component_order": list(COMPONENTS), "diagnostic_order": list(DIAGNOSTICS), "group_order": list(GROUPS),
        "comparison_order": [list(item) for item in COMPARISONS],
        "validation": {"all_exact_accounting_gates_passed": True, "residuals": residuals,
            "same_prefeedback_full_gate_and_prefix_checked": True, "only_actual_both_Go_crossed": True,
            "one_sided_actual_choices_retained": True, "recorded_common_uniform_and_own_probabilities_checked": True,
            "both_Go_rewards_identical_and_scalar_difference_from_probability": True},
        "counts": {"analyzed_parent_trial_records": int(np.prod(q.shape[:-1])), "analyzed_destination_effects": int(q.size),
            "reference_gate_paths": int(np.prod(arrays["reference_post"].shape[:-1])), "comparison_paths": int(np.prod(eligible.shape[:-1])),
            "paired_trial_records": int(eligible.size), "paired_post_trial_records": int(arrays["reference_post"].sum() * 3),
            "eligible_both_Go_trials": int(eligible.sum()), "corner_destination_effects": int(eligible.sum() * 32), "numeric_arrays": len(arrays),
            "measurement_records": len(rows), "actual_path_strata": len(paths) * len(cfg["model_seeds"]) * 72,
            **{name: 0 for name in ("new_training_trials", "new_head_update_calls", "new_internal_update_calls", "new_value_update_calls",
                "new_random_draws", "fresh_model_constructions", "new_optimization_calls")}},
        "interpretation": {"retrospective_descriptive_only": True, "new_confirmatory_hypothesis": False,
            "accounting_allocation_not_unique_causal_explanation": True, "mixed_cells_not_relearned_policy": True,
            "NoGo_potential_rewards_or_directions_not_generated": True, "same_realized_movement_budget_claimed": False,
            "parent_U_gain": False, "parent_P_gain": False, "parent_U_acquisition": False, "parent_P_acquisition": False,
            "parent_U_protection": True, "parent_P_protection": True, "parent_state_gain": False,
            "novel_input_generalization_tested": False, "new_human_experiment": False,
            "shared_neural_mechanism_claim": False, "education_effect_claim": False}}


def check_artifacts(arrays, wanted, rows, wanted_rows):
    demand(set(arrays) == set(wanted) == set(FIELDS), "Independent sixteen-array closure")
    for name, value in wanted.items():
        close(arrays[name], value, "Independent objective pathway array: " + name)
    demand(len(rows) == len(wanted_rows), "Complete ordered objective pathway measurement rows")
    for actual, expected in zip(rows, wanted_rows):
        demand(tuple(actual) == tuple(expected), "Exact objective pathway CSV columns")
        for name, value in expected.items():
            if type(value) is str:
                demand(actual[name] == value, "Objective pathway row identity: " + name)
            else:
                number = float(actual[name])
                demand(np.isfinite(number) and (number == value if type(value) is int else np.isclose(number, value, atol=1e-9, rtol=1e-9)),
                    "Objective pathway row exact count or arithmetic: " + name)


def verify_science(cfg, parent, arrays, rows, summary, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    wanted, wanted_rows = expected_artifacts(cfg, parent)
    check_artifacts(arrays, wanted, rows, wanted_rows)
    expected = expected_summary(cfg, wanted, wanted_rows, parent)
    compare_tree(summary, expected, "Independent complete objective pathways summary")
    trials = int(np.prod(parent["trial_actions"].shape))
    return {"arrays": len(wanted), "parent_trial_records_checked": trials, "destination_effects_checked": trials * 4,
        "measurement_records_checked": len(wanted_rows), "actual_paths_checked": len(expected["actual_paths"]) * len(cfg["model_seeds"]),
        "source_action_reward_strata_checked": expected["counts"]["actual_path_strata"],
        "factual_head_prefixes_and_parent_effects_checked": True, "actual_saved_uniform_threshold_choices_checked": True,
        "only_actually_obtained_rewards_and_memory_checked": True, "signed_scalar_probability_identity_checked": True,
        "same_prefeedback_gate_and_exact_prefix_checked": True, "only_post_both_actual_Go_eight_corners_checked": True,
        "state_scalar_direction_symmetric_allocations_checked": True, "signed_positive_and_negative_mass_allocations_checked": True,
        "own_other_and_all_unpaired_actual_actions_accounted": True, "all_paths_strata_costs_unknown_damage_and_endpoints_checked": True,
        "factual_diagnostics_and_nullable_means_checked": True, "no_rng_model_graph_solver_or_future_policy_calls": True,
        "new_training_trials": 0, "new_random_draws": 0, "new_optimization_calls": 0}


def load_rows(path):
    with Path(path).open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        demand(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames), "Unique objective pathway CSV columns")
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
    from drosolearn.objective_pathways_envelope import verify_envelope, sha, write_new
    cfg, _, metadata = verify_envelope(args.root, args.config, args.registration, args.result_dir, args.publication_proof)
    with np.load(args.root / cfg["learning_result"] / "arrays.npz", allow_pickle=False) as handle:
        parent = {name: handle[name] for name in handle.files}
    with np.load(args.result_dir / "arrays.npz", allow_pickle=False) as handle:
        arrays = {name: handle[name] for name in handle.files}
    rows = load_rows(args.result_dir / "measurements.csv")
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf8"))
    checks = verify_science(cfg, parent, arrays, rows, summary)
    demand(metadata["counts"] == summary["counts"], "Metadata and independently checked objective pathway counts")
    report = {"status": "passed", "study_id": cfg["study_id"], "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_sha256": sha(args.registration), "summary_sha256": sha(args.result_dir / "summary.json"),
        "scientific_producer_imported": False, "new_helper_imported": False, "counts": checks}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
