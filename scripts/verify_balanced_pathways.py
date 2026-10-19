"""Independent descriptive accounting of frozen balanced-learning trajectories.

No producer, optimizer, learner, graph constructor or random generator is
imported or called. The original full autonomous replay is a frozen upstream
receipt; this script derives saved-path arithmetic directly from its arrays.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

TOPOLOGIES = ("real", "rewired", "ann")
REPRESENTATIONS = ("raw", "identity")
METHODS = ("shared", "isolated", "feedback_selective", "balanced")
SIGNS = np.asarray([1., 1., -1., -1.])
PHASES = ("all", "partial", "full")
CATEGORIES = ("own", "known_other", "unknown_other")
FIELDS = ("probability_changes", "margin_changes", "effect_categories", "phase_before", "first_full_trials",
    "bridge_eligible", "bridge_q_cells", "bridge_q_shapley", "bridge_harm_shapley", "pair_unknown_membership",
    "objective_eligible", "objective_values", "objective_own_help_influences", "objective_matched_probability_gain",
    "objective_help_counts", "pair_action_cases", "model_seeds", "schedules", "targets")
METRICS = ("trial_count", "exposures", "nonzero_probability_effects", "positive_probability_effects",
    "negative_probability_effects", "probability_positive_mass", "probability_negative_mass", "probability_signed_sum",
    "margin_positive_mass", "margin_negative_mass", "margin_signed_sum")


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
        for key in expected: compare_tree(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, list):
        demand(isinstance(actual, list) and len(actual) == len(expected), label + " length")
        for i, value in enumerate(expected): compare_tree(actual[i], value, label + "." + str(i))
    elif type(expected) is bool:
        demand(type(actual) is bool and actual == expected, label + " flag")
    elif type(expected) is int:
        demand(type(actual) is int and actual == expected, label + " integer")
    elif type(expected) is float:
        demand(type(actual) in (float, int) and np.isfinite(actual)
               and np.isclose(actual, expected, atol=1e-9, rtol=1e-9), label + " numeric")
    else:
        demand(actual == expected, label + " value")


def sigmoid(values):
    return 1. / (1. + np.exp(-np.clip(values, -40., 40.)))


def validate_science(cfg, allow_fixture=False):
    fixed = {"study_id": "balanced_pathways_v1", "scope": "retrospective_descriptive_balanced_pathways",
        "module": "drosolearn.balanced_pathways", "protocol": "docs/balanced-pathways-protocol.md",
        "parent_result": "results/balanced_learning_v1", "baseline_tag": "balanced-learning-v1",
        "baseline_commit": "b836a780b91d81425e534dad7284485f284799ce", "topologies": list(TOPOLOGIES),
        "representations": list(REPRESENTATIONS), "methods": list(METHODS), "model_seeds": list(range(17501, 17511)),
        "training_trials": 400, "numeric_tolerance": 1e-8, "phase_definition": "known_before_count_equals_four",
        "category_definition": "own_known_before_other_unknown_before_other", "bridge_eligibility": "both_actual_Go_only",
        "bridge_bits": ["state", "scalar", "direction"], "bridge_allocation": "three_factor_symmetric_Shapley_exact_sigmoid",
        "objective_eligibility": "both_actual_Go_and_both_known_before_four", "new_confirmatory_hypothesis": False,
        "external_manifest": None}
    demand(isinstance(cfg, dict), "Independent saved-path configuration object")
    fixture = cfg.get("scope") == "integration_test"
    demand(not fixture or allow_fixture, "Invented fixtures cannot enter actual saved-path analysis")
    for key, value in fixed.items():
        if fixture and key in ("scope", "model_seeds", "training_trials"): continue
        demand(cfg.get(key) == value, "Fixed independent saved-path choice: " + key)
    if fixture:
        seeds, n = cfg.get("model_seeds"), cfg.get("training_trials")
        demand(type(seeds) is list and 1 <= len(seeds) <= 2 and len(set(seeds)) == len(seeds)
               and all(type(seed) is int and 90001 <= seed <= 90100 for seed in seeds), "Invented saved-record seeds only")
        demand(type(n) is int and 4 <= n <= 16, "Bounded invented saved-record length")


def independent_probability_accounting(pre_logits, post_logits):
    """Exact realized finite probability changes, with sign-separated masses."""
    before, after = np.asarray(pre_logits), np.asarray(post_logits)
    demand(before.shape == after.shape and before.ndim >= 2 and before.shape[-1] == 4
           and before.dtype.kind in "fiu" and after.dtype.kind in "fiu"
           and np.isfinite(before).all() and np.isfinite(after).all(), "Four finite destination logits")
    change = SIGNS * (sigmoid(after) - sigmoid(before))
    return {"delta": change, "positive": np.maximum(change, 0.), "negative": np.maximum(-change, 0.),
            "aligned_margin_delta": (after - before) * SIGNS}


def symmetric_product_accounting(g_balanced, g_selective, influence_balanced, influence_selective):
    """A symmetric algebraic identity; neither term is an intervention effect."""
    gb, gs, fb, fs = (np.asarray(value) for value in
                      (g_balanced, g_selective, influence_balanced, influence_selective))
    demand(gb.shape == gs.shape and fb.shape == fs.shape == gb.shape + (4,)
           and all(np.isfinite(value).all() for value in (gb, gs, fb, fs)), "Paired saved g/influence axes")
    direction = .5 * (gb + gs)[..., None] * (fb - fs)
    scalar = .5 * (fb + fs) * (gb - gs)[..., None]
    total = gb[..., None] * fb - gs[..., None] * fs
    close(direction + scalar, total, "Exact symmetric g-times-direction accounting")
    return {"direction_component": direction, "scalar_component": scalar, "total_difference": total}


def phase_masks(known_before):
    observed = np.asarray(known_before)
    demand(observed.ndim >= 2 and observed.shape[-1] == 4 and observed.dtype.kind in "biu"
           and np.isin(observed, (-1, 0, 1)).all(), "Explicit prefeedback observations")
    full = np.all(observed != 0, axis=-1)
    return np.ones_like(full), ~full, full


def destination_partition(sources, known_before):
    sources, known = np.asarray(sources), np.asarray(known_before)
    demand(sources.shape == known.shape[:-1] and sources.dtype.kind in "iu"
           and np.isin(sources, np.arange(4)).all(), "Current sources and prefeedback observations")
    own = np.eye(4, dtype=bool)[sources]
    unknown = (known == 0) & ~own
    familiar = (known != 0) & ~own
    demand(np.all(own.astype(int) + unknown + familiar == 1), "Self/known-other/unknown-other partition")
    return own, familiar, unknown


def independent_parent_state(cfg, parent):
    """Reconstruct the recorded heads and feedback without drawing a new choice."""
    s, n = len(cfg["model_seeds"]), cfg["training_trials"]
    grid = parent["checkpoints"].tolist()
    demand(type(grid) is list and grid == sorted(set(grid)) and grid[0] == 0 and grid[-1] == n
           and all(type(step) is int and 0 <= step <= n for step in grid), "Frozen parent checkpoint grid")
    shape = (3, 2, 4, s, n)
    close(parent["model_seeds"], np.asarray(cfg["model_seeds"]), "Frozen parent seed identity")
    close(parent["checkpoints"], np.asarray(grid), "Frozen parent checkpoint identity")
    close(parent["targets"], (SIGNS > 0).astype(int), "Frozen parent target identity")
    order, g, directions = parent["schedules"], parent["trial_update_factors"], parent["trial_directions"]
    demand(order.shape == (s, n) and order.dtype.kind in "iu" and np.isin(order, range(4)).all(),
           "Saved current-input schedule axes")
    demand(g.shape == shape and directions.shape[:-1] == shape and g.dtype.kind in "fiu"
           and directions.dtype.kind in "fiu" and np.isfinite(g).all() and np.isfinite(directions).all(),
           "Recorded update scalar/direction finite axes")
    dimension = directions.shape[-1]
    increments = g[..., None] * directions
    after_heads = np.cumsum(increments, axis=-2)
    before_heads = np.concatenate((np.zeros(shape[:-1] + (1, dimension)), after_heads[..., :-1, :]), axis=-2)
    before_logits, after_logits, influences = (np.empty(shape + (4,)) for _ in range(3))
    residual = 0.
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            rows = parent["g_" + topology + "_" + representation + "_unit_rows"]
            demand(rows.shape == (s, 4, dimension) and np.isfinite(rows).all(), "Recorded fixed feature axes")
            close(np.linalg.norm(rows, axis=-1), np.ones((s, 4)), "Recorded fixed unit features")
            for si in range(s):
                for mi in range(4):
                    at = (ti, ri, mi, si)
                    # The declared floating arithmetic is a per-path matrix
                    # product. Its exact zero signs must not depend on a
                    # different einsum reduction or sigmoid sign identity.
                    before_logits[at] = before_heads[at] @ rows[si].T
                    after_logits[at] = after_heads[at] @ rows[si].T
                    influences[at] = directions[at] @ rows[si].T
    account = independent_probability_accounting(before_logits, before_logits + g[..., None] * influences)
    account["aligned_margin_delta"] = g[..., None] * influences * SIGNS
    close(parent["trial_actual_influence"], influences, "Recorded directional destination influence")
    close(parent["trial_correct_probability_delta"], account["delta"], "Recorded actual probability change")
    close(parent["trial_aligned_margin_delta"], account["aligned_margin_delta"], "Recorded actual aligned logit change")
    source_grid = np.broadcast_to(order, shape)
    p_source = np.take_along_axis(sigmoid(before_logits), source_grid[..., None], axis=-1)[..., 0]
    close(parent["trial_probabilities"], p_source, "Saved policy from actual reconstructed head")
    actions, rewards = parent["trial_actions"], parent["trial_rewards"]
    demand(actions.shape == rewards.shape == shape and actions.dtype.kind in "biu"
           and rewards.dtype.kind in "biu" and np.isin(actions, (0, 1)).all(), "Recorded actual action/reward axes")
    demand(parent["uniforms"].shape == (s, n) and np.isfinite(parent["uniforms"]).all()
           and np.all((parent["uniforms"] >= 0.) & (parent["uniforms"] < 1.)), "Saved uniform axes/range, without new draw")
    close(actions, parent["uniforms"][None, None, None] < p_source, "No-new-draw saved action reconstruction")
    close(rewards, actions * SIGNS[source_grid].astype(int), "Obtained chosen feedback only")
    close(g, .1 * (actions - p_source) * rewards, "Recorded policy-gradient scalar")
    before, after = parent["trial_known_before"], parent["trial_known_after"]
    demand(before.shape == after.shape == shape + (4,) and before.dtype.kind in "biu"
           and after.dtype.kind in "biu" and np.isin(before, (-1, 0, 1)).all()
           and np.isin(after, (-1, 0, 1)).all(), "Recorded pre/post obtained-memory axes")
    close(before[..., 0, :], np.zeros(shape[:-1] + (4,), int), "Empty initial observed-reward memory")
    close(before[..., 1:, :], after[..., :-1, :], "Recorded observation continuity")
    observed_after = before.copy()
    for si in range(s):
        for t, source in enumerate(order[si]):
            old = before[..., si, t, source]
            action, reward = actions[..., si, t], rewards[..., si, t]
            demand(np.all((action == 0) | (old == 0) | (old == reward)), "Deterministic obtained-sign consistency")
            observed_after[..., si, t, source] = np.where(action == 1, reward, old)
    close(after, observed_after, "NoGo cannot reveal an unknown destination")
    own, familiar, unknown = destination_partition(source_grid, before)
    close(parent["trial_unknown_other_mask_before"], unknown, "Recorded unknown OTHER before actual feedback")
    close(parent["trial_unknown_other_mask_after"], (after == 0) & ~own, "Recorded unknown OTHER after actual feedback")
    close((after == 0) & ~own, unknown, "Current-source write cannot change other-unknown mask")
    for index, step in enumerate(grid):
        head = np.zeros(shape[:-1] + (dimension,)) if step == 0 else after_heads[..., step - 1, :]
        logits = np.zeros(shape[:-1] + (4,)) if step == 0 else after_logits[..., step - 1, :]
        close(parent["checkpoint_heads"][..., index, :], head, "Recorded checkpoint output weights")
        close(parent["checkpoint_logits"][..., index, :], logits, "Recorded checkpoint logits")
        close(parent["checkpoint_probabilities"][..., index, :], sigmoid(logits), "Recorded checkpoint output probabilities")
    for actual, wanted in ((parent["trial_actual_influence"], influences),
                            (parent["trial_correct_probability_delta"], account["delta"]),
                            (parent["trial_aligned_margin_delta"], account["aligned_margin_delta"]),
                            (parent["trial_probabilities"], p_source)):
        residual = max(residual, float(np.max(np.abs(actual - wanted))))
    telescope = account["delta"].sum(axis=-2) - (sigmoid(after_logits[..., -1, :] * SIGNS) - .5)
    close(telescope, np.zeros_like(telescope), "Per-destination exact saved-path telescoping")
    return {"before_logits": before_logits, "after_logits": after_logits,
        "correct_probability_delta": account["delta"], "aligned_margin_delta": account["aligned_margin_delta"],
        "positive_probability_delta": account["positive"], "negative_probability_delta": account["negative"],
        "own_mask": own, "known_other_mask": familiar, "unknown_other_mask": unknown,
        "full_known_before": np.all(before != 0, axis=-1), "nonzero_update": g != 0.,
        "influences": influences, "reconstruction_residual_max": residual,
        "telescoping_residual_max": float(np.max(np.abs(telescope)))}


def independent_bridge(z_selective, z_balanced, g_selective, g_balanced, f_selective, f_balanced):
    zs, zb, gs, gb, fs, fb = (np.asarray(value) for value in
                             (z_selective, z_balanced, g_selective, g_balanced, f_selective, f_balanced))
    demand(zs.shape == zb.shape == fs.shape == fb.shape == gs.shape + (4,) and gs.shape == gb.shape
           and all(np.isfinite(value).all() for value in (zs, zb, gs, gb, fs, fb)), "Eight corner state/scalar/direction axes")
    result = np.empty(zs.shape[:-1] + (8, 4))
    for code in range(8):
        z = zb if code & 4 else zs
        g = gb if code & 2 else gs
        f = fb if code & 1 else fs
        result[..., code, :] = SIGNS * (sigmoid(z + g[..., None] * f) - sigmoid(z))
    return result


def independent_shapley(cells):
    value = np.asarray(cells)
    demand(value.ndim >= 2 and value.shape[-2:] == (8, 4) and np.isfinite(value).all(), "Eight finite destination corner values")
    result = []
    for bit in (4, 2, 1):
        others = [item for item in (4, 2, 1) if item != bit]
        contribution = (value[..., bit, :] - value[..., 0, :]) / 3.
        contribution += ((value[..., bit | others[0], :] - value[..., others[0], :])
                         + (value[..., bit | others[1], :] - value[..., others[1], :])) / 6.
        contribution += (value[..., 7, :] - value[..., 7 ^ bit, :]) / 3.
        result.append(contribution)
    output = np.stack(result, axis=-2)
    close(output.sum(axis=-2), value[..., 7, :] - value[..., 0, :], "Exact three-player Shapley accounting")
    return output


def expected_artifacts(cfg, parent):
    state = independent_parent_state(cfg, parent)
    shape = parent["trial_actions"].shape
    s, n = len(cfg["model_seeds"]), cfg["training_trials"]
    source = np.broadcast_to(parent["schedules"], shape)
    category = np.full(shape + (4,), 2, np.int8)
    category[state["own_mask"]] = 0
    category[state["known_other_mask"]] = 1
    first = np.all(parent["trial_known_after"] != 0, axis=-1)
    discovered = first.any(axis=-1)
    first_full = np.where(discovered, first.argmax(axis=-1) + 1, n + 1).astype(np.int32)
    g_s, g_b = parent["trial_update_factors"][:, :, 2], parent["trial_update_factors"][:, :, 3]
    f_s, f_b = state["influences"][:, :, 2], state["influences"][:, :, 3]
    z_s, z_b = state["before_logits"][:, :, 2], state["before_logits"][:, :, 3]
    a_s, a_b = parent["trial_actions"][:, :, 2], parent["trial_actions"][:, :, 3]
    eligible = (a_s == 1) & (a_b == 1)
    corners = np.zeros(eligible.shape + (8, 4))
    corners[eligible] = independent_bridge(z_s[eligible], z_b[eligible],
        g_s[eligible], g_b[eligible], f_s[eligible], f_b[eligible])
    q_shapley = independent_shapley(corners)
    harm_shapley = independent_shapley(np.maximum(-corners, 0.))
    unknown_s, unknown_b = state["unknown_other_mask"][:, :, 2], state["unknown_other_mask"][:, :, 3]
    membership = np.zeros(unknown_s.shape, np.int8)
    membership[unknown_s & unknown_b] = 1
    membership[unknown_s & ~unknown_b] = 2
    membership[~unknown_s & unknown_b] = 3
    full_s, full_b = state["full_known_before"][:, :, 2], state["full_known_before"][:, :, 3]
    objective_eligible = eligible & full_s & full_b
    pair_shape = eligible.shape
    objective_values = np.zeros(pair_shape + (2,))
    own_help = np.zeros(pair_shape + (2, 2))
    help_counts = np.zeros(pair_shape, np.int8)
    matched = np.zeros(pair_shape)
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            all_d = parent["g_" + topology + "_" + representation + "_unit_rows"]
            for si in range(s):
                d, order = all_d[si], parent["schedules"][si]
                gram = d @ d.T
                for t in np.flatnonzero(objective_eligible[ti, ri, si]):
                    i = int(order[t])
                    helpful = (np.arange(4) != i) & (SIGNS[i] * SIGNS * gram[i] > cfg["numeric_tolerance"])
                    c = d[i].copy()
                    if helpful.any(): c = .5 * c + .5 * (SIGNS[i] * SIGNS[helpful, None] * d[helpful]).mean(axis=0)
                    at = (ti, ri, si, t)
                    help_counts[at] = int(helpful.sum())
                    for mi, parent_method in enumerate((2, 3)):
                        w = parent["trial_directions"][ti, ri, parent_method, si, t]
                        demand(np.linalg.norm(w) <= 1. + cfg["numeric_tolerance"]
                               and np.min(SIGNS[i] * SIGNS * (d @ w)) >= -cfg["numeric_tolerance"],
                               "Independent same-full-information feasibility and unit-cap comparison")
                        objective_values[at + (mi,)] = c @ w
                        own_help[at + (mi, 0)] = d[i] @ w
                        own_help[at + (mi, 1)] = float(np.mean(SIGNS[i] * SIGNS[helpful] * (d[helpful] @ w))) if helpful.any() else 0.
                    matched[at] = float((corners[at + (1,)] - corners[at + (0,)]).mean())
    demand(np.all(objective_values[..., 1][objective_eligible] >= objective_values[..., 0][objective_eligible] - cfg["numeric_tolerance"]),
           "Recorded balanced full-known objective no worse than feasible selective direction")
    arrays = {"probability_changes": state["correct_probability_delta"], "margin_changes": state["aligned_margin_delta"],
        "effect_categories": category, "phase_before": state["full_known_before"].astype(np.int8), "first_full_trials": first_full,
        "bridge_eligible": eligible, "bridge_q_cells": corners, "bridge_q_shapley": q_shapley,
        "bridge_harm_shapley": harm_shapley, "pair_unknown_membership": membership,
        "objective_eligible": objective_eligible, "objective_values": objective_values,
        "objective_own_help_influences": own_help, "objective_matched_probability_gain": matched,
        "objective_help_counts": help_counts, "pair_action_cases": (a_s + 2 * a_b).astype(np.int8),
        "model_seeds": parent["model_seeds"].copy(), "schedules": parent["schedules"].copy(), "targets": parent["targets"].copy()}
    demand(tuple(arrays) == FIELDS, "Independent nineteen-field saved-path closure")
    return arrays


def expected_rows(cfg, arrays):
    output = []
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            for mi, method in enumerate(METHODS):
                for si, seed in enumerate(cfg["model_seeds"]):
                    at = (ti, ri, mi, si)
                    for phase_index, phase in enumerate(("partial", "full")):
                        trial_mask = arrays["phase_before"][at] == phase_index
                        for ci, category in enumerate(CATEGORIES):
                            mask = trial_mask[:, None] & (arrays["effect_categories"][at] == ci)
                            q, margin = arrays["probability_changes"][at][mask], arrays["margin_changes"][at][mask]
                            row = {"topology": topology, "representation": representation, "method": method, "seed": seed,
                                "phase": phase, "category": category, "trial_count": int(trial_mask.sum()), "exposures": int(mask.sum()),
                                "nonzero_probability_effects": int(np.count_nonzero(q)),
                                "positive_probability_effects": int(np.count_nonzero(q > 0.)),
                                "negative_probability_effects": int(np.count_nonzero(q < 0.)),
                                "probability_positive_mass": float(np.maximum(q, 0.).sum()),
                                "probability_negative_mass": float(np.maximum(-q, 0.).sum()), "probability_signed_sum": float(q.sum()),
                                "margin_positive_mass": float(np.maximum(margin, 0.).sum()),
                                "margin_negative_mass": float(np.maximum(-margin, 0.).sum()), "margin_signed_sum": float(margin.sum())}
                            output.append(row)
    return output


def average(values):
    usable = [value for value in values if value is not None]
    return float(np.mean(usable)) if usable else None


def expected_summary(cfg, arrays, rows, parent):
    seeds, tol = cfg["model_seeds"], cfg["numeric_tolerance"]
    q = arrays["probability_changes"]
    correct = np.where(parent["targets"] == 1, parent["checkpoint_probabilities"], 1. - parent["checkpoint_probabilities"])
    initial, final = correct[..., 0, :].mean(-1), correct[..., -1, :].mean(-1)
    indexed = {}
    for row in rows:
        key = tuple(row[name] for name in ("topology", "representation", "method", "seed", "phase", "category"))
        demand(key not in indexed, "Unique saved-path phase/category measurement")
        indexed[key] = row
    contributions = np.zeros(q.shape[:4] + (2, 3))
    positive, negative = np.zeros_like(contributions), np.zeros_like(contributions)
    families, contrasts = [], []
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            for mi, method in enumerate(METHODS):
                individuals = []
                for si, seed in enumerate(seeds):
                    at = (ti, ri, mi, si)
                    for ph, phase in enumerate(("partial", "full")):
                        for ci, category in enumerate(CATEGORIES):
                            row = indexed[topology, representation, method, seed, phase, category]
                            contributions[at + (ph, ci)] = row["probability_signed_sum"] / 4.
                            positive[at + (ph, ci)] = row["probability_positive_mass"]
                            negative[at + (ph, ci)] = row["probability_negative_mass"]
                    individuals.append({"seed": seed, "initial_expected_accuracy": float(initial[at]),
                        "final_expected_accuracy": float(final[at]), "reconstructed_accuracy_gain": float(contributions[at].sum()),
                        "phase_category_endpoint_contributions": contributions[at].tolist(),
                        "phase_category_positive_probability_mass": positive[at].tolist(),
                        "phase_category_negative_probability_mass": negative[at].tolist(),
                        "first_full_trial": int(arrays["first_full_trials"][at]),
                        "actual_Go_trials": int(parent["trial_actions"][at].sum()),
                        "actual_head_path_l2": float(parent["trial_cumulative_head_path_l2"][at + (-1,)])})
                families.append({"topology": topology, "representation": representation, "method": method, "seeds": individuals,
                    "mean_initial_expected_accuracy": float(initial[ti, ri, mi].mean()),
                    "mean_final_expected_accuracy": float(final[ti, ri, mi].mean()),
                    "mean_phase_category_endpoint_contributions": contributions[ti, ri, mi].mean(axis=0).tolist(),
                    "mean_phase_category_positive_probability_mass": positive[ti, ri, mi].mean(axis=0).tolist(),
                    "mean_phase_category_negative_probability_mass": negative[ti, ri, mi].mean(axis=0).tolist()})
            paired = []
            for si, seed in enumerate(seeds):
                at, selective, balanced = (ti, ri, si), (ti, ri, 2, si), (ti, ri, 3, si)
                difference = q[balanced] - q[selective]
                harm_s, harm_b = np.maximum(-q[selective], 0.), np.maximum(-q[balanced], 0.)
                member = arrays["pair_unknown_membership"][at]
                both, only_s, only_b = member == 1, member == 2, member == 3
                eligible, obj, cases = arrays["bridge_eligible"][at], arrays["objective_eligible"][at], arrays["pair_action_cases"][at]
                objdiff = arrays["objective_values"][at][obj, 1] - arrays["objective_values"][at][obj, 0]
                matched_q = arrays["objective_matched_probability_gain"][at][obj]
                own_help = arrays["objective_own_help_influences"][at][obj]
                full_dot = np.sum(parent["trial_directions"][selective][obj]
                                  * parent["trial_directions"][balanced][obj], axis=-1)
                paired.append({"seed": seed, "endpoint_difference": float(final[balanced] - final[selective]),
                    "phase_category_endpoint_difference": (contributions[balanced] - contributions[selective]).tolist(),
                    "action_case_counts": [int(np.count_nonzero(cases == code)) for code in range(4)],
                    "action_case_endpoint_difference": [float(difference[cases == code].sum() / 4.) for code in range(4)],
                    "both_Go_probability_shapley": (arrays["bridge_q_shapley"][at].sum(axis=(0, 2)) / 4.).tolist(),
                    "both_Go_probability_difference": float(difference[eligible].sum() / 4.),
                    "unknown_damage_difference": float(harm_b[arrays["effect_categories"][balanced] == 2].sum()
                                                       - harm_s[arrays["effect_categories"][selective] == 2].sum()),
                    "unknown_membership_damage_difference": [float((harm_b - harm_s)[both].sum()),
                        -float(harm_s[only_s].sum()), float(harm_b[only_b].sum())],
                    "both_Go_shared_unknown_harm_shapley": (arrays["bridge_harm_shapley"][at] * both[:, None, :]).sum(axis=(0, 2)).tolist(),
                    "both_Go_shared_unknown_harm_difference": float(((harm_b - harm_s) * both * eligible[:, None]).sum()),
                    "unpaired_Go_shared_unknown_harm_difference": float(((harm_b - harm_s) * both * (~eligible)[:, None]).sum()),
                    "full_known_objective_trials": int(obj.sum()), "objective_gain_mean": average(objdiff.tolist()),
                    "objective_improved_trials": int(np.count_nonzero(objdiff > tol)),
                    "objective_matched_probability_gain_mean": average(matched_q.tolist()),
                    "objective_improved_but_probability_not_improved_trials": int(np.count_nonzero((objdiff > tol) & (matched_q <= 0.))),
                    "objective_own_influence_change_mean": average((own_help[:, 1, 0] - own_help[:, 0, 0]).tolist()),
                    "objective_help_influence_change_mean": average((own_help[:, 1, 1] - own_help[:, 0, 1]).tolist()),
                    "full_known_direction_cosine_mean": average(full_dot.tolist()),
                    "unknown_mask_disagreement_destination_count": int(np.count_nonzero(only_s | only_b)),
                    "first_full_trial_difference": int(arrays["first_full_trials"][balanced] - arrays["first_full_trials"][selective])})
            vector_keys = ("phase_category_endpoint_difference", "action_case_counts", "action_case_endpoint_difference",
                "both_Go_probability_shapley", "unknown_membership_damage_difference", "both_Go_shared_unknown_harm_shapley")
            scalar_keys = ("endpoint_difference", "both_Go_probability_difference", "unknown_damage_difference",
                "both_Go_shared_unknown_harm_difference", "unpaired_Go_shared_unknown_harm_difference",
                "full_known_objective_trials", "objective_gain_mean", "objective_improved_trials",
                "objective_matched_probability_gain_mean", "objective_improved_but_probability_not_improved_trials",
                "objective_own_influence_change_mean", "objective_help_influence_change_mean",
                "full_known_direction_cosine_mean",
                "unknown_mask_disagreement_destination_count", "first_full_trial_difference")
            means = {name: np.asarray([record[name] for record in paired]).mean(axis=0).tolist() for name in vector_keys}
            means.update({name: average([record[name] for record in paired]) for name in scalar_keys})
            contrasts.append({"topology": topology, "representation": representation,
                "role": "primary" if (ti, ri) == (0, 0) else "descriptive_control", "seeds": paired, "means": means})
    residuals = {"endpoint_telescoping": float(np.max(np.abs(contributions.sum(axis=(-1, -2)) - (final - initial)))),
        "per_destination_probability_Shapley": float(np.max(np.abs(arrays["bridge_q_shapley"].sum(axis=-2)
                                                   - (arrays["bridge_q_cells"][..., 7, :] - arrays["bridge_q_cells"][..., 0, :])))),
        "per_destination_harm_Shapley": float(np.max(np.abs(arrays["bridge_harm_shapley"].sum(axis=-2)
            - (np.maximum(-arrays["bridge_q_cells"][..., 7, :], 0.) - np.maximum(-arrays["bridge_q_cells"][..., 0, :], 0.))))) }
    records = [record for contrast in contrasts for record in contrast["seeds"]]
    residuals.update({
        "endpoint_phase_category_difference": float(max(abs(record["endpoint_difference"]
            - np.asarray(record["phase_category_endpoint_difference"]).sum()) for record in records)),
        "endpoint_action_case_difference": float(max(abs(record["endpoint_difference"]
            - sum(record["action_case_endpoint_difference"])) for record in records)),
        "unknown_membership_difference": float(max(abs(record["unknown_damage_difference"]
            - sum(record["unknown_membership_damage_difference"])) for record in records)),
        "paired_Go_probability_bridge": float(max(abs(record["both_Go_probability_difference"]
            - sum(record["both_Go_probability_shapley"])) for record in records)),
        "paired_Go_shared_unknown_harm_bridge": float(max(abs(record["both_Go_shared_unknown_harm_difference"]
            - sum(record["both_Go_shared_unknown_harm_shapley"])) for record in records))})
    demand(all(np.isfinite(value) and value <= tol for value in residuals.values()), "Independent exact saved-path accounting gates")
    demand(len(rows) == 24 * len(seeds) * 6 and len(arrays) == 19, "Complete nineteen-field and phase/category closure")
    eligible_count = int(arrays["bridge_eligible"].sum())
    return {"study_id": cfg["study_id"], "status": "completed", "scope": cfg["scope"], "families": families, "contrasts": contrasts,
        "phase_order": ["partial", "full"], "category_order": list(CATEGORIES), "bridge_factor_order": ["state", "scalar", "direction"],
        "action_case_order": ["neither_Go", "selective_only_Go", "balanced_only_Go", "both_Go"],
        "unknown_membership_order": ["both_unknown_difference", "selective_only_negative", "balanced_only_positive"],
        "validation": {"all_exact_accounting_gates_passed": True, "residuals": residuals,
            "last_discovery_trial_assigned_to_partial": True, "zero_effects_and_roundoff_signs_preserved": True,
            "both_Go_only_bridge_no_NoGo_zero_direction_ambiguity": True,
            "full_known_objectives_compare_same_information_constraints": True},
        "counts": {"analyzed_parent_trial_records": int(np.prod(q.shape[:-1])), "analyzed_destination_effects": int(q.size),
            "paired_trial_records": int(arrays["bridge_eligible"].size), "measurement_records": len(rows), "numeric_arrays": len(arrays),
            "bridge_eligible_trials": eligible_count, "bridge_evaluated_corner_destination_effects": eligible_count * 32,
            "full_known_objective_trials": int(arrays["objective_eligible"].sum()), "new_training_trials": 0,
            "new_head_update_calls": 0, "new_internal_update_calls": 0, "new_value_update_calls": 0,
            "new_random_draws": 0, "fresh_model_constructions": 0, "new_optimization_calls": 0},
        "interpretation": {"retrospective_descriptive_only": True, "parent_H_gain": False, "parent_H_acquisition": False,
            "parent_H_unknown": False, "new_confirmatory_hypothesis": False, "exact_path_accounting_not_unique_causal_explanation": True,
            "crossed_one_step_cells_are_not_relearned_policy": True, "crossed_directions_can_have_different_original_information": True,
            "unknown_harm_bridge_intersection_mask_fixed_across_cells": True, "same_realized_movement_budget_claimed": False,
            "internal_synapses_and_probe_activity_fixed": True, "observed_memory_is_separate_from_output_weights": True,
            "novel_input_generalization_tested": False, "new_human_experiment": False,
            "shared_neural_mechanism_claim": False, "education_effect_claim": False}}


def check_artifacts(arrays, wanted, rows, wanted_rows):
    demand(set(arrays) == set(wanted) == set(FIELDS), "Independent nineteen-array closure")
    for key, value in wanted.items(): close(arrays[key], value, "Independent saved-path artifact: " + key)
    demand(len(rows) == len(wanted_rows), "Complete phase/category CSV")
    for actual, expected in zip(rows, wanted_rows):
        demand(tuple(actual) == tuple(expected), "Exact saved-path CSV column closure/order")
        for key, value in expected.items():
            if type(value) is str: demand(actual[key] == value, "CSV identity: " + key)
            else:
                number = float(actual[key])
                demand(np.isfinite(number) and (number == value if type(value) is int
                    else np.isclose(number, value, atol=1e-9, rtol=1e-9)), "CSV exact counts/arithmetic: " + key)


def verify_science(cfg, parent, arrays, rows, summary, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    wanted = expected_artifacts(cfg, parent)
    wanted_rows = expected_rows(cfg, wanted)
    check_artifacts(arrays, wanted, rows, wanted_rows)
    expected = expected_summary(cfg, wanted, wanted_rows, parent)
    compare_tree(summary, expected, "Independent complete saved-path descriptive claims")
    count = int(np.prod(parent["trial_actions"].shape))
    return {"arrays": len(wanted), "parent_trial_records_checked": count,
        "destination_effects_checked": count * 4, "measurement_records_checked": len(wanted_rows),
        "all_fixed_saved_head_prefixes_feedback_and_checkpoints_checked": True,
        "phase_uses_prefeedback_memory_and_last_discovery_is_partial": True,
        "roundoff_signs_and_zero_effects_preserved_without_threshold": True,
        "all_destination_and_phase_endpoint_contributions_telescope": True,
        "all_eight_both_Go_corners_independently_reconstructed": True,
        "state_scalar_direction_Shapley_efficiency_checked": True,
        "unknown_mask_intersection_and_exposure_difference_separated": True,
        "full_known_objective_and_matched_exact_probability_separated": True,
        "prior_full_autonomous_replay_reused_by_frozen_hashes": True,
        "no_rng_learner_graph_or_optimizer_calls": True,
        "new_training_trials": 0, "new_random_draws": 0, "new_optimization_calls": 0}


def load_rows(path):
    with Path(path).open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        demand(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames), "Unique CSV columns")
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
    from drosolearn.pathways_envelope import verify_envelope, sha, write_new
    cfg, _, metadata = verify_envelope(args.root, args.config, args.registration, args.result_dir, args.publication_proof)
    with np.load(args.root / cfg["parent_result"] / "arrays.npz", allow_pickle=False) as handle:
        parent = {key: handle[key] for key in handle.files}
    with np.load(args.result_dir / "arrays.npz", allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    rows = load_rows(args.result_dir / "measurements.csv")
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf8"))
    checks = verify_science(cfg, parent, arrays, rows, summary)
    demand(metadata["counts"] == summary["counts"], "Metadata and independently verified summary counts")
    report = {"status": "passed", "study_id": cfg["study_id"], "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_sha256": sha(args.registration), "summary_sha256": sha(args.result_dir / "summary.json"),
        "scientific_producer_imported": False, "new_helper_imported": False, "counts": checks}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
