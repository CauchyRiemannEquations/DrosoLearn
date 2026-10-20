"""Independent post-gate state/scalar/direction saved-path accounting.

The new producer and helpers are never imported. Crossed cells are arithmetic
on recorded states and directions; no learner, graph, RNG, solver or future
policy is executed. Only common-post, both-actual-Go records are crossed.
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
METHODS = ("shared", "isolated", "feedback_selective", "balanced", "gated_balanced")
CONTROLS = ("isolated", "balanced", "feedback_selective", "shared")
FACTORS = ("state", "scalar", "direction")
SIGNS = np.asarray([1., 1., -1., -1.])
GROUPS = ("own", "other")
COMPONENTS = FACTORS + ("control_only_Go", "gated_only_Go", "neither_Go")
DIAGNOSTICS = ("aligned_source_logit_before", "correct_source_probability", "logistic_sensitivity_value",
    "abs_update_factor", "head_step_l2", "own_unit_direction_influence", "other_unit_aligned_influence_mean", "direction_l2")
FIELDS = ("probability_changes", "margin_changes", "reference_post", "pair_action_cases", "bridge_eligible", "bridge_q_cells",
    "bridge_q_shapley", "bridge_positive_shapley", "bridge_negative_shapley", "factual_pair_diagnostics", "first_post_trials",
    "model_seeds", "schedules", "targets")


def demand(condition, label):
    if not condition: raise ValueError(label)


def close(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    demand(actual.shape == expected.shape and actual.dtype.kind in "biuf" and np.isfinite(actual).all(), label + " finite axes")
    if expected.dtype.kind in "biu":
        demand(actual.dtype.kind in "biu" and np.array_equal(actual, expected), label + " exact")
    else: demand(np.allclose(actual, expected, atol=1e-9, rtol=1e-9), label + " arithmetic")


def compare_tree(actual, expected, label="summary"):
    if isinstance(expected, dict):
        demand(isinstance(actual, dict) and actual.keys() == expected.keys(), label + " keys")
        for key in expected: compare_tree(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, list):
        demand(isinstance(actual, list) and len(actual) == len(expected), label + " length")
        for i, value in enumerate(expected): compare_tree(actual[i], value, label + "." + str(i))
    elif type(expected) is bool: demand(type(actual) is bool and actual == expected, label + " flag")
    elif type(expected) is int: demand(type(actual) is int and actual == expected, label + " integer")
    elif type(expected) is float:
        demand(type(actual) in (float, int) and np.isfinite(actual) and np.isclose(actual, expected, atol=1e-9, rtol=1e-9), label + " numeric")
    else: demand(actual == expected, label + " value")


def sigmoid(values):
    return 1. / (1. + np.exp(-np.clip(values, -40., 40.)))


def independent_bridge(z_control, z_gated, g_control, g_gated, f_control, f_gated):
    zc, zg, gc, gg, fc, fg = map(np.asarray, (z_control, z_gated, g_control, g_gated, f_control, f_gated))
    demand(zc.shape == zg.shape == fc.shape == fg.shape == gc.shape + (4,) and gc.shape == gg.shape
           and all(value.dtype.kind in "fiu" and np.isfinite(value).all() for value in (zc, zg, gc, gg, fc, fg)), "Recorded eight-corner finite axes")
    cells = np.empty(gc.shape + (8, 4))
    for code in range(8):
        z, g, f = (zg if code & 4 else zc), (gg if code & 2 else gc), (fg if code & 1 else fc)
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


def validate_science(cfg, allow_fixture=False):
    fixed = {"study_id": "gated_factors_v1", "scope": "retrospective_descriptive_gated_factors",
        "module": "drosolearn.gated_factors", "protocol": "docs/gated-factors-protocol.md",
        "parent_result": "results/gated_pathways_v1", "learning_result": "results/gated_learning_v1",
        "baseline_tag": "gated-pathways-v1", "baseline_commit": "e74604a67479299825e2ab5572fd3efd8f9970d6",
        "topologies": list(TOPS), "representations": list(REPS), "methods": list(METHODS), "contrast_controls": list(CONTROLS),
        "model_seeds": list(range(17601, 17611)), "training_trials": 400, "numeric_tolerance": 1e-8,
        "phase_definition": "reference_gated_known_before_count_equals_four", "bridge_eligibility": "reference_post_and_both_actual_Go",
        "bridge_bits": list(FACTORS), "bridge_allocation": "three_factor_symmetric_Shapley_exact_sigmoid",
        "destination_groups": list(GROUPS), "new_confirmatory_hypothesis": False, "external_manifest": None}
    demand(isinstance(cfg, dict), "Independent post-gate factor configuration")
    fixture = cfg.get("scope") == "integration_test"
    demand(not fixture or allow_fixture, "Invented fixture cannot enter actual factor analysis")
    for key, value in fixed.items():
        if fixture and key in ("scope", "model_seeds", "training_trials"): continue
        demand(cfg.get(key) == value, "Fixed independent factor choice: " + key)
    if fixture:
        seeds, n = cfg.get("model_seeds"), cfg.get("training_trials")
        demand(type(seeds) is list and 1 <= len(seeds) <= 2 and len(set(seeds)) == len(seeds)
               and all(type(seed) is int and 90001 <= seed <= 90100 for seed in seeds), "Invented saved-factor seeds only")
        demand(type(n) is int and 4 <= n <= 16, "Bounded invented saved-factor length")


def factual_states(cfg, parent, accounting):
    s, n = len(cfg["model_seeds"]), cfg["training_trials"]
    shape = (3, 2, 5, s, n)
    close(parent["model_seeds"], np.asarray(cfg["model_seeds"]), "Original frozen cohort seeds")
    close(parent["targets"], (SIGNS > 0).astype(int), "Original four reward target identities")
    source = parent["schedules"]
    demand(source.shape == (s, n) and source.dtype.kind in "iu" and np.isin(source, range(4)).all(), "Original current-input schedules")
    actions = parent["trial_actions"]
    demand(actions.shape == shape and actions.dtype.kind in "biu" and np.isin(actions, (0, 1)).all(), "Original actual action histories")
    before = parent["trial_known_before"]
    demand(before.shape == shape + (4,) and before.dtype.kind in "biu" and np.isin(before, (-1, 0, 1)).all(), "Actual observations before feedback")
    post = np.all(before[:, :, 4] != 0, axis=-1)
    close(accounting["gated_phase_before"], post, "Prior common-post accounting phase")
    q, margin, logits, influence = (np.empty(shape + (4,)) for _ in range(4))
    residual = 0.
    grid = parent["checkpoints"].tolist()
    demand(grid == sorted(set(grid)) and grid[0] == 0 and grid[-1] == n, "Frozen checkpoint grid")
    g, w = parent["trial_update_factors"], parent["trial_directions"]
    demand(g.shape == shape and w.shape[:-1] == shape and np.isfinite(g).all() and np.isfinite(w).all(), "Frozen update scalar/direction axes")
    for ti, top in enumerate(TOPS):
        for ri, rep in enumerate(REPS):
            features = parent["g_" + top + "_" + rep + "_unit_rows"]
            demand(features.shape == (s, 4, w.shape[-1]) and np.isfinite(features).all(), "Frozen common decision-feature axes")
            close(np.linalg.norm(features, axis=-1), np.ones((s, 4)), "Original unit features")
            for si in range(s):
                d = features[si]
                for mi in range(5):
                    at = (ti, ri, mi, si)
                    steps = g[at][:, None] * w[at]
                    heads = np.concatenate((np.zeros((1, d.shape[1])), np.cumsum(steps, axis=0)))
                    logits[at] = heads[:-1] @ d.T
                    influence[at] = w[at] @ d.T
                    delta = g[at][:, None] * influence[at]
                    q[at] = SIGNS * (sigmoid(logits[at] + delta) - sigmoid(logits[at]))
                    margin[at] = SIGNS * delta
                    for actual, expected in ((parent["trial_correct_probability_delta"][at], q[at]),
                        (parent["trial_aligned_margin_delta"][at], margin[at]),
                        (parent["checkpoint_heads"][at], heads[grid]),
                        (parent["checkpoint_probabilities"][at], sigmoid(heads[grid] @ d.T))):
                        close(actual, expected, "Frozen parent head/effect reconstruction")
                        residual = max(residual, float(np.max(np.abs(actual - expected))))
    for name, expected in (("probability_changes", q), ("margin_changes", margin)):
        close(accounting[name], expected, "Prior saved-path factual accounting: " + name)
        residual = max(residual, float(np.max(np.abs(accounting[name] - expected))))
    return {"q": q, "margin": margin, "logits": logits, "influence": influence, "post": post,
            "residual": residual, "full_before": np.all(before != 0, axis=-1)}


def expected_artifacts(cfg, parent, accounting):
    state = factual_states(cfg, parent, accounting)
    s, n = len(cfg["model_seeds"]), cfg["training_trials"]
    pair_shape = (3, 2, 4, s, n)
    case, eligible = np.empty(pair_shape, np.int8), np.empty(pair_shape, bool)
    cells = np.zeros(pair_shape + (8, 4))
    diagnostics = np.zeros(pair_shape + (2, 8))
    for ti in range(3):
        for ri in range(2):
            for ci, control in enumerate(CONTROLS):
                mi = METHODS.index(control)
                for si, order in enumerate(parent["schedules"]):
                    at, c, g = (ti, ri, ci, si), (ti, ri, mi, si), (ti, ri, 4, si)
                    case[at] = parent["trial_actions"][c] + 2 * parent["trial_actions"][g]
                    mask = state["post"][ti, ri, si] & (case[at] == 3)
                    eligible[at] = mask
                    cells[at][mask] = independent_bridge(state["logits"][c][mask], state["logits"][g][mask],
                        parent["trial_update_factors"][c][mask], parent["trial_update_factors"][g][mask],
                        state["influence"][c][mask], state["influence"][g][mask])
                    close(cells[at][mask, 0], state["q"][c][mask], "Control factual endpoint of crossed cells")
                    close(cells[at][mask, 7], state["q"][g][mask], "Gated factual endpoint of crossed cells")
                    source = order[mask]
                    for pair_index, factual in enumerate((c, g)):
                        z = state["logits"][factual][mask]
                        f = state["influence"][factual][mask]
                        factors = parent["trial_update_factors"][factual][mask]
                        norms = np.linalg.norm(parent["trial_directions"][factual][mask], axis=-1)
                        own = np.eye(4, dtype=bool)[source]
                        p_go = sigmoid(z[np.arange(len(source)), source])
                        aligned = SIGNS[source, None] * SIGNS * f
                        rows = np.column_stack((SIGNS[source] * z[np.arange(len(source)), source],
                            np.where(SIGNS[source] == 1, p_go, 1. - p_go), p_go * (1. - p_go),
                            np.abs(factors), np.abs(factors) * norms, f[np.arange(len(source)), source],
                            np.where(own, 0., aligned).sum(axis=-1) / 3., norms))
                        diagnostics[at][mask, pair_index] = rows
    first_post = np.where(state["post"].any(axis=-1), state["post"].argmax(axis=-1) + 1, n + 1).astype(np.int32)
    close(parent["first_balanced_mode_trials"][:, :, 4], first_post, "Frozen reference first actual balanced mode")
    signed, positive, negative = (np.zeros(pair_shape + (3, 4)) for _ in range(3))
    signed[eligible] = independent_shapley(cells[eligible])
    positive[eligible] = independent_shapley(np.maximum(cells[eligible], 0.))
    negative[eligible] = independent_shapley(np.maximum(-cells[eligible], 0.))
    result = {"probability_changes": state["q"], "margin_changes": state["margin"], "reference_post": state["post"],
        "pair_action_cases": case, "bridge_eligible": eligible, "bridge_q_cells": cells,
        "bridge_q_shapley": signed, "bridge_positive_shapley": positive,
        "bridge_negative_shapley": negative, "factual_pair_diagnostics": diagnostics,
        "first_post_trials": first_post, "model_seeds": parent["model_seeds"].copy(),
        "schedules": parent["schedules"].copy(), "targets": parent["targets"].copy()}
    demand(tuple(result) == FIELDS, "Independent fourteen-array factor output closure")
    return result


def expected_rows(cfg, arrays):
    rows = []
    for ti, top in enumerate(TOPS):
        for ri, rep in enumerate(REPS):
            for ci, control in enumerate(CONTROLS):
                mi = METHODS.index(control)
                for si, seed in enumerate(cfg["model_seeds"]):
                    at = (ti, ri, ci, si)
                    c, g = (ti, ri, mi, si), (ti, ri, 4, si)
                    post, eligible = arrays["reference_post"][ti, ri, si], arrays["bridge_eligible"][at]
                    case = arrays["pair_action_cases"][at]
                    own = np.eye(4, dtype=bool)[arrays["schedules"][si]]
                    for gi, group in enumerate(GROUPS):
                        destination = own if gi == 0 else ~own
                        for ki, component in enumerate(COMPONENTS):
                            if ki < 3:
                                signed = float(arrays["bridge_q_shapley"][at][:, ki][destination].sum())
                                positive = float(arrays["bridge_positive_shapley"][at][:, ki][destination].sum())
                                negative = float(arrays["bridge_negative_shapley"][at][:, ki][destination].sum())
                                count = int(eligible.sum())
                            else:
                                code = (1, 2, 0)[ki - 3]
                                selected = post & (case == code)
                                count = int(selected.sum())
                                value = arrays["probability_changes"][c if code == 1 else g]
                                sign = -1. if code == 1 else 1. if code == 2 else 0.
                                mask = selected[:, None] & destination
                                signed = float(sign * value[mask].sum())
                                positive = float(sign * np.maximum(value[mask], 0.).sum())
                                negative = float(sign * np.maximum(-value[mask], 0.).sum())
                            rows.append({"topology": top, "representation": rep, "control": control, "seed": seed,
                                "destination_group": group, "component": component, "reference_post_trials": int(post.sum()),
                                "both_Go_trials": int(eligible.sum()), "component_trials": count, "probability_signed_sum": signed,
                                "positive_mass_allocation": positive, "negative_mass_allocation": negative,
                                "endpoint_contribution": signed / 4., "positive_endpoint_allocation": positive / 4.,
                                "negative_endpoint_allocation": negative / 4.})
    return rows


def mean_nested(values):
    if isinstance(values[0], list): return [mean_nested(list(column)) for column in zip(*values)]
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def expected_summary(cfg, arrays, rows, parent, accounting):
    state = factual_states(cfg, parent, accounting)
    q, post = arrays["probability_changes"], arrays["reference_post"]
    indexed = {}
    for row in rows:
        key = tuple(row[name] for name in ("topology", "representation", "control", "seed", "destination_group", "component"))
        demand(key not in indexed, "Unique factor component measurement")
        indexed[key] = row
    families, group_residual, component_residual = [], 0., 0.
    for ti, top in enumerate(TOPS):
        for ri, rep in enumerate(REPS):
            for ci, control in enumerate(CONTROLS):
                mi, records = METHODS.index(control), []
                for si, seed in enumerate(cfg["model_seeds"]):
                    at, c, g = (ti, ri, ci, si), (ti, ri, mi, si), (ti, ri, 4, si)
                    selected = post[ti, ri, si]
                    own = np.eye(4, dtype=bool)[arrays["schedules"][si]]
                    components, positive, negative, factual, factual_pos, factual_neg = [], [], [], [], [], []
                    for group, destination in zip(GROUPS, (own, ~own)):
                        component_rows = [indexed[top, rep, control, seed, group, name] for name in COMPONENTS]
                        components.append([row["endpoint_contribution"] for row in component_rows])
                        positive.append([row["positive_endpoint_allocation"] for row in component_rows])
                        negative.append([row["negative_endpoint_allocation"] for row in component_rows])
                        mask = selected[:, None] & destination
                        factual.append(float((q[g] - q[c])[mask].sum() / 4.))
                        factual_pos.append(float((np.maximum(q[g], 0.) - np.maximum(q[c], 0.))[mask].sum() / 4.))
                        factual_neg.append(float((np.maximum(-q[g], 0.) - np.maximum(-q[c], 0.))[mask].sum() / 4.))
                        old = float((accounting["probability_changes"][g] - accounting["probability_changes"][c])[mask].sum() / 4.)
                        group_residual = max(group_residual, abs(factual[-1] - old))
                    components, positive, negative = map(np.asarray, (components, positive, negative))
                    component_residual = max(component_residual, float(np.abs(components.sum(axis=-1) - factual).max()),
                        float(np.abs(positive.sum(axis=-1) - factual_pos).max()), float(np.abs(negative.sum(axis=-1) - factual_neg).max()))
                    eligible = arrays["bridge_eligible"][at]
                    diag = arrays["factual_pair_diagnostics"][at][eligible]
                    all_observed = state["full_before"][c] & state["full_before"][g]
                    records.append({"seed": seed, "reference_post_trials": int(selected.sum()), "both_Go_trials": int(eligible.sum()),
                        "case_counts": [int(np.count_nonzero(selected & (arrays["pair_action_cases"][at] == case))) for case in range(4)],
                        "component_endpoint_contributions": components.tolist(), "component_positive_endpoint_allocations": positive.tolist(),
                        "component_negative_endpoint_allocations": negative.tolist(), "factual_post_endpoint_difference": factual,
                        "factual_post_positive_endpoint_difference": factual_pos, "factual_post_negative_endpoint_difference": factual_neg,
                        "factual_post_total_difference": float(sum(factual)),
                        "diagnostic_means": diag.mean(axis=0).tolist() if len(diag) else [[None] * 8 for _ in range(2)],
                        "both_methods_full_before_on_eligible_trials": int(np.count_nonzero(eligible & all_observed)),
                        "control_partial_before_on_eligible_trials": int(np.count_nonzero(eligible & ~state["full_before"][c]))})
                means = {name: mean_nested([record[name] for record in records]) for name in records[0] if name != "seed"}
                families.append({"topology": top, "representation": rep, "control": control,
                    "role": "primary" if (ti, ri, ci) == (0, 0, 0) else "descriptive_control", "seeds": records, "means": means})
    cells = arrays["bridge_q_cells"]
    plus, minus, eligible = np.maximum(cells, 0.), np.maximum(-cells, 0.), arrays["bridge_eligible"]
    residuals = {"factual_parent_effects": state["residual"], "post_group_endpoint_reconstruction": float(group_residual),
        "shapley_signed_efficiency": float(np.abs(arrays["bridge_q_shapley"].sum(axis=-2) - (cells[..., 7, :] - cells[..., 0, :])).max()),
        "shapley_positive_efficiency": float(np.abs(arrays["bridge_positive_shapley"].sum(axis=-2) - (plus[..., 7, :] - plus[..., 0, :])).max()),
        "shapley_negative_efficiency": float(np.abs(arrays["bridge_negative_shapley"].sum(axis=-2) - (minus[..., 7, :] - minus[..., 0, :])).max()),
        "positive_negative_allocation_identity": float(np.abs(arrays["bridge_q_shapley"] - arrays["bridge_positive_shapley"] + arrays["bridge_negative_shapley"]).max()),
        "post_component_reconstruction": float(component_residual),
        "zero_padded_ineligible": float(max(np.max(np.abs(arrays[name][~eligible]), initial=0.) for name in (
            "bridge_q_cells", "bridge_q_shapley", "bridge_positive_shapley", "bridge_negative_shapley", "factual_pair_diagnostics")))}
    demand(all(np.isfinite(value) and value <= cfg["numeric_tolerance"] for value in residuals.values()), "Independent exact factor accounting gates")
    demand(len(arrays) == 14 and len(rows) == 6 * 4 * len(cfg["model_seeds"]) * 2 * 6, "Complete factor output closure")
    return {"study_id": cfg["study_id"], "status": "completed", "scope": cfg["scope"], "families": families,
        "group_order": list(GROUPS), "component_order": list(COMPONENTS), "diagnostic_order": list(DIAGNOSTICS), "control_order": list(CONTROLS),
        "validation": {"all_exact_accounting_gates_passed": True, "residuals": residuals,
            "common_post_uses_prefeedback_memory": True, "only_actual_both_Go_crossed": True, "own_other_partition": True,
            "all_action_cases_retained": True, "tiny_signs_preserved": True, "same_recorded_fact_corners_match": True, "no_future_policy_or_new_training": True},
        "counts": {"analyzed_parent_trial_records": int(np.prod(q.shape[:-1])), "analyzed_destination_effects": int(q.size),
            "paired_trial_records": int(eligible.size), "reference_gate_paths": int(np.prod(post.shape[:-1])),
            "comparison_paths": int(np.prod(eligible.shape[:-1])), "measurement_records": len(rows), "numeric_arrays": len(arrays),
            "paired_post_trial_records": int(post.sum() * 4), "eligible_both_Go_trials": int(eligible.sum()),
            "corner_destination_effects": int(eligible.sum() * 32), "new_training_trials": 0, "new_head_update_calls": 0,
            "new_internal_update_calls": 0, "new_value_update_calls": 0, "new_random_draws": 0, "fresh_model_constructions": 0, "new_optimization_calls": 0},
        "interpretation": {"retrospective_descriptive_only": True, "parent_H_gain": False, "parent_H_acquisition": False,
            "parent_H_protection": True, "new_confirmatory_hypothesis": False, "accounting_allocation_not_unique_causal_explanation": True,
            "mixed_one_step_cells_not_relearned_policy": True, "unpaired_actions_not_assigned_to_zero_direction": True,
            "same_realized_movement_budget_claimed": False, "internal_synapses_and_probe_activity_fixed": True,
            "observed_memory_is_separate_from_output_weights": True, "novel_input_generalization_tested": False,
            "new_human_experiment": False, "shared_neural_mechanism_claim": False, "education_effect_claim": False}}


def check_artifacts(arrays, wanted, rows, wanted_rows):
    demand(set(arrays) == set(wanted) == set(FIELDS), "Independent fourteen-array closure")
    for name, value in wanted.items(): close(arrays[name], value, "Independent factor array: " + name)
    demand(len(rows) == len(wanted_rows), "Complete ordered factor measurement rows")
    for actual, expected in zip(rows, wanted_rows):
        demand(tuple(actual) == tuple(expected), "Exact factor measurement columns")
        for name, value in expected.items():
            if type(value) is str: demand(actual[name] == value, "Factor row identity: " + name)
            else:
                number = float(actual[name])
                demand(np.isfinite(number) and (number == value if type(value) is int else np.isclose(number, value, atol=1e-9, rtol=1e-9)),
                       "Factor row exact count or arithmetic: " + name)


def verify_science(cfg, parent, accounting, arrays, rows, summary, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    wanted = expected_artifacts(cfg, parent, accounting)
    wanted_rows = expected_rows(cfg, wanted)
    check_artifacts(arrays, wanted, rows, wanted_rows)
    expected = expected_summary(cfg, wanted, wanted_rows, parent, accounting)
    compare_tree(summary, expected, "Independent complete post-gate factor summary")
    trials = int(np.prod(parent["trial_actions"].shape))
    return {"arrays": len(wanted), "parent_trial_records_checked": trials, "destination_effects_checked": trials * 4,
        "measurement_records_checked": len(wanted_rows), "factual_head_prefixes_and_prior_accounting_checked": True,
        "only_post_both_actual_Go_eight_corners_checked": True, "state_scalar_direction_symmetric_allocations_checked": True,
        "signed_positive_and_negative_mass_allocations_checked": True,
        "own_other_and_all_unpaired_actual_actions_accounted": True,
        "factual_diagnostics_and_nullable_seed_means_checked": True,
        "partial_control_information_not_silently_equalized": True,
        "no_rng_model_graph_solver_or_future_policy_calls": True,
        "new_training_trials": 0, "new_random_draws": 0, "new_optimization_calls": 0}


def load_rows(path):
    with Path(path).open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        demand(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames), "Unique factor CSV fields")
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
    from drosolearn.gated_factors_envelope import verify_envelope, sha, write_new
    cfg, _, metadata = verify_envelope(args.root, args.config, args.registration, args.result_dir, args.publication_proof)
    inputs = []
    for source in (cfg["learning_result"], cfg["parent_result"]):
        with np.load(args.root / source / "arrays.npz", allow_pickle=False) as handle:
            inputs.append({name: handle[name] for name in handle.files})
    with np.load(args.result_dir / "arrays.npz", allow_pickle=False) as handle:
        arrays = {name: handle[name] for name in handle.files}
    rows = load_rows(args.result_dir / "measurements.csv")
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf8"))
    checks = verify_science(cfg, *inputs, arrays, rows, summary)
    demand(metadata["counts"] == summary["counts"], "Metadata and independently checked factor counts")
    report = {"status": "passed", "study_id": cfg["study_id"], "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_sha256": sha(args.registration), "summary_sha256": sha(args.result_dir / "summary.json"),
        "scientific_producer_imported": False, "new_helper_imported": False, "counts": checks}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__": main()
