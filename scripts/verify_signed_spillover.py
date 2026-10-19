"""Independent arithmetic audit of saved signed spillover; no training or RNG.

Neither the new scientific producer nor its mathematical helper is imported.
The parent arrays, recorded actions and obtained rewards are sufficient inputs.
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
ALPHAS = (0., .25, .5, .75, 1.)
SIGNS = np.asarray([1., 1., -1., -1.])
CATEGORY_NAMES = ("self", "cross_same", "cross_opposite", "cross_all", "all")
RESIDUAL_NAMES = ("static_M", "trial_g", "trial_p", "trial_own_score", "trial_other_score_l2",
                  "trial_head_step_l2", "checkpoint_head", "checkpoint_logits", "checkpoint_probabilities",
                  "margin_telescoping", "probability_telescoping")
NUMERIC_FIELDS = ("static_cells", "static_helpful_cells", "static_harmful_cells", "static_neutral_cells",
                  "static_helpful_mass", "static_harmful_mass", "static_net_mass", "actual_cells",
                  "margin_helpful_cells", "margin_harmful_cells", "margin_neutral_cells",
                  "margin_helpful_sum", "margin_harmful_sum", "margin_net_sum",
                  "probability_helpful_cells", "probability_harmful_cells", "probability_neutral_cells",
                  "probability_helpful_sum", "probability_harmful_sum", "probability_net_sum")


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, atol=1e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    demand(actual.shape == expected.shape, label + " shape")
    demand(actual.dtype.kind in "biuf" and np.isfinite(actual).all(), label + " finite")
    if expected.dtype.kind in "biu":
        demand(actual.dtype.kind in "biu" and np.array_equal(actual, expected), label + " exact")
    else:
        demand(np.allclose(actual, expected, atol=atol, rtol=1e-9), label + " arithmetic")


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -40., 40.)))


def category_masks():
    own = np.eye(4, dtype=bool)
    same = SIGNS[:, None] == SIGNS[None, :]
    return np.stack((own, same & ~own, ~same, ~own, np.ones((4, 4), bool)))


def compare_tree(actual, expected, label="summary"):
    if isinstance(expected, dict):
        demand(isinstance(actual, dict) and actual.keys() == expected.keys(), label + " keys")
        for key in expected:
            compare_tree(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, list):
        demand(isinstance(actual, list) and len(actual) == len(expected), label + " length")
        for i, value in enumerate(expected):
            compare_tree(actual[i], value, label + "." + str(i))
    elif isinstance(expected, bool):
        demand(type(actual) is bool and actual == expected, label + " flag")
    elif isinstance(expected, int):
        demand(type(actual) is int and actual == expected, label + " integer")
    elif isinstance(expected, float):
        demand(type(actual) in (float, int) and np.isfinite(actual)
               and np.isclose(actual, expected, atol=1e-9, rtol=1e-9), label + " numeric")
    else:
        demand(actual == expected, label + " value")


def saved_parent_state(cfg, arrays, rows):
    """Check every recorded trial and checkpoint using stored values only.

    Scalar factors and head increments are deterministic checks of the saved
    observations; no action is sampled, no graph is regenerated, and no learner
    is trained. Four destination-score changes are reconstructed for each row.
    """
    seeds = cfg["model_seeds"]
    n, grid = cfg["training_trials"], cfg["checkpoints"]
    eta = cfg["head_lr"]
    expected_shape = (3, 2, 5, len(seeds), n)
    for key in ("trial_actions", "trial_rewards", "trial_probabilities", "trial_update_factors"):
        demand(arrays[key].shape == expected_shape, "Saved parent axis: " + key)
        demand(np.isfinite(arrays[key]).all(), "Saved parent finite: " + key)
    close(arrays["model_seeds"], np.asarray(seeds), "Parent seed identity")
    close(arrays["checkpoints"], np.asarray(grid), "Parent checkpoint identity")
    close(arrays["alphas"], np.asarray(ALPHAS), "Parent alpha identity")
    close(arrays["targets"], (SIGNS > 0).astype(int), "Parent target identity")
    demand(arrays["schedules"].shape == (len(seeds), n), "Parent schedule axes")
    demand(arrays["schedules"].dtype.kind in "iu" and np.all((arrays["schedules"] >= 0)
           & (arrays["schedules"] < 4)), "Saved parent schedule range")
    demand(all(np.array_equal(np.bincount(order, minlength=4), np.full(4, n//4))
               for order in arrays["schedules"]), "Balanced saved schedules")
    for field in ("internal_update_calls", "value_update_calls", "internal_steps", "value_steps"):
        demand(not np.any(arrays["trial_" + field]), "No saved internal/value changes")
    demand(len(rows) == int(np.prod(expected_shape)), "Complete saved parent row count")
    lookup = {}
    for row in rows:
        key = (row["topology"], row["representation"], float(row["alpha"]), int(row["seed"]), int(row["trial"]))
        demand(key[0] in TOPOLOGIES and key[1] in REPRESENTATIONS and key[2] in ALPHAS
               and key[3] in seeds and 1 <= key[4] <= n, "Saved trial identity in complete fixed axes")
        demand(key not in lookup, "Unique saved trial row")
        lookup[key] = row
    influences = np.empty((3, 2, 5, len(seeds), 4, 4))
    directions = {}
    unit_rows = {}
    margin_delta = np.empty(expected_shape + (4,))
    margin_before = np.empty_like(margin_delta)
    probability_delta = np.empty_like(margin_delta)
    reconstruction_residual = 0.
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            prefix = "t_" + topology + "_" + representation + "_"
            d_all = arrays[prefix + "base_budget_base_geometry_unit_rows"]
            w_all = arrays[prefix + "directions"]
            demand(d_all.ndim == 3 and d_all.shape[:2] == (len(seeds), 4), "Saved feature axes")
            demand(w_all.shape == (len(seeds), 5, 4, d_all.shape[-1]), "Saved direction axes")
            demand(np.isfinite(d_all).all() and np.isfinite(w_all).all(), "Finite saved geometry")
            close(np.linalg.norm(d_all, axis=-1), np.ones((len(seeds), 4)), "Unit saved decision rows")
            close(np.linalg.norm(w_all, axis=-1), np.ones((len(seeds), 5, 4)), "Unit saved update directions")
            directions[ti, ri], unit_rows[ti, ri] = w_all, d_all
            for si, seed in enumerate(seeds):
                d = d_all[si]
                for ai, alpha in enumerate(ALPHAS):
                    w = w_all[si, ai]
                    influence = w @ d.T
                    influences[ti, ri, ai, si] = influence
                    close(arrays[prefix + "influences"][si, ai], influence, "Saved W D transpose")
                    head = arrays["checkpoint_heads"][ti, ri, ai, si, 0].copy()
                    close(head, np.zeros_like(head), "Initial saved head zero")
                    path = 0.
                    square = 0.
                    def check_checkpoint(index):
                        saved_head = arrays["checkpoint_heads"][ti, ri, ai, si, index]
                        scores = d @ head
                        close(saved_head, head, "Saved checkpoint head from recorded increments")
                        close(arrays["checkpoint_logits"][ti, ri, ai, si, index], scores,
                              "Saved checkpoint scores from head")
                        close(arrays["checkpoint_probabilities"][ti, ri, ai, si, index], sigmoid(scores),
                              "Saved checkpoint probabilities from scores")
                    check_checkpoint(0)
                    for trial in range(n):
                        source = int(arrays["schedules"][si, trial])
                        at = (ti, ri, ai, si, trial)
                        scores = d @ head
                        margins = SIGNS * scores
                        probability = float(sigmoid(scores[source]))
                        action = int(arrays["trial_actions"][at])
                        reward = int(arrays["trial_rewards"][at])
                        demand(action in (0, 1) and arrays["trial_actions"][at] == action, "Saved binary action")
                        demand(arrays["trial_rewards"][at] == reward and reward == action * int(SIGNS[source]),
                               "Obtained reward only, including NoGo zero")
                        factor = eta * (action - probability) * reward
                        close(arrays["trial_probabilities"][at], np.asarray(probability), "Saved current-source probability")
                        close(arrays["trial_update_factors"][at], np.asarray(factor), "Saved observed scalar update")
                        c = SIGNS * factor * influence[source]
                        margin_delta[at], margin_before[at] = c, margins
                        probability_delta[at] = sigmoid(margins + c) - sigmoid(margins)
                        new_head = head + factor * w[source]
                        step_vector = new_head - head
                        step = float(np.linalg.norm(step_vector))
                        path += step
                        square += step * step
                        reconstruction_residual = max(reconstruction_residual, float(np.max(np.abs(
                            SIGNS * (d @ new_head - scores) - c))))
                        row = lookup[(topology, representation, alpha, seed, trial + 1)]
                        expected_row = {"example_index": source, "target": int(SIGNS[source] > 0),
                                        "probability_go": probability, "action": action, "reward": reward,
                                        "update_factor": factor, "head_step_l2": step,
                                        "own_score_step": float((d @ step_vector)[source]),
                                        "other_score_step_l2": float(np.linalg.norm(np.delete(d @ step_vector, source))),
                                        "internal_step_l2": 0., "value_step_l2": 0.,
                                        "head_update_calls": trial + 1, "internal_update_calls": 0,
                                        "value_update_calls": 0, "cumulative_head_path_l2": path,
                                        "cumulative_head_squared_l2": square, "head_norm": float(np.linalg.norm(new_head))}
                        for key, value in expected_row.items():
                            demand(key in row and np.isclose(float(row[key]), value, atol=1e-9, rtol=1e-9),
                                   "Saved CSV/array arithmetic: " + key)
                        head = new_head
                        if trial + 1 in grid:
                            check_checkpoint(grid.index(trial + 1))
    return {"influences": influences, "directions": directions, "unit_rows": unit_rows,
            "trial_margin_delta": margin_delta, "trial_margin_before": margin_before,
            "trial_probability_delta": probability_delta,
            "parent_increment_residual": reconstruction_residual}


def validate_science(cfg, allow_fixture=False):
    fixed = {"study_id": "signed_spillover_v1", "scope": "retrospective_descriptive_signed_spillover",
             "module": "drosolearn.signed_spillover", "protocol": "docs/signed-spillover-protocol.md",
             "source_result": "results/spillover_tradeoff_v1", "topologies": list(TOPOLOGIES),
             "representations": list(REPRESENTATIONS), "alphas": list(ALPHAS),
             "model_seeds": list(range(17301, 17311)), "training_trials": 400,
             "checkpoints": [0, 50, 100, 200, 400], "head_lr": .1, "numeric_tolerance": 1e-8,
             "neutral_tolerance": 1e-8, "labels": SIGNS.astype(int).tolist(),
             "categories": list(CATEGORY_NAMES), "external_manifest": None}
    fixture = allow_fixture and cfg.get("scope") == "integration_test"
    for key, value in fixed.items():
        if fixture and key in ("scope", "model_seeds", "training_trials", "checkpoints"):
            continue
        demand(cfg.get(key) == value, "Fixed signed accounting: " + key)
    if fixture:
        seeds, n, grid = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
        demand(type(seeds) is list and 1 <= len(seeds) <= 2 and len(set(seeds)) == len(seeds)
               and all(type(s) is int and 90001 <= s <= 90100 for s in seeds), "Invented seeds only")
        demand(type(n) is int and 4 <= n <= 16 and n % 4 == 0, "Invented saved trial budget only")
        demand(type(grid) is list and grid == sorted(set(grid)) and grid[0] == 0 and grid[-1] == n
               and all(type(k) is int and 0 <= k <= n for k in grid), "Invented saved checkpoints")


def expected_artifacts(cfg, parent_arrays, parent_rows):
    """Reconstruct exact eight-array contract from immutable saved observations."""
    state = saved_parent_state(cfg, parent_arrays, parent_rows)
    influences = state["influences"]
    n, grid, eta = cfg["training_trials"], cfg["checkpoints"], cfg["head_lr"]
    shape = influences.shape[:4]
    static = influences * SIGNS[None, None, None, None, :, None] * SIGNS
    pre_logits = state["trial_margin_before"] * SIGNS
    logits = parent_arrays["checkpoint_logits"]
    go = sigmoid(logits)
    margin_expected = (eta / 4. * go[..., :, None] * (1. - go[..., :, None])
                       * influences[..., None, :, :] * SIGNS[:, None] * SIGNS)
    branch_logits = (logits[..., None, :] + eta * (1. - go[..., :, None])
                     * SIGNS[:, None] * influences[..., None, :, :])
    branch_correct = sigmoid(branch_logits * SIGNS)
    current_correct = sigmoid(logits * SIGNS)
    probability_expected = go[..., :, None] / 4. * (branch_correct - current_correct[..., None, :])
    residuals = np.zeros(shape + (len(RESIDUAL_NAMES),))
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            prefix = "t_" + topology + "_" + representation + "_"
            for ai, _ in enumerate(ALPHAS):
                for si, _ in enumerate(cfg["model_seeds"]):
                    at = (ti, ri, ai, si)
                    source = parent_arrays["schedules"][si]
                    m = influences[at]
                    d = state["unit_rows"][ti, ri][si]
                    w = state["directions"][ti, ri][si, ai]
                    g = parent_arrays["trial_update_factors"][at]
                    scores = pre_logits[at]
                    p = sigmoid(scores[np.arange(n), source])
                    action, reward = parent_arrays["trial_actions"][at], parent_arrays["trial_rewards"][at]
                    expected_g = eta * (action - p) * reward
                    score_steps = g[:, None] * m[source]
                    own = score_steps[np.arange(n), source]
                    other = score_steps.copy()
                    other[np.arange(n), source] = 0.
                    head_steps = np.abs(g) * np.linalg.norm(w[source], axis=-1)
                    accumulated_heads = np.concatenate((np.zeros((1, w.shape[-1])),
                                                        np.cumsum(g[:, None] * w[source], axis=0)))
                    checkpoint_heads = accumulated_heads[grid]
                    checkpoint_logits = checkpoint_heads @ d.T
                    checkpoint_prob = sigmoid(checkpoint_logits)
                    def maxerr(a, b):
                        return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                    initial_margin = SIGNS * logits[at][0]
                    final_margin = SIGNS * logits[at][-1]
                    residuals[at] = [
                        maxerr(parent_arrays[prefix + "influences"][si, ai], m),
                        maxerr(g, expected_g), maxerr(parent_arrays["trial_probabilities"][at], p),
                        maxerr(parent_arrays["trial_own_score_steps"][at], own),
                        maxerr(parent_arrays["trial_other_score_steps"][at], np.linalg.norm(other, axis=-1)),
                        maxerr(parent_arrays["trial_head_steps"][at], head_steps),
                        maxerr(parent_arrays["checkpoint_heads"][at], checkpoint_heads),
                        maxerr(logits[at], checkpoint_logits),
                        maxerr(parent_arrays["checkpoint_probabilities"][at], checkpoint_prob),
                        maxerr(state["trial_margin_delta"][at].sum(axis=0), final_margin - initial_margin),
                        maxerr(state["trial_probability_delta"][at].sum(axis=0),
                               sigmoid(final_margin) - sigmoid(initial_margin))]
    arrays = {"static_target_aligned": static, "trial_margin_delta": state["trial_margin_delta"],
              "trial_probability_delta": state["trial_probability_delta"], "trial_pre_logits": pre_logits,
              "checkpoint_expected_margin_by_source": margin_expected,
              "checkpoint_expected_probability_by_source": probability_expected,
              "reconstruction_residuals": residuals, "final_correct_probability": current_correct[..., -1, :]}
    demand(all(np.isfinite(value).all() for value in arrays.values()), "Finite derived signed artifacts")
    demand(float(residuals.max()) <= cfg["numeric_tolerance"], "All saved-state identities required")
    return arrays, expected_rows(cfg, arrays, parent_arrays["schedules"])


def signed_values(values, mask, prefix, suffix, tolerance):
    selected = np.asarray(values)[mask]
    return {prefix + "helpful_cells": int(np.count_nonzero(selected > tolerance)),
            prefix + "harmful_cells": int(np.count_nonzero(selected < -tolerance)),
            prefix + "neutral_cells": int(np.count_nonzero(np.abs(selected) <= tolerance)),
            prefix + "helpful_" + suffix: float(np.maximum(selected, 0.).sum()),
            prefix + "harmful_" + suffix: float(np.maximum(-selected, 0.).sum()),
            prefix + "net_" + suffix: float(selected.sum())}


def expected_rows(cfg, arrays, schedules):
    rows = []
    masks, tol = category_masks(), cfg["neutral_tolerance"]
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            for ai, alpha in enumerate(ALPHAS):
                for si, seed in enumerate(cfg["model_seeds"]):
                    at = (ti, ri, ai, si)
                    for ci, category in enumerate(CATEGORY_NAMES):
                        mask = masks[ci]
                        trial_mask = mask[schedules[si]]
                        row = {"topology": topology, "representation": representation, "alpha": alpha,
                               "seed": seed, "category": category, "static_cells": int(mask.sum()),
                               "actual_cells": int(trial_mask.sum())}
                        row.update(signed_values(arrays["static_target_aligned"][at], mask, "static_", "mass", tol))
                        row.update(signed_values(arrays["trial_margin_delta"][at], trial_mask, "margin_", "sum", tol))
                        row.update(signed_values(arrays["trial_probability_delta"][at], trial_mask, "probability_", "sum", tol))
                        rows.append(row)
    return rows


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if values.size > 1 else 0.}


def expected_summary(cfg, arrays, rows):
    families, checkpoints = [], []
    masks = category_masks()
    group_size = len(cfg["model_seeds"])
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            for ai, alpha in enumerate(ALPHAS):
                matching = [row for row in rows if row["topology"] == topology
                            and row["representation"] == representation and row["alpha"] == alpha]
                categories = {}
                for category in CATEGORY_NAMES:
                    group = [row for row in matching if row["category"] == category]
                    demand(len(group) == group_size, "Complete seed category group")
                    categories[category] = {metric: mean_sd([row[metric] for row in group]) for metric in NUMERIC_FIELDS}
                families.append({"topology": topology, "representation": representation, "alpha": alpha,
                                 "role": "primary_descriptive" if (ti, ri) == (0, 0) else "descriptive_control",
                                 "final_expected_accuracy": float(arrays["final_correct_probability"][ti, ri, ai].mean()),
                                 "categories": categories})
                for ki, checkpoint in enumerate(cfg["checkpoints"]):
                    category_stats = {}
                    for ci, category in enumerate(CATEGORY_NAMES):
                        margin = (arrays["checkpoint_expected_margin_by_source"][ti, ri, ai, :, ki]
                                  * masks[ci]).sum(axis=(-1, -2)) / 4.
                        probability = (arrays["checkpoint_expected_probability_by_source"][ti, ri, ai, :, ki]
                                       * masks[ci]).sum(axis=(-1, -2)) / 4.
                        a, b = mean_sd(margin), mean_sd(probability)
                        category_stats[category] = {"mean_margin_drift": a["mean"], "sd_margin_drift": a["sd"],
                                                    "mean_probability_drift": b["mean"], "sd_probability_drift": b["sd"]}
                    checkpoints.append({"topology": topology, "representation": representation, "alpha": alpha,
                                        "checkpoint": checkpoint, "categories": category_stats})
    residuals = arrays["reconstruction_residuals"].reshape(-1, len(RESIDUAL_NAMES)).max(axis=0)
    balances, counts = [], []
    by_seed = {}
    for row in rows:
        key = (row["topology"], row["representation"], row["alpha"], row["seed"])
        by_seed.setdefault(key, {})[row["category"]] = row
    for group in by_seed.values():
        for metric in NUMERIC_FIELDS:
            for whole, parts in (("all", ("self", "cross_same", "cross_opposite")),
                                 ("cross_all", ("cross_same", "cross_opposite"))):
                error = abs(group[whole][metric] - sum(group[part][metric] for part in parts))
                balances.append(error == 0 if metric.endswith("_cells") else error <= cfg["numeric_tolerance"])
    for row in rows:
        for prefix, suffix, count in (("static_", "mass", "static_cells"),
                                      ("margin_", "sum", "actual_cells"),
                                      ("probability_", "sum", "actual_cells")):
            balances.append(abs(row[prefix + "helpful_" + suffix] - row[prefix + "harmful_" + suffix]
                                - row[prefix + "net_" + suffix]) <= cfg["numeric_tolerance"])
            counts.append(sum(row[prefix + sign + "_cells"] for sign in ("helpful", "harmful", "neutral")) == row[count])
    groups = 30 * group_size
    return {"study_id": cfg["study_id"], "status": "completed", "scope": cfg["scope"],
            "categories": list(CATEGORY_NAMES), "families": families, "checkpoint_expectations": checkpoints,
            "validation": {"residual_axis": list(RESIDUAL_NAMES), "max_residual_by_axis": residuals.tolist(),
                           "all_reconstruction_and_telescoping_checks_passed": bool(np.all(residuals <= cfg["numeric_tolerance"])),
                           "all_mass_balances_passed": bool(all(balances)),
                           "all_classification_counts_complete": bool(all(counts))},
            "counts": {"analyzed_parent_trial_records": groups * cfg["training_trials"],
                       "trial_destination_cells": groups * cfg["training_trials"] * 4,
                       "cross_destination_cells": groups * cfg["training_trials"] * 3,
                       "static_ordered_input_pairs": groups * 16,
                       "checkpoint_destination_cells": groups * len(cfg["checkpoints"]) * 4,
                       "checkpoint_source_destination_cells": groups * len(cfg["checkpoints"]) * 16,
                       "analysis_groups": groups, "conditions": 30, "model_seeds": group_size,
                       "measurement_records": len(rows), "new_training_trials": 0, "new_head_update_calls": 0,
                       "new_internal_update_calls": 0, "new_value_update_calls": 0, "new_random_draws": 0,
                       "fresh_model_constructions": 0},
            "interpretation": {"prior_outcomes_known": True, "confirmatory_behavioral_hypothesis": False,
                               "source_labels_used_in_analysis_only": True, "pathwise_accounting_not_isolated_causal_effect": True,
                               "uniform_source_checkpoint_expectation_not_actual_next_schedule": True,
                               "probability_expectation_not_sigmoid_of_expected_margin": True,
                               "neutral_counts_do_not_zero_exact_masses": True, "internal_synapses_fixed_in_parent": True,
                               "generalization_tested": False, "new_human_experiment": False,
                               "shared_neural_mechanism_claim": False, "education_effect_claim": False,
                               "output_cost_optimization_performed": False}}


def check_artifacts(expected, expected_rows, arrays, rows):
    demand(set(arrays) == set(expected), "Independent eight-field signed artifact closure")
    for key in expected:
        close(arrays[key], expected[key], "Independent signed array: " + key)
    demand(len(rows) == len(expected_rows), "Complete signed category CSV")
    for actual, wanted in zip(rows, expected_rows):
        demand(set(actual) == set(wanted), "Signed category CSV field closure")
        for key, value in wanted.items():
            if type(value) is str:
                demand(actual[key] == value, "Signed category CSV identity: " + key)
            else:
                number = float(actual[key])
                matches = (number.is_integer() and int(number) == value) if type(value) is int else np.isclose(number, value, atol=1e-9, rtol=1e-9)
                demand(np.isfinite(number) and matches,
                       "Signed category CSV arithmetic: " + key)


def verify_science(cfg, parent_arrays, parent_rows, arrays, rows, summary, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    expected, expected_csv = expected_artifacts(cfg, parent_arrays, parent_rows)
    check_artifacts(expected, expected_csv, arrays, rows)
    wanted_summary = expected_summary(cfg, expected, expected_csv)
    demand(all(wanted_summary["validation"][key] is True for key in
               ("all_reconstruction_and_telescoping_checks_passed", "all_mass_balances_passed",
                "all_classification_counts_complete")), "All independent signed accounting gates")
    compare_tree(summary, wanted_summary, "Independent complete signed summary")
    return {"arrays": len(expected), "parent_trial_records_checked": 30 * len(cfg["model_seeds"]) * cfg["training_trials"],
            "trial_destination_effects_checked": int(np.prod(expected["trial_margin_delta"].shape)),
            "static_ordered_input_pairs_checked": int(np.prod(expected["static_target_aligned"].shape)),
            "checkpoint_source_destination_effects_checked": int(np.prod(expected["checkpoint_expected_margin_by_source"].shape)),
            "category_measurement_records_checked": len(expected_csv), "no_rng_or_learner_calls": True,
            "source_destination_orientation_checked": True, "all_saved_checkpoints_checked": True,
            "obtained_reward_only_nogo_zero_checked": True, "actual_margin_and_probability_telescoping_checked": True,
            "exact_masses_separate_from_tolerance_counts_checked": True,
            "finite_go_branch_expectation_not_sigmoid_of_expected_margin_checked": True,
            "pathwise_groups_not_isolated_causal_effect": True, "no_new_behavioral_hypothesis_claim": True}


def load_rows(path):
    with Path(path).open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        demand(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames),
               "Unique saved CSV field names")
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
    from drosolearn.signed_envelope import verify_envelope, sha, write_new
    cfg, _, meta = verify_envelope(args.root, args.config, args.registration, args.result_dir, args.publication_proof)
    validate_science(cfg)
    with np.load(args.root / cfg["source_result"] / "arrays.npz", allow_pickle=False) as saved:
        parent = {key: saved[key] for key in saved.files if key.startswith(("trial_", "checkpoint_", "t_"))
                  or key in ("model_seeds", "alphas", "checkpoints", "targets", "schedules")}
    parent_rows = load_rows(args.root / cfg["source_result"] / "measurements.csv")
    with np.load(args.result_dir / "arrays.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    rows = load_rows(args.result_dir / "measurements.csv")
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf8"))
    checks = verify_science(cfg, parent, parent_rows, arrays, rows, summary)
    demand(meta["counts"] == summary["counts"], "Verified result/metadata count binding")
    report = {"status": "passed", "study_id": cfg["study_id"],
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "registration_sha256": sha(args.registration), "summary_sha256": sha(args.result_dir / "summary.json"),
              "scientific_producer_imported": False, "new_helper_imported": False,
              "counts": checks}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
