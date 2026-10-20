"""Independent descriptive accounting of frozen pre/post-gate saved paths.

No producer, learner, new graph, optimizer or RNG is imported or executed.
The parent autonomous replay and its byte-level gate checks are frozen input
receipts; the new head/effect/phase/category/cost arithmetic is recomputed here.
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
PHASES = ("pre", "post")
CATEGORIES = ("own", "known_other", "unknown_other")
CONTROLS = ("balanced", "isolated", "feedback_selective", "shared")
SIGNS = np.asarray([1., 1., -1., -1.])
FIELDS = ("probability_changes", "margin_changes", "effect_categories", "gated_phase_before", "own_full_before",
    "first_full_trials", "first_gated_balanced_trials", "head_step_l2", "head_step_squared_l2", "model_seeds", "schedules", "targets")
COSTS = ("trial_count", "go_trials", "nonzero_head_steps", "head_path_l2", "head_squared_l2",
         "endpoint_contribution", "gain_per_Go", "gain_per_path")


def demand(condition, label):
    if not condition: raise ValueError(label)


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
        for key in expected: compare_tree(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, list):
        demand(isinstance(actual, list) and len(actual) == len(expected), label + " length")
        for i, value in enumerate(expected): compare_tree(actual[i], value, label + "." + str(i))
    elif type(expected) is bool:
        demand(type(actual) is bool and actual == expected, label + " flag")
    elif type(expected) is int:
        demand(type(actual) is int and actual == expected, label + " integer")
    elif type(expected) is float:
        demand(type(actual) in (int, float) and np.isfinite(actual)
               and np.isclose(actual, expected, atol=1e-9, rtol=1e-9), label + " numeric")
    else: demand(actual == expected, label + " value")


def sigmoid(values):
    return 1. / (1. + np.exp(-np.clip(values, -40., 40.)))


def validate_science(cfg, allow_fixture=False):
    fixed = {"study_id": "gated_pathways_v1", "scope": "retrospective_descriptive_gated_pathways",
        "module": "drosolearn.gated_pathways", "protocol": "docs/gated-pathways-protocol.md",
        "parent_result": "results/gated_learning_v1", "baseline_tag": "gated-learning-v1",
        "baseline_commit": "b89e9f2baf1aed91b4451538c85ba3e9458df53b", "topologies": list(TOPS),
        "representations": list(REPS), "methods": list(METHODS), "model_seeds": list(range(17601, 17611)),
        "training_trials": 400, "numeric_tolerance": 1e-8,
        "phase_definition": "reference_gated_known_before_count_equals_four",
        "category_definition": "own_each_method_known_before_other_unknown_before_other",
        "reference_method": "gated_balanced", "contrast_controls": list(CONTROLS),
        "new_confirmatory_hypothesis": False, "external_manifest": None}
    demand(isinstance(cfg, dict), "Independent common-time saved-path configuration")
    fixture = cfg.get("scope") == "integration_test"
    demand(not fixture or allow_fixture, "Invented fixture cannot enter actual analysis")
    for key, value in fixed.items():
        if fixture and key in ("scope", "model_seeds", "training_trials"): continue
        demand(cfg.get(key) == value, "Fixed common-time analysis choice: " + key)
    if fixture:
        seeds, n = cfg.get("model_seeds"), cfg.get("training_trials")
        demand(type(seeds) is list and 1 <= len(seeds) <= 2 and len(set(seeds)) == len(seeds)
               and all(type(seed) is int and 90001 <= seed <= 90100 for seed in seeds), "Invented saved-path seeds only")
        demand(type(n) is int and 4 <= n <= 16, "Bounded invented saved-path length")


def expected_artifacts(cfg, parent):
    s, n = len(cfg["model_seeds"]), cfg["training_trials"]
    shape = (3, 2, 5, s, n)
    close(parent["model_seeds"], np.asarray(cfg["model_seeds"]), "Frozen parent seed identity")
    close(parent["targets"], (SIGNS > 0).astype(int), "Frozen four target identities")
    source = parent["schedules"]
    demand(source.shape == (s, n) and source.dtype.kind in "iu" and np.isin(source, range(4)).all(), "Saved current-input schedules")
    before, after = parent["trial_known_before"], parent["trial_known_after"]
    demand(before.shape == after.shape == shape + (4,) and before.dtype.kind in "biu" and after.dtype.kind in "biu"
           and np.isin(before, (-1, 0, 1)).all() and np.isin(after, (-1, 0, 1)).all(), "Each method's actual pre/post observed memory")
    own_full = np.all(before != 0, axis=-1)
    reference = own_full[:, :, 4]
    demand(not np.any(reference[..., :-1] & ~reference[..., 1:]), "Reference observation gate cannot close again")
    first_full = np.where(np.any(np.all(after != 0, axis=-1), axis=-1),
                          np.argmax(np.all(after != 0, axis=-1), axis=-1) + 1, n + 1).astype(np.int32)
    first_gate = np.where(reference.any(axis=-1), reference.argmax(axis=-1) + 1, n + 1).astype(np.int32)
    close(first_gate, np.minimum(first_full[:, :, 4] + 1, n + 1), "Last discovery remains in the pre phase")
    close(parent["first_balanced_mode_trials"][:, :, 4], first_gate, "Frozen parent first actual gate mode")
    category = np.where(before == 0, 2, 1).astype(np.int8)
    own = np.eye(4, dtype=bool)[np.broadcast_to(source, shape)]
    category[own] = 0
    g, w = parent["trial_update_factors"], parent["trial_directions"]
    demand(g.shape == shape and w.shape[:-1] == shape and np.isfinite(g).all() and np.isfinite(w).all(), "Saved finite update scalars/directions")
    q, margin, step = np.empty(shape + (4,)), np.empty(shape + (4,)), np.empty(shape)
    grid = parent["checkpoints"].tolist()
    demand(grid == sorted(set(grid)) and grid[0] == 0 and grid[-1] == n, "Frozen saved checkpoint grid")
    for ti, top in enumerate(TOPS):
        for ri, rep in enumerate(REPS):
            features = parent["g_" + top + "_" + rep + "_unit_rows"]
            demand(features.shape == (s, 4, w.shape[-1]) and np.isfinite(features).all(), "Frozen fixed-feature axes")
            close(np.linalg.norm(features, axis=-1), np.ones((s, 4)), "Frozen unit decision features")
            for si in range(s):
                d = features[si]
                for mi in range(5):
                    at = (ti, ri, mi, si)
                    updates = g[at][:, None] * w[at]
                    heads = np.concatenate((np.zeros((1, d.shape[1])), np.cumsum(updates, axis=0)))
                    z = heads[:-1] @ d.T
                    effect = g[at][:, None] * (w[at] @ d.T)
                    q[at] = SIGNS * (sigmoid(z + effect) - sigmoid(z))
                    margin[at] = SIGNS * effect
                    step[at] = np.linalg.norm(updates, axis=-1)
                    close(parent["trial_correct_probability_delta"][at], q[at], "Frozen factual same-state correct-probability change")
                    close(parent["trial_aligned_margin_delta"][at], margin[at], "Frozen factual signed logit change")
                    close(parent["checkpoint_heads"][at], heads[grid], "Saved output head prefix at every checkpoint")
                    close(parent["checkpoint_probabilities"][at], sigmoid(heads[grid] @ d.T), "Saved output probabilities at every checkpoint")
    close(parent["trial_head_steps"], step, "Recorded actual head movement length")
    close(parent["trial_head_step_squared_l2"], step ** 2, "Recorded actual squared movement")
    close(parent["trial_cumulative_head_path_l2"], np.cumsum(step, axis=-1), "Recorded cumulative actual path")
    close(parent["trial_cumulative_head_squared_l2"], np.cumsum(step ** 2, axis=-1), "Recorded cumulative squared movement")
    demand(parent["trial_actions"].shape == shape and parent["trial_actions"].dtype.kind in "biu"
           and np.isin(parent["trial_actions"], (0, 1)).all(), "Frozen actual Go/NoGo records")
    result = {"probability_changes": q, "margin_changes": margin, "effect_categories": category,
        "gated_phase_before": reference, "own_full_before": own_full, "first_full_trials": first_full,
        "first_gated_balanced_trials": first_gate, "head_step_l2": step, "head_step_squared_l2": step ** 2,
        "model_seeds": parent["model_seeds"].copy(), "schedules": source.copy(), "targets": parent["targets"].copy()}
    demand(tuple(result) == FIELDS, "Twelve independent factual output fields")
    return result


def expected_rows(cfg, arrays, parent):
    rows = []
    for ti, top in enumerate(TOPS):
        for ri, rep in enumerate(REPS):
            for mi, method in enumerate(METHODS):
                for si, seed in enumerate(cfg["model_seeds"]):
                    at = (ti, ri, mi, si)
                    for ph, phase in enumerate(PHASES):
                        selected = arrays["gated_phase_before"][ti, ri, si] == ph
                        go = int(parent["trial_actions"][at][selected].sum())
                        path = float(arrays["head_step_l2"][at][selected].sum())
                        energy = float(arrays["head_step_squared_l2"][at][selected].sum())
                        for ci, category in enumerate(CATEGORIES):
                            mask = selected[:, None] & (arrays["effect_categories"][at] == ci)
                            q, margin = arrays["probability_changes"][at][mask], arrays["margin_changes"][at][mask]
                            contribution = float(q.sum() / 4.)
                            rows.append({"topology": top, "representation": rep, "method": method, "seed": seed,
                                "phase": phase, "category": category, "trial_count": int(selected.sum()), "exposures": int(mask.sum()),
                                "nonzero_probability_effects": int(np.count_nonzero(q)), "positive_probability_effects": int(np.count_nonzero(q > 0.)),
                                "negative_probability_effects": int(np.count_nonzero(q < 0.)), "probability_positive_mass": float(np.maximum(q, 0.).sum()),
                                "probability_negative_mass": float(np.maximum(-q, 0.).sum()), "probability_signed_sum": float(q.sum()),
                                "margin_positive_mass": float(np.maximum(margin, 0.).sum()), "margin_negative_mass": float(np.maximum(-margin, 0.).sum()),
                                "margin_signed_sum": float(margin.sum()), "go_trials": go,
                                "nonzero_head_steps": int(np.count_nonzero(arrays["head_step_l2"][at][selected])),
                                "head_path_l2": path, "head_squared_l2": energy,
                                "gain_per_Go": contribution / go if go else None, "gain_per_path": contribution / path if path else None,
                                "endpoint_contribution": contribution})
    return rows


def mean_or_none(values):
    usable = [value for value in values if value is not None]
    return float(np.mean(usable)) if usable else None


def expected_summary(cfg, arrays, rows, parent):
    seeds, tol = cfg["model_seeds"], cfg["numeric_tolerance"]
    q = arrays["probability_changes"]
    correct = np.where(arrays["targets"] == 1, parent["checkpoint_probabilities"], 1. - parent["checkpoint_probabilities"])
    initial, final = correct[..., 0, :].mean(-1), correct[..., -1, :].mean(-1)
    indexed = {}
    for row in rows:
        key = tuple(row[name] for name in ("topology", "representation", "method", "seed", "phase", "category"))
        demand(key not in indexed, "Unique common-phase/category measurement row")
        indexed[key] = row
    contribution = np.zeros(q.shape[:4] + (2, 3))
    positive, negative = np.zeros_like(contribution), np.zeros_like(contribution)
    phase_costs, families, contrasts = {}, [], []
    for ti, top in enumerate(TOPS):
        for ri, rep in enumerate(REPS):
            for mi, method in enumerate(METHODS):
                records = []
                for si, seed in enumerate(seeds):
                    at = (ti, ri, mi, si)
                    costs = []
                    for ph, phase in enumerate(PHASES):
                        for ci, category in enumerate(CATEGORIES):
                            row = indexed[top, rep, method, seed, phase, category]
                            contribution[at + (ph, ci)] = row["endpoint_contribution"]
                            positive[at + (ph, ci)] = row["probability_positive_mass"]
                            negative[at + (ph, ci)] = row["probability_negative_mass"]
                        row = indexed[top, rep, method, seed, phase, "own"]
                        selected = arrays["gated_phase_before"][ti, ri, si] == ph
                        gain = float(q[at][selected].sum() / 4.)
                        go, path = row["go_trials"], row["head_path_l2"]
                        costs.append({"trial_count": row["trial_count"], "go_trials": go,
                            "nonzero_head_steps": row["nonzero_head_steps"], "head_path_l2": path,
                            "head_squared_l2": row["head_squared_l2"], "endpoint_contribution": gain,
                            "gain_per_Go": gain / go if go else None, "gain_per_path": gain / path if path else None})
                    phase_costs[at] = costs
                    records.append({"seed": seed, "initial_expected_accuracy": float(initial[at]), "final_expected_accuracy": float(final[at]),
                        "reconstructed_accuracy_gain": float(contribution[at].sum()), "phase_category_endpoint_contributions": contribution[at].tolist(),
                        "phase_category_positive_probability_mass": positive[at].tolist(), "phase_category_negative_probability_mass": negative[at].tolist(),
                        "first_full_trial": int(arrays["first_full_trials"][at]),
                        "reference_first_balanced_trial": int(arrays["first_gated_balanced_trials"][ti, ri, si]),
                        "phase_costs": costs, "own_full_vs_reference_phase_disagreement_trials": int(np.count_nonzero(
                            arrays["own_full_before"][at] != arrays["gated_phase_before"][ti, ri, si]))})
                means = [{name: mean_or_none([record["phase_costs"][ph][name] for record in records]) for name in COSTS} for ph in range(2)]
                families.append({"topology": top, "representation": rep, "method": method, "seeds": records,
                    "mean_initial_expected_accuracy": float(initial[ti, ri, mi].mean()), "mean_final_expected_accuracy": float(final[ti, ri, mi].mean()),
                    "mean_phase_category_endpoint_contributions": contribution[ti, ri, mi].mean(axis=0).tolist(),
                    "mean_phase_category_positive_probability_mass": positive[ti, ri, mi].mean(axis=0).tolist(),
                    "mean_phase_category_negative_probability_mass": negative[ti, ri, mi].mean(axis=0).tolist(), "mean_phase_costs": means})
            for control in CONTROLS:
                mi, records = METHODS.index(control), []
                for si, seed in enumerate(seeds):
                    g, c = (ti, ri, 4, si), (ti, ri, mi, si)
                    records.append({"seed": seed, "endpoint_difference": float(final[g] - final[c]),
                        "phase_category_endpoint_difference": (contribution[g] - contribution[c]).tolist(),
                        "phase_category_positive_probability_mass_difference": (positive[g] - positive[c]).tolist(),
                        "phase_category_negative_probability_mass_difference": (negative[g] - negative[c]).tolist(),
                        "phase_endpoint_difference": (contribution[g].sum(axis=-1) - contribution[c].sum(axis=-1)).tolist(),
                        "phase_head_path_difference": [phase_costs[g][ph]["head_path_l2"] - phase_costs[c][ph]["head_path_l2"] for ph in range(2)],
                        "phase_Go_difference": [phase_costs[g][ph]["go_trials"] - phase_costs[c][ph]["go_trials"] for ph in range(2)],
                        "own_full_trial_difference": int(arrays["first_full_trials"][g] - arrays["first_full_trials"][c]),
                        "unknown_mask_disagreement_destination_count": int(np.count_nonzero(
                            (arrays["effect_categories"][g] == 2) != (arrays["effect_categories"][c] == 2)))})
                means = {name: np.asarray([record[name] for record in records]).mean(axis=0).tolist() for name in records[0] if name != "seed"}
                contrasts.append({"topology": top, "representation": rep, "control": control,
                    "role": "primary" if (ti, ri, mi) == (0, 0, 3) else "descriptive_control", "seeds": records, "means": means})
    cost_path = max(abs(sum(cost["head_path_l2"] for cost in costs) - parent["trial_cumulative_head_path_l2"][at + (-1,)])
                    for at, costs in phase_costs.items())
    cost_energy = max(abs(sum(cost["head_squared_l2"] for cost in costs) - parent["trial_cumulative_head_squared_l2"][at + (-1,)])
                      for at, costs in phase_costs.items())
    residuals = {"endpoint_telescoping": float(np.max(np.abs(contribution.sum(axis=(-1, -2)) - (final - initial)))),
        "positive_negative_identity": float(np.max(np.abs(contribution - (positive - negative) / 4.))),
        "phase_category_partition": float(np.max(np.abs(contribution.sum(axis=(-1, -2)) - q.sum(axis=(-1, -2)) / 4.))),
        "contrast_endpoint": float(max(abs(record["endpoint_difference"] - np.asarray(record["phase_category_endpoint_difference"]).sum())
            for contrast in contrasts for record in contrast["seeds"])), "phase_cost_path": float(cost_path), "phase_cost_energy": float(cost_energy)}
    demand(all(np.isfinite(value) and value <= tol for value in residuals.values()), "Independent exact common-phase accounting gates")
    demand(len(rows) == 30 * len(seeds) * 6 and len(arrays) == 12, "Complete common-phase output closure")
    return {"study_id": cfg["study_id"], "status": "completed", "scope": cfg["scope"], "families": families, "contrasts": contrasts,
        "phase_order": list(PHASES), "category_order": list(CATEGORIES), "contrast_control_order": list(CONTROLS),
        "validation": {"all_exact_accounting_gates_passed": True, "residuals": residuals,
            "common_reference_phase_uses_prefeedback_memory": True, "last_discovery_trial_assigned_to_pre": True,
            "each_method_categories_use_own_observations": True, "zero_effects_and_roundoff_signs_preserved": True,
            "no_counterfactual_policy_or_direction_evaluated": True},
        "counts": {"analyzed_parent_trial_records": int(np.prod(q.shape[:-1])), "analyzed_destination_effects": int(q.size),
            "reference_gate_paths": 6 * len(seeds), "paired_comparison_paths": 6 * len(seeds) * 4,
            "paired_comparison_trial_records": 6 * len(seeds) * 4 * cfg["training_trials"], "measurement_records": len(rows), "numeric_arrays": len(arrays),
            "new_training_trials": 0, "new_head_update_calls": 0, "new_internal_update_calls": 0, "new_value_update_calls": 0,
            "new_random_draws": 0, "fresh_model_constructions": 0, "new_optimization_calls": 0},
        "interpretation": {"retrospective_descriptive_only": True, "parent_H_gain": False, "parent_H_acquisition": False,
            "parent_H_protection": True, "new_confirmatory_hypothesis": False, "exact_path_accounting_not_unique_causal_explanation": True,
            "reference_gate_is_common_time_split_not_all_methods_knowledge": True, "same_realized_movement_budget_claimed": False,
            "internal_synapses_and_probe_activity_fixed": True, "observed_memory_is_separate_from_output_weights": True,
            "novel_input_generalization_tested": False, "new_human_experiment": False, "shared_neural_mechanism_claim": False, "education_effect_claim": False}}


def check_artifacts(arrays, wanted, rows, wanted_rows):
    demand(set(arrays) == set(wanted) == set(FIELDS), "Independent twelve-array closure")
    for name, value in wanted.items(): close(arrays[name], value, "Independent factual common-phase array: " + name)
    demand(len(rows) == len(wanted_rows), "Complete common-phase CSV row closure")
    for actual, expected in zip(rows, wanted_rows):
        demand(tuple(actual) == tuple(expected), "Exact common-phase CSV field order")
        for name, value in expected.items():
            if value is None: demand(actual[name] is None or actual[name] == "", "Undefined denominator must be empty: " + name)
            elif type(value) is str: demand(actual[name] == value, "CSV fixed identity: " + name)
            else:
                number = float(actual[name])
                demand(np.isfinite(number) and (number == value if type(value) is int else np.isclose(number, value, atol=1e-9, rtol=1e-9)),
                       "CSV count or arithmetic: " + name)


def verify_science(cfg, parent, arrays, rows, summary, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    wanted = expected_artifacts(cfg, parent)
    wanted_rows = expected_rows(cfg, wanted, parent)
    check_artifacts(arrays, wanted, rows, wanted_rows)
    expected = expected_summary(cfg, wanted, wanted_rows, parent)
    compare_tree(summary, expected, "Independent complete common-reference phase summary")
    trials = int(np.prod(parent["trial_actions"].shape))
    return {"arrays": len(wanted), "parent_trial_records_checked": trials, "destination_effects_checked": trials * 4,
        "measurement_records_checked": len(wanted_rows), "all_factual_head_prefixes_and_parent_effects_reconstructed": True,
        "all_methods_use_same_reference_prefeedback_time_boundary": True,
        "own_method_observations_used_for_known_unknown_categories": True,
        "last_discovery_trial_pre_and_next_trial_post_checked": True,
        "positive_negative_masses_and_endpoint_contributions_separated": True,
        "costs_Go_counts_and_nullable_descriptive_ratios_checked": True,
        "all_four_controls_and_all_thirty_families_checked": True,
        "parent_full_autonomous_replay_and_byte_gate_reused_by_frozen_hashes": True,
        "no_counterfactual_direction_policy_rng_graph_or_solver_calls": True,
        "new_training_trials": 0, "new_random_draws": 0, "new_optimization_calls": 0}


def load_rows(path):
    with Path(path).open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        demand(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames), "Unique measurement CSV fields")
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
    from drosolearn.gated_pathways_envelope import verify_envelope, sha, write_new
    cfg, _, metadata = verify_envelope(args.root, args.config, args.registration, args.result_dir, args.publication_proof)
    with np.load(args.root / cfg["parent_result"] / "arrays.npz", allow_pickle=False) as handle:
        parent = {name: handle[name] for name in handle.files}
    with np.load(args.result_dir / "arrays.npz", allow_pickle=False) as handle:
        arrays = {name: handle[name] for name in handle.files}
    rows = load_rows(args.result_dir / "measurements.csv")
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf8"))
    checks = verify_science(cfg, parent, arrays, rows, summary)
    demand(metadata["counts"] == summary["counts"], "Metadata and independently verified summary counts")
    report = {"status": "passed", "study_id": cfg["study_id"], "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_sha256": sha(args.registration), "summary_sha256": sha(args.result_dir / "summary.json"),
        "scientific_producer_imported": False, "new_helper_imported": False, "counts": checks}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__": main()
