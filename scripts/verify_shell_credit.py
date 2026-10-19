"""Independent arithmetic for the finite stationary shell-credit diagnostic.

The frozen earlier independent anatomical/encoder implementation is reused.
No scientific producer is imported by the verification implementation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np


_PATH = Path(__file__).resolve().with_name("verify_human_shell.py")
_SPEC = importlib.util.spec_from_file_location("frozen_shell_credit_independent_core", _PATH)
core = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(core)
NAMESPACE = 202610191
CONDITIONS = ("shared", "isolated", "uniform")


def validate_science(cfg, allow_fixture=False):
    expected = {"study_id": "shell_credit_v1", "graph": "data/flywire_783_mb_left_kc512_s701",
                "conditions": list(CONDITIONS), "n_shells": 6, "rewarded_shell_count": 3,
                "reward_probability": .8, "head_lr": .1, "stream_namespace": NAMESPACE,
                "gram_condition_max": 1e8, "geometry_tolerance": 1e-9,
                "q_reference": {"alpha": .1, "beta": 4.},
                "criteria": {"acquisition_accuracy": .75, "acquisition_gain": .15,
                             "seed_pass_fraction": .8, "endpoint_effect": .1, "positive_seed_fraction": .8}}
    for key, value in expected.items():
        core.demand(cfg.get(key) == value, f"Registered scientific setting changed: {key}")
    seeds = cfg.get("seeds", [])
    core.demand(bool(seeds) and len(set(seeds)) == len(seeds)
                and all(type(s) is int and s >= 0 for s in seeds), "Canonical unique seed axis")
    count, every = cfg.get("exposures_per_shell"), cfg.get("checkpoint_every_exposures")
    core.demand(type(count) is int and type(every) is int and count >= 6 and every > 0 and count % every == 0,
                "Complete exposure/checkpoint blocks required")
    if allow_fixture and cfg.get("scope") == "integration_test":
        core.demand(not set(seeds) & set(range(15001, 15006)), "Scientific seeds forbidden in fixtures")
    else:
        core.demand(cfg.get("scope") == "preregistered_stationary_credit_diagnostic"
                    and seeds == list(range(15001, 15006)) and (count, every) == (120, 10),
                    "Fixed five-seed prospective budget required")


def stream(seed, component):
    core.demand(type(seed) is int and seed >= 0 and type(component) is int and component in range(4),
                "Canonical independent RNG address required")
    return np.random.default_rng(np.random.SeedSequence([NAMESPACE, seed, component]))


def trial_plan(seed, exposures):
    core.demand(type(exposures) is int and exposures > 0, "Positive complete exposure blocks required")
    labels = np.zeros(6, dtype=int)
    labels[stream(seed, 0).permutation(6)[:3]] = 1
    schedule_rng = stream(seed, 1)
    shells = np.concatenate([schedule_rng.permutation(np.arange(1, 7)) for _ in range(exposures)])
    outcome_u = stream(seed, 2).random(len(shells))
    action_u = stream(seed, 3).random(len(shells))
    potential = np.where(outcome_u < .8, 2 * labels[shells - 1] - 1, 0)
    return {"target_go": labels, "shells": shells, "potential_rewards": potential,
            "outcome_uniforms": outcome_u, "action_uniforms": action_u}


def geometry(feature_gaps):
    """Use the SVD identity for dual rows, independent of the producer's solve."""
    d = np.asarray(feature_gaps, dtype=float)
    core.demand(d.ndim == 2 and d.shape[0] == 6 and d.shape[1] >= 6 and np.isfinite(d).all(),
                "Six finite cue feature vectors required")
    u, singular, vt = np.linalg.svd(d, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * max(d.shape) * np.finfo(float).eps))
    gram = d @ d.T
    condition = float(np.linalg.cond(gram))
    core.demand(rank == 6 and np.isfinite(condition) and condition <= 1e8,
                "Registered rank/Gram-condition gate failed")
    dual = (u / singular[None, :]) @ vt
    residual = float(np.max(np.abs(d @ dual.T - np.eye(6))))
    core.demand(residual <= 1e-9, "Independent dual residual exceeds fixed tolerance")
    return {"gram": gram, "dual_rows": dual, "rank": rank, "gram_condition": condition,
            "dual_residual_max": residual}


def influence_matrices(d, dual, rate):
    """Actual zero-head Go updates, signs [-1,+1], source cue then target cue."""
    d, dual = np.asarray(d), np.asarray(dual)
    shared, isolated, q = [], [], []
    for reward in (-1, 1):
        shared_delta = rate * .5 * reward * d
        isolated_delta = rate * .5 * reward * np.sum(d * d, axis=1)[:, None] * dual
        shared.append(shared_delta @ d.T)
        isolated.append(isolated_delta @ d.T)
        q.append(np.eye(6) * 4 * .1 * reward)
    shared, isolated, q = np.array(shared), np.array(isolated), np.array(q)
    off = ~np.eye(6, dtype=bool)
    core.demand(np.max(np.abs(isolated[:, off])) <= 1e-9, "Isolated update changes other cues")
    core.array_close(np.diagonal(shared, axis1=1, axis2=2), np.diagonal(isolated, axis1=1, axis2=2),
                     "Matched own-logit update at identical probability", atol=1e-9)
    return shared, isolated, q


def replay(d, dual, plan, condition, rate, checkpoints):
    core.demand(condition in CONDITIONS, "Unknown credit condition")
    d, dual = np.asarray(d), np.asarray(dual)
    n = len(plan["shells"])
    head = np.zeros(d.shape[1])
    result = {key: np.empty(n, dtype=float) for key in
              ("probabilities", "actions", "rewards", "head_steps", "internal_steps", "head_update_calls", "internal_update_calls")}
    saved_heads, probes = [], []
    checkpoint_set = set(int(t) for t in checkpoints)
    core.demand(0 in checkpoint_set and n in checkpoint_set and len(checkpoint_set) == len(checkpoints),
                "Unique baseline and endpoint checkpoints required")
    def save():
        saved_heads.append(head.copy())
        probes.append(np.array([core.logistic(value) for value in d @ head]))
    save()
    exposure = np.zeros(6, dtype=int)
    rows = []
    for index, shell_value in enumerate(plan["shells"]):
        shell = int(shell_value); cue = shell - 1
        p = core.logistic(d[cue] @ head)
        action = int(plan["action_uniforms"][index] < p)
        reward = int(plan["potential_rewards"][index]) if action else 0
        old = head.copy()
        if condition != "uniform":
            factor = rate * (action - p) * reward
            direction = d[cue] if condition == "shared" else float(d[cue] @ d[cue]) * dual[cue]
            head = head + factor * direction
        step = float(np.linalg.norm(head - old))
        for field, value in (("probabilities", p), ("actions", action), ("rewards", reward), ("head_steps", step),
                             ("internal_steps", 0.), ("head_update_calls", int(condition != "uniform")), ("internal_update_calls", 0)):
            result[field][index] = value
        exposure[cue] += 1
        correct_p = p if plan["target_go"][cue] else 1 - p
        rows.append({"condition": condition, "trial": index + 1, "shell": shell, "exposure": int(exposure[cue]),
                     "target_go": int(plan["target_go"][cue]), "action": action, "reward": reward,
                     "p_go": p, "p_correct": correct_p, "head_step_l2": step, "internal_step_l2": 0.,
                     "head_update_called": int(condition != "uniform"), "internal_update_called": 0})
        if index + 1 in checkpoint_set:
            save()
    result["heads"], result["probe_probabilities"] = np.array(saved_heads), np.array(probes)
    return result, rows


def expected_artifacts(cfg, graph):
    """Construct every recorded quantity with independent arithmetic and RNG."""
    seeds = cfg["seeds"]
    n = cfg["exposures_per_shell"] * 6
    checkpoints = np.arange(0, n + 1, cfg["checkpoint_every_exposures"] * 6)
    expected = {"seeds": np.asarray(seeds), "checkpoints": checkpoints,
                "pre": graph["pre"], "post": graph["post"]}
    accumulated = {}
    rows = []
    # All source gates precede all task/action draws, as in the registered order.
    witnesses, geometries = [], []
    for seed in seeds:
        witness = core.AuditPolicy("fixed", seed, cfg["head_lr"], graph).witness()
        gaps = witness["probe_decision"][:, 1] - witness["probe_decision"][:, 0]
        geom = geometry(gaps)
        signed_influence = influence_matrices(gaps, geom["dual_rows"], cfg["head_lr"])
        witnesses.append(witness); geometries.append((gaps, geom, signed_influence))
    for seed, witness, (gaps, geom, signed_influence) in zip(seeds, witnesses, geometries):
        one = {**trial_plan(seed, cfg["exposures_per_shell"]), "feature_gaps": gaps, **geom}
        for key, source in (("encoder", "encoder"), ("bias", "bias"), ("feature_scale", "feature_scale"),
                            ("weights_initial", "weights"), ("kc_initial", "probe_kc"),
                            ("mbon_initial", "probe_mbon"), ("decision_initial", "probe_decision")):
            one[key] = witness[source]
        for key, value in zip(("influence_shared", "influence_isolated", "influence_q_reference"),
                              signed_influence):
            one[key] = value
        trajectories = []
        for condition in CONDITIONS:
            trajectory, ledger = replay(gaps, geom["dual_rows"], one, condition, cfg["head_lr"], checkpoints)
            trajectories.append(trajectory)
            rows.extend({"seed": seed, **row} for row in ledger)
        for key in trajectories[0]:
            one[key] = np.stack([trajectory[key] for trajectory in trajectories])
        for key, source in (("weights_final", "weights_initial"), ("kc_final", "kc_initial"),
                            ("mbon_final", "mbon_initial"), ("decision_final", "decision_initial")):
            one[key] = np.stack([one[source]] * len(CONDITIONS))
        for key, value in one.items():
            accumulated.setdefault(key, []).append(value)
    expected.update({key: np.stack(value) for key, value in accumulated.items()})
    return expected, rows


def expected_summary(cfg, a):
    """Recompute decisions, counts and resource declarations without producer code."""
    outcomes, acquisition, effects, static = [], [], [], []
    c = cfg["criteria"]
    for si, seed in enumerate(cfg["seeds"]):
        target = a["target_go"][si]
        final = []
        for ci, condition in enumerate(CONDITIONS):
            correct = np.where(target[None, :] == 1, a["probe_probabilities"][si, ci],
                               1 - a["probe_probabilities"][si, ci]).mean(axis=1)
            before, after = float(correct[0]), float(correct[-1]); final.append(after)
            row = {"seed": seed, "condition": condition, "before_accuracy": before, "after_accuracy": after,
                   "gain": after - before,
                   "acquired": bool(after >= c["acquisition_accuracy"] and after - before >= c["acquisition_gain"]),
                   "head_update_calls": int(a["head_update_calls"][si, ci].sum()),
                   "internal_update_calls": int(a["internal_update_calls"][si, ci].sum()),
                   "head_nonzero_steps": int(np.count_nonzero(a["head_steps"][si, ci])),
                   "internal_nonzero_steps": int(np.count_nonzero(a["internal_steps"][si, ci]))}
            row["head_change_l2"] = float(np.linalg.norm(a["heads"][si, ci, -1] - a["heads"][si, ci, 0]))
            for label, end, begin in (("internal", "weights_final", "weights_initial"),
                                       ("kc", "kc_final", "kc_initial"), ("mbon", "mbon_final", "mbon_initial"),
                                       ("decision_feature", "decision_final", "decision_initial")):
                row[label + "_change_l2"] = float(np.linalg.norm(a[end][si, ci] - a[begin][si]))
            outcomes.append(row)
        effects.append({"seed": seed, "isolated_minus_shared_endpoint": final[1] - final[0]})
        shared, isolated, q = [a[key][si, 1] for key in
                               ("influence_shared", "influence_isolated", "influence_q_reference")]
        off = ~np.eye(6, dtype=bool)
        static.append({"seed": seed, "rank": int(a["rank"][si]),
                       "gram_condition": float(a["gram_condition"][si]),
                       "dual_residual_max": float(a["dual_residual_max"][si]),
                       "matched_own_logit_error_max": float(np.abs(np.diag(shared) - np.diag(isolated)).max()),
                       "shared_mean_abs_offdiagonal": float(np.abs(shared[off]).mean()),
                       "isolated_max_abs_offdiagonal": float(np.abs(isolated[off]).max()),
                       "actor_mean_initial_own_logit_step": float(np.diag(shared).mean()),
                       "q_reference_initial_own_logit_step": float(np.diag(q).mean()),
                       "q_reference_over_actor_mean_step": float(np.diag(q).mean() / np.diag(shared).mean())})
    for condition in CONDITIONS:
        chosen = [item for item in outcomes if item["condition"] == condition]
        passed = sum(item["acquired"] for item in chosen)
        acquisition.append({"condition": condition, "acquired_seeds": passed, "n_seeds": len(chosen),
                            "criterion_met": passed / len(chosen) >= c["seed_pass_fraction"],
                            "mean_endpoint_accuracy": float(np.mean([item["after_accuracy"] for item in chosen]))})
    differences = [item["isolated_minus_shared_endpoint"] for item in effects]
    mean_effect = float(np.mean(differences)); positive = sum(value > 0 for value in differences)
    effect_pass = mean_effect >= c["endpoint_effect"] and positive / len(differences) >= c["positive_seed_fraction"]
    isolated_pass = acquisition[1]["criterion_met"]
    m = a["heads"].shape[-1]
    return {"status": "completed", "study_id": cfg["study_id"],
            "counts": {"actual_choice_feedback_trials": len(cfg["seeds"]) * 3 * cfg["exposures_per_shell"] * 6,
                       "head_update_calls": int(a["head_update_calls"].sum()),
                       "internal_update_calls": int(a["internal_update_calls"].sum()), "new_human_observations": 0,
                       "static_signed_source_cases": len(cfg["seeds"]) * 2 * 6,
                       "computational_seed_units": len(cfg["seeds"])},
            "outcomes": outcomes, "acquisition": acquisition,
            "paired_effects": effects, "static_geometry": static,
            "H_credit": {"effect_criterion_met": bool(effect_pass), "mean_endpoint_effect": mean_effect,
                         "positive_seeds": positive, "n_seeds": len(differences),
                         "isolated_acquisition_gate": bool(isolated_pass), "supported": bool(effect_pass and isolated_pass),
                         "interpretation": "stationary computational cross-cue update intervention; not the cause of the historical human-task failure"},
            "resources": {"actual_choice_feedback_trials": len(cfg["seeds"]) * 3 * cfg["exposures_per_shell"] * 6,
                          "head_update_calls": int(a["head_update_calls"].sum()),
                          "internal_update_calls": int(a["internal_update_calls"].sum()),
                          "head_coordinates_per_model": m,
                          "active_feature_coordinates": int(np.any(a["feature_gaps"] != 0, axis=(0, 1)).sum()),
                          "trainable_head_coordinates": {"shared": m, "isolated": m, "uniform": 0},
                          "isolated_extra_fixed_coefficients": 6 * m,
                          "isolated_precomputation": "six-by-six Gram solve; not an anatomical connection or trainable biological parameter",
                          "equal_update_l2_not_claimed": True, "same_local_own_logit_step": True,
                          "diverged_autonomous_paths_have_equal_steps_not_claimed": True},
            "acquisition_scope": "six stationary identity cues; not reversal, held-out human validation, or mathematical generalization",
            "historical_results_replaced": False, "human_similarity_tested": False,
            "biological_mechanism_tested": False, "calibration_objective_causal_effect_tested": False}


def verify_against_graph(cfg, arrays, rows, summary, graph, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    expected, ledger = expected_artifacts(cfg, graph)
    core.demand(set(arrays) == set(expected), "Complete numeric artifact inventory differs")
    integer_fields = {"seeds", "checkpoints", "pre", "post", "shells", "target_go", "potential_rewards", "rank",
                      "actions", "rewards", "head_update_calls", "internal_update_calls"}
    for field, value in expected.items():
        observed = np.asarray(arrays[field])
        core.demand(observed.dtype.kind in "biuf" and np.isfinite(observed).all(), f"Nonfinite/non-numeric {field}")
        if field in integer_fields:
            core.demand(observed.dtype.kind in "biu" and np.array_equal(observed, value), f"Exact integer ledger {field}")
        else:
            core.array_close(observed, value, f"Independent full replay: {field}", atol=1e-9)
    core.demand(len(rows) == len(ledger), "Full trial row count")
    for index, (actual, expected_row) in enumerate(zip(rows, ledger)):
        if all(isinstance(v, str) for v in actual.values()):
            core.check_csv_row(actual, expected_row, f"Trial ledger {index}")
        else:
            core.compare_tree(actual, expected_row, f"Trial ledger {index}")
    # Decisions are recomputed from the submitted verified arrays to avoid letting
    # harmless solver rounding change a boundary decision in the audit itself.
    core.compare_tree(summary, expected_summary(cfg, arrays), "Complete independently computed summary")
    total = len(ledger)
    return {"arrays": len(expected), "trial_records": total, "independently_replayed_actions": total,
            "seeds": len(cfg["seeds"]), "conditions": len(CONDITIONS),
            "head_update_calls": int(expected["head_update_calls"].sum()), "internal_update_calls": 0,
            "initial_influence_entries": len(cfg["seeds"]) * 3 * 2 * 6 * 6,
            "summary_and_resource_fields_checked": True, "no_hidden_potential_content_updates": True,
            "all_anatomical_weights_and_feature_activations_fixed": True}


def verify_science(cfg, root, arrays, rows, summary, cache=None):
    validate_science(cfg)
    graph = core.raw_graph(Path(root) / cfg["graph"])
    core.demand((graph["n_kc"], graph["n_mbon"], len(graph["pre"])) == (512, 48, 1474), "Frozen partial graph scope")
    return verify_against_graph(cfg, arrays, rows, summary, graph)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(args.root.resolve() / "src"))
    from drosolearn.prospective import verify_envelope, sha, write_new
    cfg, reg, metadata = verify_envelope(args.root, args.config, args.registration, args.result_dir, args.cache)
    core.demand(metadata["study_id"] == cfg["study_id"] and metadata["stage"] == "completed",
                "Completed metadata study identity")
    core.demand(all(metadata.get(key) is False for key in
                    ("new_human_experiment", "shared_neural_mechanism_tested", "education_effect_tested")),
                "Metadata interpretation boundary changed")
    stamps = [datetime.fromisoformat(value) for value in (reg["registered_at_utc"], metadata["started_at_utc"], metadata["completed_at_utc"])]
    core.demand(all(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0 for stamp in stamps)
                and stamps[0] < stamps[1] <= stamps[2] <= datetime.now(timezone.utc), "Prospective UTC chronology")
    with np.load(args.result_dir / "arrays.npz", allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    with (args.result_dir / "measurements.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        core.demand(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)), "CSV unique header")
        rows = list(reader)
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf-8"))
    counts = verify_science(cfg, args.root, arrays, rows, summary, args.cache)
    core.demand(metadata.get("counts", {}) == summary.get("counts", {}), "Metadata count binding")
    report = {"status": "passed", "study_id": cfg["study_id"],
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "registration_sha256": sha(args.registration), "summary_sha256": sha(args.result_dir / "summary.json"),
              "scientific_producer_imported": False, "counts": counts}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
