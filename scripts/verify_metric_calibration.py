"""Independent finite-state and sampled-path replay of no-learning policies.

This verifier never imports the calibration producer or the primary transition
metric. Expected counts are sums of joint probabilities; their ratio is not an
assertion about the expectation of an empirical, random-denominator ratio.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "configs/metric_calibration_v1.json"
SCOPE = "synthetic_no_learning_policy_calibration_not_species_comparison"
POLICIES = ("iid_fair", "iid_biased", "sticky")
SCHEDULES = ("fixed", "reversal")
ENVIRONMENTS = {
    "deterministic": (1., 0.),
    "probabilistic": (.8, .2),
    "uninformative": (.5, .5),
    "swapped": (.2, .8),
}
TRIAL_FIELDS = ("stream_id", "seed", "schedule", "policy", "environment", "trial", "phase", "state",
                "action", "reward", "action_uniform", "reward_uniform")
LEDGER_FIELDS = ("stream_id", "source_position", "next_source_position", "trial", "next_trial",
                 "eligible", "reasons", "feedback", "stay")
FROZEN_PATHS = {
    "src/drosolearn/__init__.py", "src/drosolearn/feedback_transition.py", "src/drosolearn/metric_calibration.py",
    "scripts/run_metric_calibration.py", "scripts/verify_metric_calibration.py",
    "tests/test_metric_calibration.py", "tests/test_metric_calibration_run.py", "tests/test_metric_calibration_verify.py",
    CONFIG_PATH, "docs/metric-calibration-protocol.md", "requirements-lock.txt",
}
PARENT_PATHS = ["configs/feedback_transition_v1.json", "docs/feedback-transition-protocol.md",
                "results/feedback_transition_v1/verification.json"]


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(VerificationError("Nonfinite JSON number")))


def exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(exact(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(exact(a, b) for a, b in zip(actual, expected))
    return actual == expected


def close(actual, expected):
    """Exact structures and types, allowing only roundoff in analytic floats."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(close(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(close(a, b) for a, b in zip(actual, expected))
    if type(expected) is float:
        return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0., abs_tol=1e-12)
    return actual == expected


def safe(root, relative):
    require(type(relative) is str and relative and "\\" not in relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "POSIX relative input path required")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Unsafe or missing input path")
    return path


def utc(value):
    require(type(value) is str, "UTC timestamp must be text")
    stamp = datetime.fromisoformat(value)
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0, "Timezone-aware UTC required")
    return stamp


def policy_law(policy):
    require(policy in POLICIES, "Unregistered nonlearning policy")
    if policy == "sticky":
        return [.5, .5], [[.9, .1], [.1, .9]]
    p = .5 if policy == "iid_fair" else .8
    return [1. - p, p], [[1. - p, p], [1. - p, p]]


def replay_stream(seed, schedule, policy, environment, trials, boundary):
    """Regenerate actual choices and actual feedback from two independent streams."""
    require(type(seed) is int and seed >= 0 and type(trials) is int and type(boundary) is int
            and 0 < boundary < trials, "Fixed integer seed and two nonempty phases required")
    require(schedule in SCHEDULES and policy in POLICIES and environment in ENVIRONMENTS,
            "Unknown registered calibration coordinate")
    actions = np.random.default_rng(seed + 410000).random(trials)
    rewards = np.random.default_rng(seed + 420000).random(trials)
    stream_id = f"{seed}/{schedule}/{policy}/{environment}"
    records = []
    match, other = ENVIRONMENTS[environment]
    previous = None
    for index, (u, v) in enumerate(zip(actions, rewards, strict=True)):
        if policy == "sticky":
            action = int(u < .5) if previous is None else previous if u < .9 else 1 - previous
        else:
            action = int(u < (.5 if policy == "iid_fair" else .8))
        state = int(schedule == "reversal" and index >= boundary)
        reward = int(v < (match if action == state else other))
        records.append({"stream_id": stream_id, "seed": seed, "schedule": schedule, "policy": policy,
                        "environment": environment, "trial": index + 1,
                        "phase": "phase1" if index < boundary else "phase2", "state": state,
                        "action": action, "reward": reward,
                        "action_uniform": float(u), "reward_uniform": float(v)})
        previous = action
    return records


def independent_counts(records, *, stratum="all", kind="synthetic_no_learning", coordinate="stable_option"):
    """Keep every original adjacent pair, marking phase/state edges explicitly."""
    require(stratum in ("all", "phase1", "phase2") and isinstance(records, list) and records,
            "Named nonempty calibration stratum required")
    chosen = records if stratum == "all" else [record for record in records if record["phase"] == stratum]
    require(chosen, "Both preregistered phases must remain nonempty")
    stream_id = records[0]["stream_id"]
    counts = {str(feedback): {"denominator": 0, "stay": 0, "switch": 0} for feedback in (0, 1)}
    ledger, reasons_count = [], Counter()
    for current, following in zip(chosen, chosen[1:]):
        reasons = []
        if current["phase"] != following["phase"]:
            reasons.append("phase_boundary")
        if current["state"] != following["state"]:
            reasons.append("state_boundary")
        reasons.sort()
        feedback = current["reward"]
        require(type(feedback) is int and feedback in (0, 1), "Actual binary reward required")
        eligible = not reasons
        stayed = int(current["action"] == following["action"]) if eligible else None
        if eligible:
            cell = counts[str(feedback)]
            cell["denominator"] += 1
            cell["stay"] += stayed
            cell["switch"] += 1 - stayed
        reasons_count.update(reasons)
        ledger.append({"stream_id": stream_id,
                       "source_position": str(current["trial"]), "next_source_position": str(following["trial"]),
                       "trial": str(current["trial"]), "next_trial": str(following["trial"]),
                       "eligible": str(int(eligible)), "reasons": "|".join(reasons),
                       "feedback": str(feedback), "stay": str(stayed) if stayed is not None else ""})
    for cell in counts.values():
        denominator = cell["denominator"]
        cell["stay_rate"] = cell["stay"] / denominator if denominator else None
        cell["switch_rate"] = cell["switch"] / denominator if denominator else None
    eligible = sum(cell["denominator"] for cell in counts.values())
    summary = {"stream_id": stream_id, "kind": kind, "coordinate": coordinate, "trials": len(chosen),
               "candidates": len(ledger), "eligible": eligible, "excluded": len(ledger) - eligible,
               "reason_counts": dict(sorted(reasons_count.items())), "by_feedback": counts}
    return summary, ledger


def expected_counts(records, policy, environment, *, stratum="all"):
    """Propagate the two-state action law and sum eligible joint probabilities."""
    require(stratum in ("all", "phase1", "phase2") and records, "Known nonempty analytic stratum required")
    require(environment in ENVIRONMENTS, "Known reward environment required")
    require(policy in POLICIES, "Known nonlearning policy required")
    if policy == "sticky":
        marginal = [Fraction(1, 2), Fraction(1, 2)]
        transition = [[Fraction(9, 10), Fraction(1, 10)], [Fraction(1, 10), Fraction(9, 10)]]
    else:
        p1 = Fraction(1, 2) if policy == "iid_fair" else Fraction(4, 5)
        marginal = [1 - p1, p1]
        transition = [[1 - p1, p1], [1 - p1, p1]]
    match, other = (Fraction(str(value)) for value in ENVIRONMENTS[environment])
    accum = {str(feedback): {"expected_denominator": Fraction(0), "expected_stay": Fraction(0),
                            "expected_switch": Fraction(0)}
             for feedback in (0, 1)}
    for current, following in zip(records, records[1:]):
        included = (stratum == "all" or current["phase"] == following["phase"] == stratum)
        included = included and current["phase"] == following["phase"] and current["state"] == following["state"]
        if included:
            for action in (0, 1):
                q = match if action == current["state"] else other
                for feedback, outcome_probability in ((0, 1 - q), (1, q)):
                    joint = marginal[action] * outcome_probability
                    cell = accum[str(feedback)]
                    cell["expected_denominator"] += joint
                    cell["expected_stay"] += joint * transition[action][action]
                    cell["expected_switch"] += joint * transition[action][1 - action]
        marginal = [sum(marginal[action] * transition[action][next_action] for action in (0, 1))
                    for next_action in (0, 1)]
    for cell in accum.values():
        denominator = cell["expected_denominator"]
        cell["ratio_of_expected_counts"] = cell["expected_stay"] / denominator if denominator else None
        for key, value in cell.items():
            if value is not None:
                cell[key] = float(value)
    return {"by_feedback": accum}


def checked_config(root, config_path):
    path = Path(config_path)
    if not path.is_absolute():
        path = safe(root, config_path)
    require(path.resolve().is_relative_to(root) and path.is_file(), "Config must stay within the project")
    cfg = read_json(path)
    require(cfg["study_id"] == "metric_calibration_v1" and cfg["scope"] in (SCOPE, "integration_test"),
            "Registered calibration identity differs")
    n, seeds = cfg["trials"], cfg["seeds"]
    require(type(n) is int and n >= 4 and n % 4 == 0 and type(seeds) is list and seeds
            and all(type(seed) is int and seed >= 0 for seed in seeds) and len(set(seeds)) == len(seeds),
            "Unique integer seeds and complete equal phases required")
    if cfg["scope"] == SCOPE:
        require(n == 800 and seeds == list(range(9001, 9011)), "Prospective scientific budget differs")
    else:
        require(not set(seeds) & set(range(9001, 9011)), "Scientific seeds cannot be synthetic fixtures")
    expected_policies = [{"id": "iid_fair", "kind": "iid", "p1": .5},
                         {"id": "iid_biased", "kind": "iid", "p1": .8},
                         {"id": "sticky", "kind": "sticky", "initial_p1": .5, "p_stay": .9}]
    expected_environments = [{"id": name, "reward1_given_choice_state0": list(ENVIRONMENTS[name])}
                             for name in ("probabilistic", "deterministic", "uninformative", "swapped")]
    require(exact(cfg["policies"], expected_policies) and exact(cfg["environments"], expected_environments),
            "Fixed nonlearning laws or reward probabilities differ")
    require(exact(cfg["report_segments"], [{"id": "phase1", "start_trial": 1, "end_trial": n // 2},
                                          {"id": "phase2", "start_trial": n // 2 + 1, "end_trial": n}])
            and exact(cfg["schedules"], [{"id": "fixed", "state0_end_trial": n},
                                         {"id": "reversal", "state0_end_trial": n // 2}]),
            "Original phase partition or state schedules differ")
    require(exact(cfg["rng"], {"implementation": "numpy.random.default_rng", "action_seed_offset": 410000,
                              "feedback_seed_offset": 420000, "array_length": n,
                              "coupling": "same_action_and_feedback_arrays_per_seed_reused_across_all_policies_environments_schedules"}),
            "Independent/shared RNG law differs")
    for key in ("learning", "policy_reset_at_reversal", "monte_carlo_inference", "pool_species_or_conditions"):
        require(cfg[key] is False, "No-learning or no-species scope changed")
    require(cfg["all_registered_runs_retained"] is True and cfg["state1_rule"] == "swap_choice_reward1_probabilities"
            and cfg["metric"] == "actual_original_adjacent_choice_stay_switch_conditioned_on_actual_feedback_at_t"
            and exact(cfg["exclude_boundaries"], ["state", "phase", "task"]) and cfg["zero_denominator"] is None
            and type(cfg["analytic_tolerance"]) is float and cfg["analytic_tolerance"] == 1e-12
            and cfg["analytic_primary_target"] == "ratio_of_expected_stay_count_to_expected_denominator_not_mean_of_random_ratios"
            and cfg["segment_policy"] == "same_phase_boundary_after_trial400_in_both_schedules_all_and_each_phase_reported",
            "Fixed denominator, estimand or interpretation rule differs")
    require(type(cfg["frozen_files"]) is list and len(cfg["frozen_files"]) == len(FROZEN_PATHS)
            and set(cfg["frozen_files"]) == FROZEN_PATHS and exact(cfg["parent_files"], PARENT_PATHS),
            "Source or parent coverage differs")
    return cfg, path


def independent_summary(streams, cfg):
    cases = []
    for schedule in cfg["schedules"]:
        for policy in cfg["policies"]:
            for environment in cfg["environments"]:
                group = [stream for stream in streams if (stream["schedule"], stream["policy"], stream["environment"]) ==
                         (schedule["id"], policy["id"], environment["id"])]
                require([stream["seed"] for stream in group] == cfg["seeds"], "Every seed must remain in each summary")
                for stratum in ("all", "phase1", "phase2"):
                    feedback = {}
                    for outcome in ("0", "1"):
                        cells = [stream["strata"][stratum]["empirical"]["by_feedback"][outcome] for stream in group]
                        denominator, stayed = sum(cell["denominator"] for cell in cells), sum(cell["stay"] for cell in cells)
                        ratios = [cell["stay_rate"] for cell in cells if cell["stay_rate"] is not None]
                        feedback[outcome] = {
                            "summed_denominator": denominator, "summed_stay": stayed, "summed_switch": denominator - stayed,
                            "ratio_of_summed_empirical_counts": stayed / denominator if denominator else None,
                            "mean_defined_seed_ratio": float(statistics.mean(ratios)) if ratios else None,
                            "sd_defined_seed_ratio": float(statistics.stdev(ratios)) if len(ratios) > 1 else None,
                            "defined_seeds": len(ratios),
                            "ratio_of_expected_counts": group[0]["strata"][stratum]["analytic"]["by_feedback"][outcome]["ratio_of_expected_counts"],
                        }
                    cases.append({"schedule": schedule["id"], "policy": policy["id"], "environment": environment["id"],
                                  "stratum": stratum, "by_feedback": feedback})
    def contrast(schedule, stratum):
        item = next(case for case in cases if (case["schedule"], case["policy"], case["environment"], case["stratum"]) ==
                    (schedule, "iid_biased", "probabilistic", stratum))
        return item["by_feedback"]["1"]["ratio_of_expected_counts"] - item["by_feedback"]["0"]["ratio_of_expected_counts"]
    first_bias = contrast("fixed", "all")
    sticky_errors = [abs(case["by_feedback"][feedback]["ratio_of_expected_counts"] - .9)
                     for case in cases if case["policy"] == "sticky" for feedback in ("0", "1")]
    pooled, phase1, phase2 = (contrast("reversal", stratum) for stratum in ("all", "phase1", "phase2"))
    tolerance = cfg["analytic_tolerance"]
    claims = {
        "H1_no_learning_feedback_difference": {"analytic_difference": first_bias, "reference": -9 / 34,
            "criterion_met": abs(first_bias + 9 / 34) <= tolerance and abs(first_bias) >= .2},
        "H2_no_learning_high_persistence": {"maximum_absolute_error_from_0_9": max(sticky_errors),
            "criterion_met": all(error <= tolerance for error in sticky_errors)},
        "H3_pooling_masks_phase_difference": {"pooled_difference": pooled, "phase1_difference": phase1,
            "phase2_difference": phase2, "criterion_met": abs(pooled) <= tolerance and phase1 <= -.2 and phase2 >= .2},
    }
    return {"study_id": cfg["study_id"], "cases": cases, "analytic_claims": claims,
            "parameter_updates": 0, "monte_carlo_exact_match_required": False,
            "scope": "No learning policy counterexamples; MC ratios described without confidence intervals or species inference"}


def verify(registration_path, result_dir, root=None, config_path=CONFIG_PATH):
    root = Path(root if root is not None else ROOT).resolve()
    cfg, config_file = checked_config(root, config_path)
    registration_path, result_dir = Path(registration_path).resolve(), Path(result_dir).resolve()
    reg = read_json(registration_path)
    require(reg["study_id"] == cfg["study_id"] and exact(reg["config"], cfg)
            and reg["config_sha256"] == digest(config_file), "Registered config identity differs")
    frozen = {path: digest(safe(root, path)) for path in sorted(FROZEN_PATHS)}
    parent_files = {path: digest(safe(root, path)) for path in PARENT_PATHS}
    require(exact(reg["frozen_files"], frozen) and exact(reg["parent_files"], parent_files),
            "Frozen code, tests, protocol or parent bytes changed")
    require(read_json(safe(root, PARENT_PATHS[-1]))["status"] == "passed", "Parent measurement gate is closed")
    require(type(reg["source_git_commit"]) is str and re.fullmatch(r"[a-f0-9]{40}", reg["source_git_commit"]),
            "Scientific source Git identity missing")
    copied_reg = safe(result_dir, "registration.json")
    require(copied_reg.read_bytes() == registration_path.read_bytes(), "Copied registration bytes differ")
    artifacts = read_json(safe(result_dir, "artifact_hashes.json"))
    expected_artifacts = {"registration.json", "metadata.json", "trials.csv", "ledger.csv", "metrics.json", "summary.json"}
    require(type(artifacts) is dict and set(artifacts) == expected_artifacts, "Primary artifact coverage differs")
    for path, checksum in artifacts.items():
        require(type(checksum) is str and re.fullmatch(r"[a-f0-9]{64}", checksum)
                and checksum == digest(safe(result_dir, path)), "Primary artifact bytes changed")
    metadata = read_json(safe(result_dir, "metadata.json"))
    require(metadata["status"] == "completed" and metadata["registration_sha256"] == digest(registration_path)
            and metadata["source_git_commit"] == reg["source_git_commit"]
            and metadata["config_sha256"] == reg["config_sha256"] and exact(metadata["config"], cfg)
            and exact(metadata["frozen_files"], frozen) and exact(metadata["parent_files"], parent_files),
            "Completed run source or scope differs")
    require(utc(reg["registered_at_utc"]) <= utc(metadata["started_at_utc"]) <= utc(metadata["completed_at_utc"]),
            "Source registration did not precede execution")
    require(type(metadata["runtime"]) is dict and set(metadata["runtime"]) == {"python", "numpy"}
            and all(type(value) is str and value for value in metadata["runtime"].values()), "Runtime provenance differs")
    metrics, summaries = read_json(safe(result_dir, "metrics.json")), []
    trial_count, candidate_count, eligible_count = 0, 0, 0
    with safe(result_dir, "trials.csv").open(encoding="utf-8", newline="") as trial_handle, \
            safe(result_dir, "ledger.csv").open(encoding="utf-8", newline="") as ledger_handle:
        trials_reader, ledger_reader = csv.reader(trial_handle), csv.reader(ledger_handle)
        require(next(trials_reader, None) == list(TRIAL_FIELDS) and next(ledger_reader, None) == list(LEDGER_FIELDS),
                "Actual trial or original-adjacency ledger schema differs")
        for seed in cfg["seeds"]:
            for schedule in cfg["schedules"]:
                for policy in cfg["policies"]:
                    for environment in cfg["environments"]:
                        records = replay_stream(seed, schedule["id"], policy["id"], environment["id"], cfg["trials"], cfg["trials"] // 2)
                        for record in records:
                            require(next(trials_reader, None) == [str(record[field]) for field in TRIAL_FIELDS],
                                    "Independent registered RNG, nonlearning action or actual reward replay differs")
                            trial_count += 1
                        strata = {}
                        for stratum in ("all", "phase1", "phase2"):
                            empirical, ledger = independent_counts(records, stratum=stratum)
                            strata[stratum] = {"empirical": empirical,
                                               "analytic": expected_counts(records, policy["id"], environment["id"], stratum=stratum)}
                            if stratum == "all":
                                candidate_count += empirical["candidates"]
                                eligible_count += empirical["eligible"]
                                for pair in ledger:
                                    require(next(ledger_reader, None) == [pair[field] for field in LEDGER_FIELDS],
                                            "Independent original-adjacency ledger or boundary exclusion differs")
                        for feedback in ("0", "1"):
                            require(all(strata["all"]["empirical"]["by_feedback"][feedback][key] ==
                                        sum(strata[phase]["empirical"]["by_feedback"][feedback][key] for phase in ("phase1", "phase2"))
                                        for key in ("denominator", "stay", "switch")), "Phase counts do not sum to all counts")
                        summaries.append({"seed": seed, "schedule": schedule["id"], "policy": policy["id"],
                                          "environment": environment["id"], "strata": strata})
        require(next(trials_reader, None) is None and next(ledger_reader, None) is None, "Extra actual trial or ledger rows")
    require(close(metrics, {"study_id": cfg["study_id"], "streams": summaries}),
            "Independent per-stream counts, analytic joint probabilities or stratum scope differs")
    summary = independent_summary(summaries, cfg)
    require(close(read_json(safe(result_dir, "summary.json")), summary),
            "Independent seed denominators, summary statistics or analytic claim criteria differ")
    expected_metadata = {
        "study_id": cfg["study_id"], "status": "completed", "started_at_utc": metadata["started_at_utc"],
        "config": cfg, "config_sha256": digest(config_file), "source_git_commit": reg["source_git_commit"],
        "registration_sha256": digest(registration_path), "frozen_files": frozen, "parent_files": parent_files,
        "parameter_updates": 0, "runtime": metadata["runtime"], "human_or_species_comparison": False,
        "completed_at_utc": metadata["completed_at_utc"], "streams": len(summaries), "trials": trial_count,
        "candidate_pairs": candidate_count, "eligible_pairs": eligible_count,
    }
    require(exact(metadata, expected_metadata), "Actual execution budget or no-learning interpretation differs")
    return {"status": "passed", "scope": "independent_no_learning_calibration", "registration_sha256": digest(registration_path),
            "config_sha256": digest(config_file), "streams_verified": len(summaries), "actual_trials_verified": trial_count,
            "candidate_pairs_verified": candidate_count, "eligible_pairs_verified": eligible_count,
            "strata_verified": 3 * len(summaries), "summary_cases_verified": len(summary["cases"]),
            "frozen_files_verified": len(frozen), "parent_files_verified": len(parent_files),
            "primary_artifacts_verified": len(artifacts), "rational_joint_probability_calculation": True,
            "parameter_updates": 0, "human_or_species_comparison": False, "errors": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Verification output already exists")
    try:
        result = verify(args.registration, args.result_dir, root=args.root, config_path=args.config)
    except Exception as error:
        result = {"status": "failed", "scope": "independent_no_learning_calibration",
                  "errors": [f"{type(error).__name__}: {error}"]}
    result["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "scope", "errors")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
