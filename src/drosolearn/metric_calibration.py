"""Prospective no-learning calibration of feedback-conditioned persistence."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

import numpy as np

from .feedback_transition import measure_stream

ROOT = Path(__file__).resolve().parents[2]
SCOPE = "synthetic_no_learning_policy_calibration_not_species_comparison"
TRIAL_FIELDS = ["stream_id", "seed", "schedule", "policy", "environment", "trial", "phase", "state",
                "action", "reward", "action_uniform", "reward_uniform"]
LEDGER_FIELDS = ["stream_id", "source_position", "next_source_position", "trial", "next_trial",
                 "eligible", "reasons", "feedback", "stay"]
POLICIES = [{"id": "iid_fair", "kind": "iid", "p1": .5},
            {"id": "iid_biased", "kind": "iid", "p1": .8},
            {"id": "sticky", "kind": "sticky", "initial_p1": .5, "p_stay": .9}]
ENVIRONMENTS = [{"id": name, "reward1_given_choice_state0": q} for name, q in
                (("probabilistic", [.8, .2]), ("deterministic", [1., 0.]),
                 ("uninformative", [.5, .5]), ("swapped", [.2, .8]))]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(root, name):
    require(type(name) is str and "\\" not in name and not Path(name).is_absolute()
            and ".." not in Path(name).parts, "Unsafe relative path")
    root = Path(root).resolve()
    path = (root / name).resolve()
    require(path.is_relative_to(root), "Path escapes root")
    return path


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def validate_config(cfg):
    require(cfg["study_id"] == "metric_calibration_v1" and cfg["scope"] in (SCOPE, "integration_test"), "Wrong calibration scope")
    n = cfg["trials"]
    require(type(n) is int and n >= 4 and n % 4 == 0, "Complete even phases required")
    seeds = cfg["seeds"]
    require(isinstance(seeds, list) and seeds and len(set(seeds)) == len(seeds)
            and all(type(s) is int and s >= 0 for s in seeds), "Invalid independent RNG seeds")
    require(cfg["policies"] == POLICIES and cfg["environments"] == ENVIRONMENTS, "Fixed no-learning policies and environments required")
    require(cfg["schedules"] == [{"id": "fixed", "state0_end_trial": n}, {"id": "reversal", "state0_end_trial": n // 2}], "Fixed paired schedules required")
    require(cfg["report_segments"] == [{"id": "phase1", "start_trial": 1, "end_trial": n // 2},
                                      {"id": "phase2", "start_trial": n // 2 + 1, "end_trial": n}], "Original phase partition required")
    require(cfg["rng"]["action_seed_offset"] == 410000 and cfg["rng"]["feedback_seed_offset"] == 420000
            and cfg["rng"]["array_length"] == n and cfg["rng"]["implementation"] == "numpy.random.default_rng", "Fixed shared RNG rule required")
    require(cfg["learning"] is False and cfg["policy_reset_at_reversal"] is False
            and cfg["monte_carlo_inference"] is False and cfg["pool_species_or_conditions"] is False,
            "No adaptation or species inference is allowed")
    require(cfg["exclude_boundaries"] == ["state", "phase", "task"] and cfg["zero_denominator"] is None
            and cfg["analytic_tolerance"] == 1e-12, "Fixed metric boundary/zero rules required")
    require(len(set(cfg["frozen_files"])) == len(cfg["frozen_files"]), "Repeated frozen source")
    if cfg["scope"] == SCOPE:
        require(n == 800 and seeds == list(range(9001, 9011)), "Fixed prospective800-trial ten-seed budget required")
    else:
        require(not set(seeds) & set(range(9001, 9011)), "Scientific seeds cannot be development fixtures")


def freeze(root, cfg):
    return {name: digest(safe(root, name)) for name in sorted(cfg["frozen_files"])}


def parents(root, cfg):
    proof = json.loads(safe(root, "results/feedback_transition_v1/verification.json").read_text(encoding="utf8"))
    require(proof["status"] == "passed", "Verified parent measurement required")
    return {name: digest(safe(root, name)) for name in cfg["parent_files"]}


def register(root, config_path, out):
    cfg = json.loads(Path(config_path).read_text(encoding="utf8"))
    validate_config(cfg)
    record = {"study_id": cfg["study_id"], "registered_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "config": cfg, "config_sha256": digest(config_path), "frozen_files": freeze(root, cfg),
              "parent_files": parents(root, cfg),
              "notice": "Prospective private Git freeze of synthetic calibration; no blinded human or species hypothesis"}
    write_new(out, record)
    return record


def policy_actions(policy, uniforms):
    u = np.asarray(uniforms)
    require(u.ndim == 1 and len(u) > 0 and u.dtype.kind == "f" and np.isfinite(u).all()
            and ((u >= 0) & (u < 1)).all(), "Finite original uniform array required")
    require(policy in POLICIES, "Unknown fixed policy")
    if policy["kind"] == "iid":
        return (u < policy["p1"]).astype(int)
    action = np.empty(len(u), dtype=int)
    action[0] = int(u[0] < policy["initial_p1"])
    for i in range(1, len(u)):
        action[i] = action[i - 1] if u[i] < policy["p_stay"] else 1 - action[i - 1]
    return action


def simulate(cfg, seed, schedule, policy, environment):
    validate_config(cfg)
    require(seed in cfg["seeds"] and schedule in cfg["schedules"] and policy in cfg["policies"]
            and environment in cfg["environments"], "Unregistered simulation cell")
    n = cfg["trials"]
    u = np.random.default_rng(seed + cfg["rng"]["action_seed_offset"]).random(n)
    v = np.random.default_rng(seed + cfg["rng"]["feedback_seed_offset"]).random(n)
    actions = policy_actions(policy, u)
    stream = f"{seed}/{schedule['id']}/{policy['id']}/{environment['id']}"
    rows = []
    for i in range(n):
        state = int(i >= schedule["state0_end_trial"])
        q = environment["reward1_given_choice_state0"][int(actions[i]) ^ state]
        reward = int(v[i] < q)
        rows.append({"stream_id": stream, "kind": "synthetic_no_learning", "coordinate": "stable_option",
                     "source_position": i + 1, "trial": i + 1, "task": "synthetic_feedback",
                     "phase": "phase1" if i < n // 2 else "phase2", "state": str(state),
                     "choice": int(actions[i]), "feedback": reward, "actual": True, "training": True,
                     "seed": seed, "schedule": schedule["id"], "policy": policy["id"], "environment": environment["id"],
                     "action_uniform": float(u[i]), "reward_uniform": float(v[i])})
    return rows


def analytic(rows, policy, environment):
    """Ratio of expected counts under stationary actions, not expectation of ratio."""
    require(rows and policy in POLICIES and environment in ENVIRONMENTS, "Known analytic policy and rows required")
    if policy["kind"] == "iid":
        pi = np.array([1 - policy["p1"], policy["p1"]])
        stay = pi.copy()
    else:
        pi = np.array([.5, .5])
        stay = np.array([policy["p_stay"], policy["p_stay"]])
    denominators, numerators = np.zeros(2), np.zeros(2)
    eligible_by_state = {"0": 0, "1": 0}
    for current, following in zip(rows, rows[1:]):
        if any(current[k] != following[k] for k in ("state", "phase", "task")) or following["trial"] != current["trial"] + 1:
            continue
        require(current["state"] in eligible_by_state, "Unknown reference state")
        eligible_by_state[current["state"]] += 1
    for state, count in eligible_by_state.items():
        q = np.array(environment["reward1_given_choice_state0"])[::(-1 if state == "1" else 1)]
        for f in (0, 1):
            mass = pi * (q if f else 1 - q)
            denominators[f] += count * mass.sum()
            numerators[f] += count * (mass @ stay)
    return {"by_feedback": {str(f): {"expected_denominator": float(denominators[f]),
              "expected_stay": float(numerators[f]), "expected_switch": float(denominators[f] - numerators[f]),
              "ratio_of_expected_counts": float(numerators[f] / denominators[f]) if denominators[f] else None}
              for f in (0, 1)}}


def summarize(streams, cfg):
    cases = []
    for schedule in cfg["schedules"]:
        for policy in cfg["policies"]:
            for environment in cfg["environments"]:
                matched = [r for r in streams if (r["schedule"], r["policy"], r["environment"]) ==
                           (schedule["id"], policy["id"], environment["id"])]
                for stratum in ("all", "phase1", "phase2"):
                    cells = [r["strata"][stratum] for r in matched]
                    feedback = {}
                    for f in ("0", "1"):
                        empirical = [cell["empirical"]["by_feedback"][f] for cell in cells]
                        d, s = sum(v["denominator"] for v in empirical), sum(v["stay"] for v in empirical)
                        ratios = [v["stay_rate"] for v in empirical if v["stay_rate"] is not None]
                        feedback[f] = {"summed_denominator": d, "summed_stay": s, "summed_switch": d - s,
                                       "ratio_of_summed_empirical_counts": s / d if d else None,
                                       "mean_defined_seed_ratio": float(np.mean(ratios)) if ratios else None,
                                       "sd_defined_seed_ratio": float(np.std(ratios, ddof=1)) if len(ratios) > 1 else None,
                                       "defined_seeds": len(ratios),
                                       "ratio_of_expected_counts": cells[0]["analytic"]["by_feedback"][f]["ratio_of_expected_counts"]}
                    cases.append({"schedule": schedule["id"], "policy": policy["id"], "environment": environment["id"],
                                  "stratum": stratum, "by_feedback": feedback})
    def cell(schedule, policy, env, stratum):
        return next(r["by_feedback"] for r in cases if (r["schedule"], r["policy"], r["environment"], r["stratum"]) ==
                    (schedule, policy, env, stratum))
    def difference(values):
        return values["1"]["ratio_of_expected_counts"] - values["0"]["ratio_of_expected_counts"]
    bias = difference(cell("fixed", "iid_biased", "probabilistic", "all"))
    sticky = [c["by_feedback"][f]["ratio_of_expected_counts"] for c in cases if c["policy"] == "sticky" for f in ("0", "1")]
    pooled, first, second = (difference(cell("reversal", "iid_biased", "probabilistic", key)) for key in ("all", "phase1", "phase2"))
    tolerance = cfg["analytic_tolerance"]
    claims = {"H1_no_learning_feedback_difference": {"analytic_difference": bias, "reference": -9 / 34,
                   "criterion_met": abs(bias + 9 / 34) <= tolerance and abs(bias) >= .2},
              "H2_no_learning_high_persistence": {"maximum_absolute_error_from_0_9": max(abs(v - .9) for v in sticky),
                   "criterion_met": all(abs(v - .9) <= tolerance for v in sticky)},
              "H3_pooling_masks_phase_difference": {"pooled_difference": pooled, "phase1_difference": first,
                   "phase2_difference": second, "criterion_met": abs(pooled) <= tolerance and first <= -.2 and second >= .2}}
    return {"study_id": cfg["study_id"], "cases": cases, "analytic_claims": claims,
            "parameter_updates": 0, "monte_carlo_exact_match_required": False,
            "scope": "No learning policy counterexamples; MC ratios described without confidence intervals or species inference"}


def run(root, config_path, registration_path, out):
    started = datetime.now(timezone.utc)
    root, out = Path(root).resolve(), Path(out).resolve()
    cfg, reg = (json.loads(Path(p).read_text(encoding="utf8")) for p in (config_path, registration_path))
    validate_config(cfg)
    registered = datetime.fromisoformat(reg["registered_at_utc"])
    require(type(reg["source_git_commit"]) is str and re.fullmatch(r"[a-f0-9]{40}", reg["source_git_commit"])
            and registered.tzinfo is not None and registered.utcoffset().total_seconds() == 0
            and registered <= started, "Invalid registered source version or UTC chronology")
    require(reg["study_id"] == cfg["study_id"] and reg["config"] == cfg and reg["config_sha256"] == digest(config_path)
            and reg["frozen_files"] == freeze(root, cfg) and reg["parent_files"] == parents(root, cfg), "Registered source/config/parent bytes differ")
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "registration.json", reg)
    metadata = {"study_id": cfg["study_id"], "status": "running", "started_at_utc": started.isoformat(),
                "config": cfg, "config_sha256": digest(config_path), "source_git_commit": reg["source_git_commit"],
                "registration_sha256": digest(registration_path), "frozen_files": reg["frozen_files"],
                "parent_files": reg["parent_files"], "parameter_updates": 0,
                "runtime": {"python": platform.python_version(), "numpy": np.__version__},
                "human_or_species_comparison": False}
    write_new(out / "metadata.json", metadata)
    try:
        reports = []
        with (out / "trials.csv").open("x", encoding="utf8", newline="") as trials_handle, (out / "ledger.csv").open("x", encoding="utf8", newline="") as ledger_handle:
            tw, lw = csv.DictWriter(trials_handle, fieldnames=TRIAL_FIELDS), csv.DictWriter(ledger_handle, fieldnames=LEDGER_FIELDS)
            tw.writeheader()
            lw.writeheader()
            for seed in cfg["seeds"]:
                for schedule in cfg["schedules"]:
                    for policy in cfg["policies"]:
                        for env in cfg["environments"]:
                            rows = simulate(cfg, seed, schedule, policy, env)
                            strata = {}
                            for label in ("all", "phase1", "phase2"):
                                subset = rows if label == "all" else [r for r in rows if r["phase"] == label]
                                empirical, ledger = measure_stream(subset)
                                strata[label] = {"empirical": empirical, "analytic": analytic(subset, policy, env)}
                                if label == "all":
                                    lw.writerows(ledger)
                            reports.append({"seed": seed, "schedule": schedule["id"], "policy": policy["id"],
                                            "environment": env["id"], "strata": strata})
                            tw.writerows({k: (r["choice"] if k == "action" else r["feedback"] if k == "reward" else r[k]) for k in TRIAL_FIELDS} for r in rows)
        write_new(out / "metrics.json", {"study_id": cfg["study_id"], "streams": reports})
        summary = summarize(reports, cfg)
        write_new(out / "summary.json", summary)
        metadata.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        streams=len(reports), trials=len(reports) * cfg["trials"],
                        candidate_pairs=sum(r["strata"]["all"]["empirical"]["candidates"] for r in reports),
                        eligible_pairs=sum(r["strata"]["all"]["empirical"]["eligible"] for r in reports))
    except Exception as error:
        metadata.update(status="failed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        error_type=type(error).__name__, error_message=str(error))
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")
    write_new(out / "artifact_hashes.json", {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["register", "run"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Output already exists")
    require(args.mode != "run" or args.registration is not None, "Run requires registration")
    result = register(args.root, args.config, args.out) if args.mode == "register" else run(args.root, args.config, args.registration, args.out)
    print(json.dumps({"status": result.get("status", "registered"), "out": str(args.out)}))
    return int(result.get("status") == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
