"""Independent integer-weighted full-path replay of a time-mixture counterexample.

The producer's local-event routines are never imported. For n choices this tool
visits every one of 2**(2*n-1) choice/feedback bit paths, with integer weights
over a common base-ten denominator. It audits a logical counterexample, not a
human or animal learning effect and not a sampled random-denominator ratio.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
STUDY = "time_mixture_v1"
CONFIG = "configs/time_mixture_v1.json"
METRIC = "ratio_of_expected_counts_not_expectation_of_sampled_ratios"
CASE_IDS = ("joint_drift", "p_constant", "q_constant", "joint_constant")
RESOURCES = ("learning_updates", "learner_constructions", "sampled_trials", "rng_draws")
FLAGS = ("human_or_species_comparison", "causal_effect_estimated")
PRIMARY = {"registration.json", "metadata.json", "metrics.json", "local_events.json"}
SOURCES = {
    "src/drosolearn/time_mixture.py", "scripts/run_time_mixture.py", "tests/test_time_mixture.py",
    CONFIG, "docs/time-mixture-protocol.md", "scripts/verify_time_mixture.py",
    "tests/test_time_mixture_verifier.py", "docs/time-mixture-independent-audit.md",
    "src/drosolearn/__init__.py", "pyproject.toml", "requirements-lock.txt", "requirements-human-candidate-audit.txt",
}
PARENTS = {f"{prefix}{study}_v1{suffix}" for study in ("conditional_transition", "metric_calibration")
           for prefix, suffix in (("configs/", ".json"), ("results/registration_", ".json"),
                                  ("results/", "/metadata.json"), ("results/", "/metrics.json"),
                                  ("results/", "/verification.json"), ("results/", "/artifact_hashes.json"))}
PARENTS |= {"results/registration_human_task_structure_v1.json", "results/human_task_structure_v1/audit.json",
            "results/human_task_structure_v1/verification.json", "docs/conditional-transition-protocol.md",
            "docs/metric-calibration-protocol.md"}


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(items):
        out = {}
        for key, value in items:
            require(key not in out, "Duplicate JSON key")
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf8"), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(VerificationError("Nonfinite JSON")))


def same(actual, expected, context):
    def exact(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(b, dict):
            return a.keys() == b.keys() and all(exact(a[k], v) for k, v in b.items())
        if isinstance(b, list):
            return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
        return a == b
    require(exact(actual, expected), context)


def safe(root, relative):
    require(type(relative) is str and relative and "\\" not in relative and ":" not in relative
            and not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Safe POSIX relative path required")
    base = Path(root).resolve()
    target = (base / relative).resolve()
    require(target.is_relative_to(base) and target != base, "Path escaped its root")
    require(target.is_file(), "Required file is missing: " + relative)
    return target


def utc(value):
    require(type(value) is str, "UTC timestamp must be text")
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0, "Explicit UTC timestamp required")
    return result


def rational(value):
    require(type(value) is str and re.fullmatch(r"(?:0|[1-9][0-9]*)/[1-9][0-9]*", value), "Canonical nonnegative fraction string required")
    out = Fraction(value)
    require(f"{out.numerator}/{out.denominator}" == value and 0 <= out <= 1 and (out * 10).denominator == 1,
            "Probability must be canonical, in [0,1], and have an integer base-ten weight")
    return out


def pack(value):
    if value is None:
        return None
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def cell(denominator, stay):
    return {"expected_denominator": pack(denominator), "expected_stay": pack(stay),
            "expected_switch": pack(denominator - stay),
            "expected_count_ratio": pack(stay / denominator) if denominator else None}


def exhaustive(case):
    """Enumerate entire bit paths; only invented short cases are used in tests."""
    require(type(case) is dict and set(case) == {"id", "p0", "q"}, "Exact case shape")
    require(type(case["id"]) is str and type(case["p0"]) is list and 2 <= len(case["p0"]) <= 9,
            "Bounded finite timeline required")
    p = [rational(v) for v in case["p0"]]
    n = len(p)
    require(type(case["q"]) is list and len(case["q"]) == n
            and all(type(v) is list and len(v) == 2 for v in case["q"]), "Two feedback probabilities at every original time")
    q = [[rational(v) for v in row] for row in case["q"]]
    p_units = [int(v * 10) for v in p]
    q_units = [[int(v * 10) for v in row] for row in q]
    total_denominator = 10 ** (2 * n - 1)
    weighted_events = [0] * ((n - 1) * 8)
    mass, paths = 0, 0
    for choices in product((0, 1), repeat=n):
        choice_weight = math.prod(p_units[t] if action == 0 else 10 - p_units[t]
                                  for t, action in enumerate(choices))
        for rewards in product((0, 1), repeat=n - 1):
            feedback_weight = math.prod(q_units[t][choices[t]] if reward else 10 - q_units[t][choices[t]]
                                        for t, reward in enumerate(rewards))
            weight = choice_weight * feedback_weight
            mass += weight
            paths += 1
            for t, reward in enumerate(rewards):
                index = t * 8 + choices[t] * 4 + choices[t + 1] * 2 + reward
                weighted_events[index] += weight
    require(paths == 2 ** (2 * n - 1) and mass == total_denominator, "All complete path weights sum exactly to one")
    events = []
    for t in range(n - 1):
        for action, next_action, feedback in product((0, 1), repeat=3):
            index = t * 8 + action * 4 + next_action * 2 + feedback
            events.append({"case_id": case["id"], "t": t + 1, "current_choice": action,
                           "next_choice": next_action, "feedback": feedback,
                           "weight": pack(Fraction(weighted_events[index], total_denominator))})

    def selected(times):
        out = {}
        for action in (0, 1):
            by_feedback, ratios = {}, {}
            for feedback in (0, 1):
                denominator = Fraction(sum(weighted_events[t * 8 + action * 4 + b * 2 + feedback]
                                           for t in times for b in (0, 1)), total_denominator)
                stay = Fraction(sum(weighted_events[t * 8 + action * 4 + action * 2 + feedback]
                                    for t in times), total_denominator)
                by_feedback[str(feedback)] = cell(denominator, stay)
                ratios[feedback] = stay / denominator if denominator else None
            difference = ratios[1] - ratios[0] if all(v is not None for v in ratios.values()) else None
            out[str(action)] = {"by_feedback": by_feedback, "difference_1_minus_0": pack(difference)}
        return out

    by_time = [{"t": t + 1, "p0_t": pack(p[t]), "p0_next": pack(p[t + 1]),
                "q0_t": pack(q[t][0]), "q1_t": pack(q[t][1]), "by_current_choice": selected((t,))}
               for t in range(n - 1)]
    metrics = {"case_id": case["id"], "by_current_choice": selected(range(n - 1)), "by_time": by_time}
    return events, metrics, {"complete_paths": paths, "path_weight_sum": pack(Fraction(mass, total_denominator)),
                             "original_adjacent_pairs": n - 1, "local_events": len(events)}


def validate_config(cfg):
    fields = {"study_id", "scope", "trials", "cases", *RESOURCES, *FLAGS, "metric", "parent_files", "parent_gate_identity", "frozen_files"}
    require(type(cfg) is dict and set(cfg) == fields, "Exact source configuration fields")
    require(type(cfg) is dict and cfg.get("study_id") == STUDY and cfg.get("metric") == METRIC, "Registered study and estimand")
    require(cfg.get("scope") in ("registered_exact_time_mixture_calibration", "integration_test"), "Exact logical-calibration scope")
    require(type(cfg.get("trials")) is int and 2 <= cfg["trials"] <= 9, "Bounded timeline")
    require(type(cfg.get("cases")) is list and [c.get("id") for c in cfg["cases"]] == list(CASE_IDS), "All four fixed conditions in order")
    require(all(len(c["p0"]) == cfg["trials"] for c in cfg["cases"]), "Original timeline retained in every condition")
    for key in RESOURCES:
        require(type(cfg.get(key)) is int and cfg[key] == 0, "No learning, sampled trials, constructions or RNG: " + key)
    for key in FLAGS:
        require(cfg.get(key) is False, "No human/species or causal effect estimate")
    require(type(cfg.get("frozen_files")) is list and len(cfg["frozen_files"]) == len(set(cfg["frozen_files"]))
            and set(cfg["frozen_files"]) == SOURCES, "Unique complete frozen source/import closure")
    require(type(cfg.get("parent_files")) is list and len(cfg["parent_files"]) == len(set(cfg["parent_files"]))
            and set(cfg["parent_files"]) == PARENTS, "Unique complete parent closure")
    if cfg["scope"] == "registered_exact_time_mixture_calibration":
        p0 = ["4/5"] * 4 + ["1/5"] * 5
        q = [["9/10", "2/5"] for _ in range(4)] + [["3/5", "1/10"] for _ in range(5)]
        same(cfg["cases"], [{"id": "joint_drift", "p0": p0, "q": q},
                             {"id": "p_constant", "p0": ["4/5"] * 9, "q": q},
                             {"id": "q_constant", "p0": p0, "q": [["9/10", "2/5"] for _ in range(9)]},
                             {"id": "joint_constant", "p0": ["4/5"] * 9, "q": [["9/10", "2/5"] for _ in range(9)]}],
             "Exact predeclared time-varying policy and reward controls")
        require(cfg["trials"] == 9, "Scientific complete path has nine choices and eight outcomes")
    else:
        require(cfg["trials"] <= 4, "Engineering fixtures must be short and invented")


def source_imports_are_independent():
    source = Path(__file__).read_text(encoding="utf8")
    modules = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            require(node.level == 0 and node.module, "No relative producer imports")
            modules.append(node.module.split(".")[0])
    require(all(module == "__future__" or module in sys.stdlib_module_names for module in modules),
            "Independent verifier imports standard library only")
    return sorted(set(modules))


def parent_gate(root, cfg, registered):
    identity = cfg["parent_gate_identity"]
    require(type(identity) is dict and set(identity) == {"published_commit", "file_sha256"}
            and type(identity["published_commit"]) is str and re.fullmatch(r"[a-f0-9]{40}", identity["published_commit"])
            and set(identity["file_sha256"]) == PARENTS, "Exact pinned parent identity")
    actual = {}
    for relative, checksum in identity["file_sha256"].items():
        require(type(checksum) is str and re.fullmatch(r"[a-f0-9]{64}", checksum), "Parent SHA-256 syntax")
        actual[relative] = digest(safe(root, relative))
    same(actual, identity["file_sha256"], "All pinned published parent bytes")
    parent_primary = 0
    for study in ("conditional_transition", "metric_calibration"):
        base = f"results/{study}_v1"
        checked = read_json(safe(root, base + "/verification.json"))
        meta = read_json(safe(root, base + "/metadata.json"))
        config_path = safe(root, f"configs/{study}_v1.json")
        reg_path = safe(root, f"results/registration_{study}_v1.json")
        require(checked["status"] == "passed" and checked.get("errors") == []
                and meta["status"] == "completed" and checked["registration_sha256"] == digest(reg_path)
                and checked["config_sha256"] == digest(config_path)
                and meta["registration_sha256"] == checked["registration_sha256"], "Verified parent registration/config/result binding")
        require(utc(checked["verified_at_utc"]) <= registered, "Parent independent check precedes current registration")
        inventory = read_json(safe(root, base + "/artifact_hashes.json"))
        expected = {"ledger.csv", "metadata.json", "metrics.json", "registration.json"}
        if study == "metric_calibration":
            expected |= {"summary.json", "trials.csv"}
        require(type(inventory) is dict and set(inventory) == expected, "Complete unchanged parent primary inventory")
        for relative, checksum in inventory.items():
            require(type(checksum) is str and re.fullmatch(r"[a-f0-9]{64}", checksum)
                    and digest(safe(root, base + "/" + relative)) == checksum, "Every parent primary artifact byte")
            parent_primary += 1
        require((safe(root, base + "/registration.json")).read_bytes() == reg_path.read_bytes(), "Parent original/copied registration bytes")
    task_path = safe(root, "results/human_task_structure_v1/audit.json")
    task = read_json(task_path)
    checked = read_json(safe(root, "results/human_task_structure_v1/verification.json"))
    same(task["gate"], {"source_coded_descriptive_metric_allowed": True,
                        "exact_historical_task_execution_version_verified": False,
                        "direct_human_model_similarity_allowed": False}, "Historical human direct-comparison gate remains closed")
    require(task["status"] == "pass" and checked["status"] == "pass" and checked.get("errors") == []
            and checked["source_coded_descriptive_gate_passed"] is True and checked["result_sha256"] == digest(task_path)
            and checked["registration_sha256"] == digest(safe(root, "results/registration_human_task_structure_v1.json"))
            and utc(checked["verified_at_utc"]) <= registered, "Independent human descriptive-only parent binding")
    gate = {"passed": True, "published_commit": identity["published_commit"],
            "direct_human_model_similarity_allowed": False, "human_raw_records_read": 0,
            "scope": "synthetic_exact_counterexample_not_reanalysis_of_human_or_model_behaviour"}
    return actual, gate, parent_primary


def audit(root, registration, result_dir):
    root, reg_path, output = (Path(path).resolve() for path in (root, registration, result_dir))
    reg = read_json(reg_path)
    require(set(reg) == {"study_id", "registration_type", "registered_at_utc", "source_git_commit", "config",
                         "config_sha256", "frozen_files", "parent_files", "gate"}, "Exact registration shape")
    require(reg["study_id"] == STUDY and reg["registration_type"] == "exact_no_learning_time_mixture"
            and type(reg["source_git_commit"]) is str and re.fullmatch(r"[a-f0-9]{40}", reg["source_git_commit"]), "Source Git and exact logical registration")
    cfg = read_json(safe(root, CONFIG))
    validate_config(cfg)
    same(reg["config"], cfg, "Registered configuration values")
    require(reg["config_sha256"] == digest(safe(root, CONFIG)), "Exact source configuration bytes")
    registered = utc(reg["registered_at_utc"])
    require(registered <= datetime.now(timezone.utc), "Prior UTC registration")
    frozen = {relative: digest(safe(root, relative)) for relative in sorted(SOURCES)}
    same(reg["frozen_files"], frozen, "Complete frozen code, tests, protocol and import dependencies")
    require(digest(__file__) == frozen["scripts/verify_time_mixture.py"], "Actually executed verifier matches frozen source bytes")
    imported = source_imports_are_independent()
    parents, gate, parent_primary = parent_gate(root, cfg, registered)
    same(reg["parent_files"], parents, "Registered complete parent hash map")
    same(reg["gate"], gate, "Registered descriptive-only gate")
    require((output / "registration.json").read_bytes() == reg_path.read_bytes(), "Identical original/copied registration bytes")
    artifacts = read_json(output / "artifact_hashes.json")
    require(type(artifacts) is dict and set(artifacts) == PRIMARY, "Exactly all four primary artifacts")
    actual_files = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    require(actual_files <= PRIMARY | {"artifact_hashes.json", "verification.json"}, "No unregistered extra primary artifact")
    for relative, checksum in artifacts.items():
        require(type(checksum) is str and re.fullmatch(r"[a-f0-9]{64}", checksum)
                and digest(safe(output, relative)) == checksum, "Current primary artifact hash: " + relative)
    meta = read_json(output / "metadata.json")
    expected_meta = {"study_id", "status", "started_at_utc", "completed_at_utc", "config", "config_sha256", "source_git_commit",
                     "registration_sha256", "frozen_files", "parent_files", "gate", *RESOURCES, *FLAGS,
                     "cases", "original_adjacent_pairs", "local_events"}
    require(type(meta) is dict and set(meta) == expected_meta and meta["study_id"] == STUDY and meta["status"] == "completed", "Complete exact run metadata")
    for key in ("config", "config_sha256", "source_git_commit", "frozen_files", "parent_files", "gate"):
        same(meta[key], reg[key], "Run registration binding: " + key)
    require(meta["registration_sha256"] == digest(reg_path), "Run original registration SHA")
    require(registered <= utc(meta["started_at_utc"]) <= utc(meta["completed_at_utc"]) <= datetime.now(timezone.utc), "Registration/run/completion UTC chronology")
    for key in RESOURCES:
        require(type(meta[key]) is int and meta[key] == 0, "No resource inflation or hidden learning/sampling: " + key)
    for key in FLAGS:
        require(meta[key] is False, "Logical result does not become causal/species evidence")
    events, cases, path_counts = [], [], []
    for case in cfg["cases"]:
        case_events, case_metrics, counts = exhaustive(case)
        events.extend(case_events)
        cases.append(case_metrics)
        path_counts.append({"case_id": case["id"], **counts})

    def zero(c):
        return all(row["difference_1_minus_0"] in (None, pack(0)) for row in c["by_current_choice"].values())
    hypotheses = {"time_mixture_counterexample": cases[0]["by_current_choice"]["0"]["difference_1_minus_0"] == pack(Fraction(9, 56))
                  and cases[0]["by_current_choice"]["1"]["difference_1_minus_0"] == pack(-Fraction(9, 56)),
                  "constant_p_control_zero": zero(cases[1]), "constant_q_control_zero": zero(cases[2]),
                  "joint_constant_control_zero": zero(cases[3]),
                  "exact_time_conditional_zero": all(zero(row) for c in cases for row in c["by_time"])}
    expected_metrics = {"study_id": STUDY, "cases": cases, "hypotheses": hypotheses, "interpretation": METRIC}
    same(read_json(output / "local_events.json"), {"study_id": STUDY, "events": events}, "Every event reconstructed from exhaustive complete paths")
    same(read_json(output / "metrics.json"), expected_metrics, "All exact fractions, time coordinates, counts, nulls, ratios and hypotheses")
    counts = {"cases": len(cases), "original_adjacent_pairs": len(cases) * (cfg["trials"] - 1), "local_events": len(events),
              "complete_paths_per_case": 2 ** (2 * cfg["trials"] - 1),
              "complete_paths_total": sum(c["complete_paths"] for c in path_counts), "source_files": len(frozen),
              "parent_files": len(parents), "parent_primary_artifacts": parent_primary, "primary_artifacts": len(artifacts),
              **{key: 0 for key in RESOURCES}}
    for key in ("cases", "original_adjacent_pairs", "local_events"):
        require(type(meta[key]) is int and meta[key] == counts[key], "Complete metadata accounting: " + key)
    return {"scope": "independent_exact_no_learning_logical_counterexample_not_human_model_comparison", "counts": counts,
            "path_counts": path_counts, "hypotheses": hypotheses, "source_imports": imported,
            "registration_sha256": digest(reg_path), "config_sha256": reg["config_sha256"],
            "metrics_sha256": digest(output / "metrics.json"), "local_events_sha256": digest(output / "local_events.json"),
            "artifact_hashes_sha256": digest(output / "artifact_hashes.json"), "gate": gate,
            "checks": ["Full integer-weighted path probability mass equals one exactly in every case",
                       "Every original adjacent coordinate and event, aggregate and time-conditioned rational cell",
                       "All current source, configuration, parent, primary artifact bytes and UTC identities",
                       "Descriptive human gate remains closed; no learning or RNG and no human data reanalysis"]}


def verify(root, registration, result_dir):
    report = {"study_id": STUDY, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "failed", "passed": False, "errors": [], "counts": {},
              "method": "independent_integer_weighted_exhaustive_complete_choice_and_feedback_paths",
              "not_verified": ["Human/model similarity or differences, historical task identity, causal effects or shared neural mechanisms",
                               "Authenticity of remote publication beyond pinned local bytes; separate archive evidence is required",
                               "Expectation of an empirical random-denominator ratio"]}
    try:
        report.update(audit(root, registration, result_dir))
        report.update(status="passed", passed=True)
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Exclusive verification report required")
    report = verify(args.root, args.registration, args.result_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "counts", "errors")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
