"""Exact time-mixture calibration; no sampled trials or learning are performed."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CASE_IDS = ("joint_drift", "p_constant", "q_constant", "joint_constant")
METRIC = "ratio_of_expected_counts_not_expectation_of_sampled_ratios"
SOURCES = (
    "src/drosolearn/time_mixture.py", "scripts/run_time_mixture.py", "tests/test_time_mixture.py",
    "configs/time_mixture_v1.json", "docs/time-mixture-protocol.md",
    "scripts/verify_time_mixture.py", "tests/test_time_mixture_verifier.py", "docs/time-mixture-independent-audit.md",
    "src/drosolearn/__init__.py", "pyproject.toml", "requirements-lock.txt", "requirements-human-candidate-audit.txt",
)
RESOURCES = ("learning_updates", "learner_constructions", "sampled_trials", "rng_draws")
PARENTS = tuple(path for study in ("conditional_transition", "metric_calibration") for path in (
    f"configs/{study}_v1.json", f"results/registration_{study}_v1.json", f"results/{study}_v1/metadata.json",
    f"results/{study}_v1/metrics.json", f"results/{study}_v1/verification.json", f"results/{study}_v1/artifact_hashes.json",
)) + ("results/registration_human_task_structure_v1.json", "results/human_task_structure_v1/audit.json",
      "results/human_task_structure_v1/verification.json", "docs/conditional-transition-protocol.md", "docs/metric-calibration-protocol.md")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def safe(root, relative):
    require(type(relative) is str and relative and "\\" not in relative and ":" not in relative and not Path(relative).is_absolute(), "Relative POSIX path required")
    require(all(part not in ("", ".", "..") for part in relative.split("/")), "Canonical contained path required")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Path escaped repository")
    return path


def ratio(value):
    """Read an exact canonical rational probability, never an approximate float."""
    require(type(value) is str and re.fullmatch(r"(?:0|[1-9][0-9]*)/(?:[1-9][0-9]*)", value), "Canonical numerator/denominator string required")
    result = Fraction(value)
    require(value == f"{result.numerator}/{result.denominator}" and 0 <= result <= 1, "Reduced rational probability in [0,1] required")
    return result


def encoded(value):
    return None if value is None else {"numerator": value.numerator, "denominator": value.denominator}


def _case_vectors(case, trials):
    require(type(trials) is int and trials >= 2, "At least two original trials required")
    require(type(case) is dict and set(case) == {"id", "p0", "q"} and case["id"] in CASE_IDS, "Exact known case fields required")
    require(type(case["p0"]) is list and type(case["q"]) is list and len(case["p0"]) == len(case["q"]) == trials,
            "One choice probability and reward pair per original trial required")
    require(all(type(pair) is list and len(pair) == 2 for pair in case["q"]), "Two reward probabilities per original trial required")
    return [ratio(value) for value in case["p0"]], [[ratio(value) for value in pair] for pair in case["q"]]


def validate_config(cfg):
    fields = {"study_id", "scope", "trials", "cases", *RESOURCES, "human_or_species_comparison", "causal_effect_estimated",
              "metric", "parent_files", "parent_gate_identity", "frozen_files"}
    require(type(cfg) is dict and set(cfg) == fields and cfg["study_id"] == "time_mixture_v1", "Exact study configuration fields required")
    require(cfg["scope"] in ("registered_exact_time_mixture_calibration", "integration_test"), "Known explicit scope required")
    require(type(cfg["trials"]) is int and cfg["trials"] >= 2, "At least two original trials required")
    require(type(cfg["cases"]) is list and all(type(case) is dict for case in cfg["cases"]) and [case.get("id") for case in cfg["cases"]] == list(CASE_IDS), "All four cases in fixed order required")
    for case in cfg["cases"]:
        _case_vectors(case, cfg["trials"])
    require(all(type(cfg[key]) is int and cfg[key] == 0 for key in RESOURCES)
            and cfg["human_or_species_comparison"] is False and cfg["causal_effect_estimated"] is False and cfg["metric"] == METRIC,
            "Exact no-learning expectation-count scope required")
    for field in ("parent_files", "frozen_files"):
        require(type(cfg[field]) is list and cfg[field] and all(type(path) is str for path in cfg[field]) and len(set(cfg[field])) == len(cfg[field]), "Explicit unique source and parent inventories required")
        for path in cfg[field]:
            safe(ROOT, path)
    identity = cfg["parent_gate_identity"]
    require(type(identity) is dict and set(identity) == {"published_commit", "file_sha256"}
            and type(identity["published_commit"]) is str and re.fullmatch(r"[a-f0-9]{40}", identity["published_commit"])
            and type(identity["file_sha256"]) is dict and set(identity["file_sha256"]) == set(cfg["parent_files"])
            and all(type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) for value in identity["file_sha256"].values()), "Exact pinned parent hashes required")
    if cfg["scope"] != "integration_test":
        require(cfg["trials"] == 9 and set(cfg["frozen_files"]) == set(SOURCES) and set(cfg["parent_files"]) == set(PARENTS), "Registered nine-trial scope and complete source/parent closure required")
        drift_p, constant_p = ["4/5"] * 4 + ["1/5"] * 5, ["4/5"] * 9
        drift_q, constant_q = [["9/10", "2/5"] for _ in range(4)] + [["3/5", "1/10"] for _ in range(5)], [["9/10", "2/5"] for _ in range(9)]
        expected = [(drift_p, drift_q), (constant_p, drift_q), (drift_p, constant_q), (constant_p, constant_q)]
        require(all(case["p0"] == p and case["q"] == q for case, (p, q) in zip(cfg["cases"], expected)), "Prospective cases cannot change")


def _empty():
    return {str(choice): {str(feedback): [Fraction(0), Fraction(0), Fraction(0)] for feedback in (0, 1)} for choice in (0, 1)}


def _finish(counts):
    result = {}
    for choice, cells in counts.items():
        values = {}
        for feedback, (denominator, stay, switch) in cells.items():
            require(stay + switch == denominator, "Exact expected counts do not partition")
            values[feedback] = {"expected_denominator": encoded(denominator), "expected_stay": encoded(stay), "expected_switch": encoded(switch),
                                "expected_count_ratio": encoded(stay / denominator if denominator else None)}
        d0, s0, _ = cells["0"]
        d1, s1, _ = cells["1"]
        result[choice] = {"by_feedback": values, "difference_1_minus_0": encoded(s1 / d1 - s0 / d0 if d0 and d1 else None)}
    return result


def enumerate_case(case, trials):
    """Enumerate eight local events for every pair, including clock transitions."""
    p, q = _case_vectors(case, trials)
    all_counts, events, timeline = _empty(), [], []
    for t in range(trials - 1):
        counts = _empty()
        total_weight = Fraction(0)
        for current in (0, 1):
            for following in (0, 1):
                for feedback in (0, 1):
                    weight = (p[t] if current == 0 else 1 - p[t]) * (p[t + 1] if following == 0 else 1 - p[t + 1]) * (q[t][current] if feedback else 1 - q[t][current])
                    for destination in (counts, all_counts):
                        destination[str(current)][str(feedback)][0] += weight
                        destination[str(current)][str(feedback)][1 if current == following else 2] += weight
                    events.append({"case_id": case["id"], "t": t + 1, "current_choice": current,
                                   "next_choice": following, "feedback": feedback, "weight": encoded(weight)})
                    total_weight += weight
        require(total_weight == 1, "Local events must form a complete probability space")
        timeline.append({"t": t + 1, "p0_t": encoded(p[t]), "p0_next": encoded(p[t + 1]), "q0_t": encoded(q[t][0]),
                         "q1_t": encoded(q[t][1]), "by_current_choice": _finish(counts)})
    require(sum(cell[0] for cells in all_counts.values() for cell in cells.values()) == trials - 1, "Every original pair must be retained")
    return {"case_id": case["id"], "by_current_choice": _finish(all_counts), "by_time": timeline}, events


def _zero_difference(case):
    return all(row["difference_1_minus_0"] in (None, encoded(Fraction(0))) for row in case["by_current_choice"].values())


def analyse(cfg):
    validate_config(cfg)
    cases, events = [], []
    for case in cfg["cases"]:
        result, local = enumerate_case(case, cfg["trials"])
        cases.append(result)
        events.extend(local)
    expected = encoded(Fraction(9, 56))
    hypotheses = {
        "time_mixture_counterexample": cases[0]["by_current_choice"]["0"]["difference_1_minus_0"] == expected
            and cases[0]["by_current_choice"]["1"]["difference_1_minus_0"] == encoded(-Fraction(9, 56)),
        "constant_p_control_zero": _zero_difference(cases[1]), "constant_q_control_zero": _zero_difference(cases[2]),
        "joint_constant_control_zero": _zero_difference(cases[3]),
        "exact_time_conditional_zero": all(_zero_difference(row) for case in cases for row in case["by_time"]),
    }
    return {"study_id": cfg["study_id"], "cases": cases, "hypotheses": hypotheses, "interpretation": METRIC}, {"study_id": cfg["study_id"], "events": events}


def parent_gate(root, cfg):
    expected = cfg["parent_gate_identity"]["file_sha256"]
    require(all(digest(safe(root, path)) == checksum for path, checksum in expected.items()), "A pinned published parent changed")
    for study in ("metric_calibration", "conditional_transition"):
        check = read_json(safe(root, f"results/{study}_v1/verification.json"))
        metadata = read_json(safe(root, f"results/{study}_v1/metadata.json"))
        require(check["status"] == "passed" and metadata["status"] == "completed"
                and check["registration_sha256"] == digest(safe(root, f"results/registration_{study}_v1.json"))
                and check["config_sha256"] == digest(safe(root, f"configs/{study}_v1.json"))
                and metadata["registration_sha256"] == check["registration_sha256"], "Independent parent verification does not bind registration/config")
        inventory = read_json(safe(root, f"results/{study}_v1/artifact_hashes.json"))
        require(inventory["metrics.json"] == digest(safe(root, f"results/{study}_v1/metrics.json"))
                and all(digest(safe(root, f"results/{study}_v1/" + path)) == checksum for path, checksum in inventory.items()), "Parent primary artifacts differ")
    task = read_json(safe(root, "results/human_task_structure_v1/audit.json"))
    checked = read_json(safe(root, "results/human_task_structure_v1/verification.json"))
    require(task["status"] == "pass" and task["gate"] == {"source_coded_descriptive_metric_allowed": True,
            "exact_historical_task_execution_version_verified": False, "direct_human_model_similarity_allowed": False}
            and checked["status"] == "pass" and checked["result_sha256"] == digest(safe(root, "results/human_task_structure_v1/audit.json"))
            and checked["registration_sha256"] == digest(safe(root, "results/registration_human_task_structure_v1.json"))
            and checked["source_coded_descriptive_gate_passed"] is True, "Human descriptive-only gate must remain unchanged")
    return {"passed": True, "published_commit": cfg["parent_gate_identity"]["published_commit"],
            "direct_human_model_similarity_allowed": False, "human_raw_records_read": 0,
            "scope": "synthetic_exact_counterexample_not_reanalysis_of_human_or_model_behaviour"}


def hashes(root, paths):
    return {path: digest(safe(root, path)) for path in sorted(paths)}


def register(root, config_path, out):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    require(not Path(out).exists(), "Registration output must be new")
    cfg = read_json(config_path)
    validate_config(cfg)
    gate = parent_gate(root, cfg)
    record = {"study_id": cfg["study_id"], "registration_type": "exact_no_learning_time_mixture",
              "registered_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "config": cfg, "config_sha256": digest(config_path), "frozen_files": hashes(root, cfg["frozen_files"]),
              "parent_files": hashes(root, cfg["parent_files"]), "gate": gate}
    require(re.fullmatch(r"[a-f0-9]{40}", record["source_git_commit"]), "Exact source Git commit required")
    write_new(out, record)
    return record


def run(root, config_path, registration_path, out):
    root, config_path, registration_path, out = (Path(path).resolve() for path in (root, config_path, registration_path, out))
    cfg, reg = read_json(config_path), read_json(registration_path)
    validate_config(cfg)
    started = datetime.now(timezone.utc)
    registered = datetime.fromisoformat(reg["registered_at_utc"])
    require(registered.tzinfo is not None, "Registration requires an aware UTC timestamp")
    require(reg["study_id"] == cfg["study_id"] and reg["registration_type"] == "exact_no_learning_time_mixture"
            and reg["config"] == cfg and reg["config_sha256"] == digest(config_path)
            and reg["frozen_files"] == hashes(root, cfg["frozen_files"]) and reg["parent_files"] == hashes(root, cfg["parent_files"])
            and reg["gate"] == parent_gate(root, cfg)
            and re.fullmatch(r"[a-f0-9]{40}", reg["source_git_commit"])
            and registered.astimezone(timezone.utc) <= started,
            "Prior source, config, parent, gate or registration identity differs")
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "registration.json", reg)
    metadata = {"study_id": cfg["study_id"], "status": "running", "started_at_utc": started.isoformat(),
                "config": cfg, "config_sha256": digest(config_path), "source_git_commit": reg["source_git_commit"],
                "registration_sha256": digest(registration_path), "frozen_files": reg["frozen_files"], "parent_files": reg["parent_files"],
                "gate": reg["gate"], **{key: 0 for key in RESOURCES}, "human_or_species_comparison": False,
                "causal_effect_estimated": False, "cases": 0, "original_adjacent_pairs": 0, "local_events": 0}
    try:
        metrics, ledger = analyse(cfg)
        write_new(out / "metrics.json", metrics)
        write_new(out / "local_events.json", ledger)
        metadata.update(status="completed", cases=len(cfg["cases"]), original_adjacent_pairs=len(cfg["cases"]) * (cfg["trials"] - 1), local_events=len(ledger["events"]))
    except Exception as error:
        metadata.update(status="failed", error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_new(out / "metadata.json", metadata)
        write_new(out / "artifact_hashes.json", {path.name: digest(path) for path in sorted(out.iterdir()) if path.is_file()})
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--register", action="store_true")
    modes.add_argument("--registration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Existing output cannot be overwritten")
    result = register(args.root, args.config, args.out) if args.register else run(args.root, args.config, args.registration, args.out)
    print(json.dumps({"status": result.get("status", "registered"), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
