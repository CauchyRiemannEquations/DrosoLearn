"""Independent verification of the prospective tuple-key RNG correction.

The original independent arithmetic implementation is reused unchanged. Neither
the original nor corrected experiment producer is imported by this verifier.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess

import numpy as np


CORE_PATH = Path(__file__).resolve().with_name("verify_human_shell.py")
SPEC = importlib.util.spec_from_file_location("frozen_shell_independent_core", CORE_PATH)
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)
NAMESPACE = 20261019
SOURCE_ADDITIONS = {
    "src/drosolearn/human_shell_rng.py", "tests/test_human_shell_rng.py", "docs/human-shell-rng-protocol.md",
    "configs/human_shell_rng_v2.json", "scripts/run_human_shell_rng.py", "tests/test_human_shell_rng_run.py",
    "scripts/verify_human_shell_rng.py", "tests/test_human_shell_rng_verify.py", "docs/human-shell-rng-design-note.md",
}
PARENT_NAMES = ("calibration.npz", "calibration.csv", "registration.json", "metadata.json", "selection.json",
                "trajectories.npz", "trajectory_metrics.csv", "witnesses.npz", "summary.json", "cohort_audit.json",
                "verification.json", "artifact_hashes.json")


def entropy_tuple(seed, pid):
    core.demand(type(seed) is int and seed >= 0, "Nonnegative integer model seed required")
    core.demand(type(pid) is str and re.fullmatch(r"A[0-9]{4}", pid) is not None, "Canonical participant ID required")
    return (NAMESPACE, seed, int(pid[1:]))


def generator(seed, pid):
    return np.random.default_rng(np.random.SeedSequence(entropy_tuple(seed, pid)))


def key_audit(ids, seeds):
    """Configuration-only enumeration; constructing states draws no uniforms."""
    core.demand(ids and seeds and len(ids) == len(set(ids)) and len(seeds) == len(set(seeds)), "Unique nonempty axes required")
    old, new_rows = {}, []
    tuples, states = set(), set()
    for seed in seeds:
        for pid in ids:
            entropy = entropy_tuple(seed, pid)
            old.setdefault(seed + 300000 + int(pid[1:]), []).append([seed, pid])
            seq = np.random.SeedSequence(entropy)
            state4 = seq.generate_state(4).astype("<u4")
            state_hash = hashlib.sha256(state4.tobytes()).hexdigest()
            initial = generator(seed, pid).bit_generator.state
            canonical = json.dumps(initial, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            pcg_hash = hashlib.sha256(canonical).hexdigest()
            core.demand(entropy not in tuples and state_hash not in states, "Corrected RNG initialization collision")
            tuples.add(entropy); states.add(state_hash)
            new_rows.append({"seed": seed, "participant_id": pid, "entropy": list(entropy),
                             "seedsequence_state_uint32": state4.tolist(), "seedsequence_state_sha256": state_hash,
                             "pcg64_initial_state": initial, "pcg64_initial_state_sha256": pcg_hash})
    core.demand(len({row["pcg64_initial_state_sha256"] for row in new_rows}) == len(new_rows), "PCG64 initial-state collision")
    collisions = {str(key): cells for key, cells in old.items() if len(cells) > 1}
    return {"cells": len(ids) * len(seeds), "old_unique_integer_keys": len(old),
            "old_collision_groups": len(collisions), "old_collision_cells": sum(len(cells) for cells in collisions.values()),
            "old_duplicate_uses_beyond_first": sum(len(cells) - 1 for cells in collisions.values()),
            "old_equal_key_pairs": sum(len(cells) * (len(cells) - 1) // 2 for cells in collisions.values()),
            "old_max_group_size": max(map(len, old.values())), "new_unique_entropy_tuples": len(tuples),
            "new_unique_seedsequence_states": len(states), "new_unique_pcg64_states": len(new_rows),
            "rng_draws_in_key_audit": 0, "old_collisions": collisions, "new_keys": new_rows}


def replay_tuple(policy, record, seed):
    """An environment reads potential outcomes only after the model chooses Go."""
    n = len(record["shell"])
    uniforms = generator(seed, record["id"]).random(n)
    fields = ("p_go", "action", "reward", "head_step", "internal_step", "value_step")
    tape = {name: np.empty(n, float) for name in fields}
    for index, u in enumerate(uniforms):
        shell = int(record["shell"][index])
        p = policy.probability(shell)
        action = int(u < p)
        reward = (0, 1, -1)[int(record["content"][index])] if action else 0
        steps = policy.update(shell, action, reward)
        tape["p_go"][index], tape["action"][index], tape["reward"][index] = p, action, reward
        for field, value in zip(fields[3:], steps):
            tape[field][index] = value
    return tape


def validate_repair_config(cfg, original):
    core.demand(cfg["study_id"] == "human_shell_rng_v2" and cfg["scope"] == "prospectively_registered_action_stream_repair",
                "Corrected study identity required")
    allowed = {"study_id", "scope", "source_files", "parent_files", "repair_contract"}
    core.demand({key: value for key, value in cfg.items() if key not in allowed}
                == {key: value for key, value in original.items() if key not in allowed},
                "Repair changed a learner, participant, seed budget, or criterion")
    expected_sources = set(original["source_files"]) | SOURCE_ADDITIONS
    expected_parents = set(original["parent_files"]) | {"results/human_shell_v1/" + name for name in PARENT_NAMES}
    expected_parents.add("results/human_shell_rng_design_issue_v1.json")
    core.demand(len(cfg["source_files"]) == len(set(cfg["source_files"]))
                and set(cfg["source_files"]) == expected_sources, "Exact corrected source inventory required")
    core.demand(len(cfg["parent_files"]) == len(set(cfg["parent_files"]))
                and set(cfg["parent_files"]) == expected_parents, "Exact corrected parent inventory required")
    contract = cfg["repair_contract"]
    required = {"original_study_id": "human_shell_v1", "original_result_dir": "results/human_shell_v1",
                "new_execution": "autonomous_only", "action_entropy_namespace": NAMESPACE,
                "action_generator": "default_rng(SeedSequence([20261019, seed, int(participant_id[1:])])).random(trials_per_person)",
                "original_action_cells": 145, "original_unique_additive_addresses": 88,
                "required_new_unique_tuple_addresses": 145, "new_autonomous_modeled_trials": 344520,
                "new_calibration_trials": 0, "new_conditional_trials": 0,
                "preserve_original_calibration_and_selected_rates": True,
                "preserve_original_conditional_array_bytes": True,
                "preserve_original_conditional_csv_lines": True,
                "preserve_original_conditional_witness_bytes": True,
                "preserve_H_prediction_exactly": True, "original_outcome_blind_design_audit": True,
                "original_source_and_results_immutable": True}
    core.compare_tree(contract, required, "Repair contract", subset=True)


def conditional_csv_bytes(path):
    """Keep literal original CSV bytes; these generated CSVs contain no multiline fields."""
    lines = Path(path).read_bytes().splitlines(keepends=True)
    core.demand(bool(lines), "Missing measurement CSV")
    header = next(csv.reader(io.StringIO(lines[0].decode("utf-8-sig"))))
    core.demand(len(header) == len(set(header)) and "mode" in header, "Invalid measurement header")
    mode_index = header.index("mode")
    conditional = []
    for raw in lines[1:]:
        values = next(csv.reader(io.StringIO(raw.decode("utf-8"))))
        core.demand(len(values) == len(header), "Unexpected multiline or malformed CSV record")
        core.demand(values[mode_index] in ("conditional", "autonomous"), "Unknown measurement mode")
        if values[mode_index] == "conditional":
            conditional.append(raw)
    return lines[0], b"".join(conditional), len(conditional)


def validate_unchanged_model_source(reg, original_reg):
    core.demand(all(reg["source_files"].get(name) == expected for name, expected in original_reg["source_files"].items()),
                "Repair changed original learner/source bytes")
    core.demand(reg["graph_files"] == original_reg["graph_files"], "Repair changed anatomical source bytes")
    core.demand(reg["external_files"] == original_reg["external_files"], "Repair changed human source bytes")


def identical_array_bytes(actual, original, description):
    actual, original = np.asarray(actual), np.asarray(original)
    core.demand(actual.shape == original.shape and actual.dtype == original.dtype
                and actual.tobytes() == original.tobytes(), "Preserved array changed: " + description)


def assert_parent_reuse(parent, result, cfg):
    parent, result = Path(parent), Path(result)
    for name in ("selection.json", "cohort_audit.json"):
        core.demand((parent / name).read_bytes() == (result / name).read_bytes(), "Preserved parent bytes changed: " + name)
    p_header, p_lines, p_count = conditional_csv_bytes(parent / "trajectory_metrics.csv")
    n_header, n_lines, n_count = conditional_csv_bytes(result / "trajectory_metrics.csv")
    expected_count = len(cfg["conditions"]) * len(cfg["test_seeds"]) * len(cfg["test_ids"])
    core.demand(p_header == n_header and p_lines == n_lines and p_count == n_count == expected_count,
                "Original conditional CSV records changed")
    halves = {}
    with np.load(parent / "trajectories.npz", allow_pickle=False) as old, np.load(result / "trajectories.npz", allow_pickle=False) as new:
        core.demand(set(old.files) == set(new.files), "Corrected trajectory array inventory changed")
        for name in old.files:
            identical_array_bytes(new[name][0], old[name][0], name + " conditional half")
            halves[name] = hashlib.sha256(old[name][0].tobytes()).hexdigest()
    conditional_witnesses = initial_witnesses = 0
    with np.load(parent / "witnesses.npz", allow_pickle=False) as old, np.load(result / "witnesses.npz", allow_pickle=False) as new:
        core.demand(set(old.files) == set(new.files), "Corrected witness inventory changed")
        for name in old.files:
            if name.startswith("conditional_"):
                identical_array_bytes(new[name], old[name], name)
                conditional_witnesses += 1
            if name.startswith("autonomous_") and "_initial_" in name:
                identical_array_bytes(new[name], old[name], name)
                initial_witnesses += 1
    parent_summary, summary = core.read_json(parent / "summary.json"), core.read_json(result / "summary.json")
    core.demand(summary["H_prediction"] == parent_summary["H_prediction"], "Previously fixed conditional prediction conclusion changed")
    for name in ("human_metrics", "conditional_family_mean_nll", "conditional_family_mean_brier"):
        core.demand(summary[name] == parent_summary[name], "Previously fixed observations changed: " + name)
    core.demand(not (result / "calibration.npz").exists() and not (result / "calibration.csv").exists(),
                "Corrected run must reference parent calibration artifacts")
    return {"conditional_half_sha256": halves, "conditional_csv_sha256": hashlib.sha256(p_lines).hexdigest(),
            "conditional_csv_records": p_count, "conditional_witness_fields": conditional_witnesses,
            "unchanged_autonomous_initial_witness_fields": initial_witnesses}


def verify(root, config, registration, result_dir, cache):
    root, result_dir, cache = Path(root).resolve(), Path(result_dir).resolve(), Path(cache).resolve()
    cfg, reg = core.read_json(config), core.read_json(registration)
    original_cfg = core.read_json(root / "configs/human_shell_v1.json")
    validate_repair_config(cfg, original_cfg)
    core.demand(reg["config"] == cfg and reg["config_sha256"] == core.digest(config), "Corrected registered config differs")
    core.demand(reg["registration_type"] == "action_stream_repair_without_refitting", "Wrong corrected registration type")
    core.demand(reg["study_id"] == cfg["study_id"], "Corrected registration identity")
    core.demand((result_dir / "registration.json").read_bytes() == Path(registration).read_bytes(), "Corrected registration copy differs")
    for group in ("source_files", "parent_files", "graph_files"):
        names = cfg[group]
        core.demand(len(names) == len(set(names)) and set(reg[group]) == set(names), "Frozen inventory differs: " + group)
        for name, expected in reg[group].items():
            core.demand(core.digest(core.contained(root, name)) == expected, "Frozen corrected input changed: " + name)
    for name, expected in reg["source_files"].items():
        blob = subprocess.check_output(["git", "show", reg["source_git_commit"] + ":" + name], cwd=root)
        core.demand(hashlib.sha256(blob).hexdigest() == expected, "Corrected source commit differs: " + name)
    core.demand(reg["parent_release_tag"] == "human-shell-v1-design-limited", "Original design-limited release tag required")
    tagged = subprocess.check_output(["git", "rev-parse", reg["parent_release_tag"]], cwd=root, text=True).strip()
    core.demand(tagged == reg["parent_release_commit"], "Original release identity changed")
    for name, expected in reg["parent_files"].items():
        blob = subprocess.check_output(["git", "show", reg["parent_release_commit"] + ":" + name], cwd=root)
        core.demand(hashlib.sha256(blob).hexdigest() == expected, "Original release parent bytes differ: " + name)
    core.demand(core.digest(__file__) == reg["source_files"]["scripts/verify_human_shell_rng.py"]
                and core.digest(CORE_PATH) == reg["source_files"]["scripts/verify_human_shell.py"], "Independent arithmetic sources not frozen")
    core.demand(core.read_json(root / "results/human_task_structure_v1/audit.json")["gate"]["direct_human_model_similarity_allowed"] is False,
                "Historical source gate must remain closed")
    required_gate = {"previous_human_data_gate_unchanged": True, "neural_mechanism_or_drug_effect_tested": False,
                     "parent_arithmetic_verification_passed": True, "original_autonomous_inference_design_limited": True,
                     "action_address_repair_only": True, "no_refitting": True}
    core.compare_tree(reg["gate"], required_gate, "Corrected gate", subset=True)
    manifest = core.read_json(core.contained(root, cfg["manifest"]))
    core.demand(not cache.is_relative_to(root) and set(reg["external_files"]) == set(manifest["files"]), "External source scope")
    for name, specification in manifest["files"].items():
        path = core.contained(cache, name)
        core.demand(path.stat().st_size == specification["bytes"] and core.digest(path) == specification["sha256"], "Corrected external bytes changed")
        core.demand(reg["external_files"][name] == {"bytes": specification["bytes"], "sha256": specification["sha256"]}, "Corrected external registration differs")
    parent_relative = cfg["repair_contract"]["original_result_dir"]
    parent = (root / parent_relative).resolve()
    core.demand(parent.is_relative_to(root) and parent.is_dir(), "Contained original result required")
    parent_reg, parent_verification = core.read_json(parent / "registration.json"), core.read_json(parent / "verification.json")
    core.demand(parent_reg["config"] == original_cfg and parent_verification["status"] == "passed", "Original independent verification required")
    core.demand(parent_verification["summary_sha256"] == core.digest(parent / "summary.json"), "Original verified summary changed")
    core.demand(parent_verification["registration_sha256"] == core.digest(parent / "registration.json"),
                "Original verification belongs to a different registration")
    validate_unchanged_model_source(reg, parent_reg)
    for name, expected in parent_reg["source_files"].items():
        core.demand(core.digest(core.contained(root, name)) == expected, "Original source changed after its registration")
    for name, expected in core.read_json(parent / "artifact_hashes.json").items():
        core.demand(core.digest(core.contained(parent, name)) == expected, "Original scientific artifact changed")
    metadata = core.read_json(result_dir / "metadata.json")
    expected_metadata = {"study_id": cfg["study_id"], "status": "completed", "stage": "completed",
                         "source_git_commit": reg["source_git_commit"], "registration_sha256": core.digest(registration),
                         "config_sha256": core.digest(config), "new_calibration_observations": 0,
                         "new_conditional_predictions": 0, "reused_conditional_predictions": 344520,
                         "new_autonomous_choices": 344520, "original_human_people": 49, "original_human_trials": 10584,
                         "combined_container_observations": 689040,
                         "calibration_people": 20, "heldout_people": 29, "test_model_initialization_seeds": 5,
                         "selection_unchanged": True, "conditional_predictions_unchanged": True,
                         "human_model_comparison_tested": True, "shared_neural_mechanism_tested": False,
                         "drug_effect_tested": False}
    core.compare_tree(metadata, expected_metadata, "Corrected metadata", subset=True)
    stamps = [datetime.fromisoformat(parent_verification["verified_at_utc"]), datetime.fromisoformat(reg["registered_at_utc"]),
              datetime.fromisoformat(metadata["started_at_utc"]), datetime.fromisoformat(metadata["completed_at_utc"])]
    core.demand(all(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0 for stamp in stamps), "UTC correction timeline")
    core.demand(stamps[0] <= stamps[1] < stamps[2] <= stamps[3] <= datetime.now(timezone.utc), "Prospective correction timeline")
    expected_artifacts = {"registration.json", "metadata.json", "selection.json", "reused_parent_data.json", "trajectories.npz",
                          "trajectory_metrics.csv", "witnesses.npz", "summary.json", "cohort_audit.json", "rng_keys.json"}
    artifact_map = core.read_json(result_dir / "artifact_hashes.json")
    core.demand(set(artifact_map) == expected_artifacts, "Corrected primary artifact inventory")
    for name, expected in artifact_map.items():
        core.demand(core.digest(core.contained(result_dir, name)) == expected, "Corrected primary artifact changed: " + name)
    preserved = assert_parent_reuse(parent, result_dir, cfg)
    expected_reused = {"parent_study_id": "human_shell_v1", "parent_result_dir": parent_relative,
                       "parent_files": {name: core.digest(parent / name) for name in PARENT_NAMES}, **preserved,
                       "new_calibration_observations": 0, "new_conditional_predictions": 0}
    core.compare_tree(core.read_json(result_dir / "reused_parent_data.json"), expected_reused, "Parent reuse manifest")
    addresses = key_audit(cfg["test_ids"], cfg["test_seeds"])
    core.compare_tree(core.read_json(result_dir / "rng_keys.json"), addresses, "Independent action-key audit")
    core.demand(addresses["cells"] == addresses["new_unique_entropy_tuples"]
                == addresses["new_unique_seedsequence_states"] == addresses["new_unique_pcg64_states"] == 145,
                "Corrected action-state uniqueness")
    records, calibration_ids, test_ids, excluded = core.parse_cohort(cache / "Dataset_TNU.zip")
    core.demand(calibration_ids == cfg["calibration_ids"] and test_ids == cfg["test_ids"], "Corrected fixed participant split")
    core.check_source_audit(cache / "Dataset_TNU.zip", manifest, core.read_json(result_dir / "cohort_audit.json"))
    for pid, record in records.items():
        record["id"] = pid
    selection = core.read_json(parent / "selection.json")
    base, graph_cache = core.raw_graph(root / cfg["graph"]), {}
    old_rows = core.read_measurements(parent / "trajectory_metrics.csv", ("mode", "condition", "seed", "participant_id"))
    rows = core.read_measurements(result_dir / "trajectory_metrics.csv", ("mode", "condition", "seed", "participant_id"))
    model_behavior = np.empty((11, 29, 5, 3)); nll = np.empty((11, 29, 5)); brier = np.empty((11, 29, 5))
    fields = {"probabilities": "p_go", "actions": "action", "rewards": "reward", "head_steps": "head_step",
              "internal_steps": "internal_step", "value_steps": "value_step"}
    replayed = witness_checks = conditional_records = 0
    expected_witness_names = set()
    with np.load(result_dir / "trajectories.npz", allow_pickle=False) as source, np.load(result_dir / "witnesses.npz", allow_pickle=False) as witness:
        core.demand(set(source.files) == set(fields), "Corrected trajectory inventory")
        arrays = {name: source[name] for name in fields}
        core.demand(all(array.shape == (2, 11, 5, 29, 216) for array in arrays.values()), "Corrected trajectory tensor axes")
        for si, seed in enumerate(cfg["test_seeds"]):
            for ci, condition in enumerate(core.CONDITIONS):
                graph = core.graph_for(condition, seed, base, graph_cache)
                for pi, pid in enumerate(test_ids):
                    conditional_key = ("conditional", condition, str(seed), pid)
                    core.demand(conditional_key in rows and rows.pop(conditional_key) == old_rows[conditional_key], "Conditional measurement values changed")
                    conditional_records += 1
                    p = arrays["probabilities"][0, ci, si, pi]
                    a = arrays["actions"][0, ci, si, pi]
                    core.demand(np.array_equal(a, records[pid]["action"]), "Preserved conditional actions differ from human source")
                    nll[ci, pi, si], brier[ci, pi, si] = core.score_choices(p, a)
                    core.scalar_close(float(old_rows[conditional_key]["mean_nll"]), nll[ci, pi, si], "Preserved conditional NLL")
                    core.scalar_close(float(old_rows[conditional_key]["brier"]), brier[ci, pi, si], "Preserved conditional Brier")
                    policy = core.AuditPolicy(condition, seed, selection["rates"].get(condition, cfg["rates"][0]),
                                              graph, selection["calibration_bias"])
                    before = policy.witness()
                    tape = replay_tuple(policy, records[pid], seed)
                    after = policy.witness()
                    for field, key in fields.items():
                        core.array_close(arrays[field][1, ci, si, pi], tape[key], f"Corrected/{condition}/{seed}/{pid}/{field}")
                    expected = core.expected_measurement(policy, before, after, tape, records[pid], condition, seed, pid, mode="autonomous")
                    key = ("autonomous", condition, str(seed), pid)
                    core.demand(key in rows, "Missing corrected autonomous measurement")
                    core.check_csv_row(rows.pop(key), expected, "Corrected autonomous row")
                    model_behavior[ci, pi, si] = core.behavioral_metrics(records[pid], tape["action"])
                    if pi == 0:
                        for stage, values in (("initial", before), ("final", after)):
                            for field, value in values.items():
                                name = f"autonomous_{condition}_{stage}_{field}"
                                core.demand(name in witness.files and witness[name].shape == (5, *np.asarray(value).shape), "Corrected witness axes")
                                core.array_close(witness[name][si], value, name)
                                expected_witness_names.add(name); witness_checks += 1
                    replayed += 1
            print(f"independent tuple-key autonomous seed {seed} complete", flush=True)
        core.demand({name for name in witness.files if name.startswith("autonomous_")} == expected_witness_names,
                    "Unexpected corrected autonomous witness fields")
    core.demand(not rows and replayed == conditional_records == 1595, "Missing or extra corrected measurement rows")
    expected_summary = core.expected_summary(cfg, records, model_behavior, nll, brier)
    summary = core.read_json(result_dir / "summary.json")
    core.compare_tree(summary, expected_summary, "Independent corrected summary", subset=True)
    core.demand(set(summary) == set(expected_summary), "Unexpected corrected summary claim fields")
    return {"status": "passed", "study_id": cfg["study_id"], "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "registration_sha256": core.digest(registration), "summary_sha256": core.digest(result_dir / "summary.json"),
            "source_files": len(reg["source_files"]), "parent_files": len(reg["parent_files"]),
            "graph_files": len(reg["graph_files"]), "external_files": len(reg["external_files"]),
            "primary_artifacts": len(expected_artifacts), "human_people": 49, "human_original_trials": 10584,
            "calibration_people": 20, "test_people": 29, "structural_exclusions": excluded,
            "independent_autonomous_trajectories": replayed, "independent_autonomous_observations": replayed * 216,
            "independent_witness_array_checks": witness_checks, "new_calibration_observations": 0,
            "new_conditional_predictions": 0, "preserved_conditional_predictions": conditional_records * 216,
            "combined_container_observations": 689040,
            "unique_action_entropy_tuples": 145, "unique_seedsequence_states": 145, "unique_pcg64_initial_states": 145,
            "conditional_arrays_byte_identical": True, "conditional_csv_byte_identical": True,
            "conditional_witnesses_byte_identical": True, "autonomous_initial_states_byte_identical": True,
            "selected_rates_and_H_prediction_unchanged": True, "parent_arithmetic_verification_passed": True,
            "original_autonomous_design_limitation_preserved": True, "verifier_imports_producer": False,
            "verifier_modifies_original_results": False, "all_frozen_inputs_match": True,
            "interpretation": "independent corrected autonomous replay only; no parameter refit, extra seeds, or hidden feedback"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    core.demand(not args.out.exists(), "New corrected verification output required")
    report = verify(args.root, args.config, args.registration, args.result_dir, args.cache)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": report["status"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
