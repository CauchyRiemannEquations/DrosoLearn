"""Repair-runner guards over invented files and configuration metadata only."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module("shell_rng_runner_tests", "scripts/run_human_shell_rng.py")
verifier = load_module("shell_rng_verifier_tests", "scripts/verify_human_shell_rng.py")


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def metadata_config_pair(tmp_path):
    parent = runner.read(ROOT / "configs/human_shell_v1.json")
    repair = runner.read(ROOT / "configs/human_shell_rng_v2.json")
    put_json(tmp_path / "configs/human_shell_v1.json", parent)
    return repair


def test_exact_repair_science_config_accepted_without_execution(tmp_path):
    runner.validate_config(tmp_path, metadata_config_pair(tmp_path))


@pytest.mark.parametrize("key,value", [("trials_per_person", 432), ("bootstrap_repeats", 5),
                                       ("equivalence_margin", .5), ("prediction_margin", 0),
                                       ("learning_gain_margin", 0), ("test_seeds", [8]),
                                       ("rates", [.1, 1, 10]), ("bootstrap_seed", 123)])
def test_fixed_science_budget_or_criterion_cannot_change(tmp_path, key, value):
    cfg = metadata_config_pair(tmp_path)
    cfg[key] = value
    with pytest.raises(ValueError):
        runner.validate_config(tmp_path, cfg)


def test_participant_split_and_order_cannot_change(tmp_path):
    cfg = metadata_config_pair(tmp_path)
    cfg["test_ids"] = list(reversed(cfg["test_ids"]))
    with pytest.raises(ValueError):
        runner.validate_config(tmp_path, cfg)


def test_repair_cannot_refit_or_reexecute_conditional(tmp_path):
    cfg = metadata_config_pair(tmp_path)
    cfg["repair_contract"]["new_conditional_trials"] = 1
    with pytest.raises(ValueError):
        runner.validate_config(tmp_path, cfg)


@pytest.mark.parametrize("group,remove", [("source_files", "src/drosolearn/human_shell_rng.py"),
                                         ("parent_files", "results/human_shell_v1/trajectories.npz")])
def test_frozen_closure_cannot_replace_required_file_with_unrelated_file(tmp_path, group, remove):
    cfg = metadata_config_pair(tmp_path)
    cfg[group][cfg[group].index(remove)] = "README.md"
    with pytest.raises(ValueError):
        runner.validate_config(tmp_path, cfg)


@pytest.mark.parametrize("group", ["source_files", "parent_files"])
def test_frozen_closure_cannot_have_duplicate_paths(tmp_path, group):
    cfg = metadata_config_pair(tmp_path)
    cfg[group][-1] = cfg[group][0]
    with pytest.raises(ValueError):
        runner.validate_config(tmp_path, cfg)


def test_key_inventory_matches_independent_initial_state_audit_no_uniform_draws():
    cfg = {"test_ids": ["A9001", "A9002"], "test_seeds": [41, 42]}
    actual = runner.key_inventory(cfg)
    assert actual == verifier.key_audit(cfg["test_ids"], cfg["test_seeds"])
    assert actual["rng_draws_in_key_audit"] == 0
    assert actual["cells"] == actual["new_unique_entropy_tuples"] == 4
    assert actual["old_unique_integer_keys"] == 3


@pytest.mark.parametrize("ids,seeds", [([], [41]), (["A9001"], []), (["A9001", "A9001"], [41]),
                                       (["A9001"], [41, 41]), (["A１２３４"], [41]), (["A123"], [41])])
def test_key_inventory_rejects_empty_duplicate_or_invalid_axes(ids, seeds):
    with pytest.raises(ValueError):
        runner.key_inventory({"test_ids": ids, "test_seeds": seeds})


def parent_csv(tmp_path):
    path = tmp_path / "parent.csv"
    path.write_bytes(b'mode,condition,seed,participant_id,mean_nll,brier,head_update_calls,head_nonzero_steps\r\n'
                     b'"conditional","fixed",41,"A9001",0.30000000000000004,0.20,3,2\r\n'
                     b'"autonomous","fixed",41,"A9001",,,3,1\r\n')
    return path


def test_read_rows_restores_numeric_types_and_missing_values(tmp_path):
    rows = runner.read_rows(parent_csv(tmp_path))
    assert rows[0]["seed"] == 41 and type(rows[0]["seed"]) is int
    assert rows[0]["head_update_calls"] == 3 and type(rows[0]["head_nonzero_steps"]) is int
    assert type(rows[0]["mean_nll"]) is float and rows[0]["brier"] == .2
    assert rows[1]["mean_nll"] is None and rows[1]["brier"] is None


def test_conditional_csv_is_literal_byte_copy_and_output_exclusive(tmp_path):
    parent = parent_csv(tmp_path)
    rows = runner.read_rows(parent)
    rows[1]["head_nonzero_steps"] = 3
    out = tmp_path / "new.csv"
    digest, count = runner.write_combined_csv(out, parent, rows)
    header, original, n, _ = runner.conditional_bytes(parent)
    new_header, copied, copied_n, _ = runner.conditional_bytes(out)
    assert header == new_header and original == copied and count == n == copied_n == 1
    assert digest == hashlib.sha256(original).hexdigest()
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        runner.write_combined_csv(out, parent, rows)
    assert out.read_bytes() == before


def test_conditional_metric_tamper_rejected_before_output(tmp_path):
    parent = parent_csv(tmp_path)
    rows = runner.read_rows(parent)
    rows[0]["mean_nll"] = .4
    out = tmp_path / "bad.csv"
    with pytest.raises(ValueError):
        runner.write_combined_csv(out, parent, rows)
    assert not out.exists()


@pytest.mark.parametrize("payload", [b"", b"mode,mode\nconditional,conditional\n",
                                    b"mode,value\ninvalid,1\n", b"mode,value\nconditional\n"])
def test_malformed_conditional_csv_rejected(tmp_path, payload):
    path = tmp_path / "bad.csv"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        runner.conditional_bytes(path)


def test_copy_exact_is_exclusive_and_byte_preserving(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "copy"
    source.write_bytes(b"\x00bytes\r\n")
    runner.copy_exact(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    source.write_bytes(b"changed")
    with pytest.raises(FileExistsError):
        runner.copy_exact(source, destination)
    assert destination.read_bytes() == b"\x00bytes\r\n"


def reuse_fixture(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    for name in runner.PARENT_NAMES:
        (parent / name).write_bytes(b"invented-parent-" + name.encode())
    arrays = {"probabilities": np.array([[[.2, .3]], [[.5, .6]]]),
              "actions": np.array([[[0, 1]], [[1, 0]]], dtype=np.int8)}
    witnesses = {"conditional_fixed_initial_head": np.array([[0., 0.]]),
                 "autonomous_fixed_initial_head": np.array([[0., 0.]]),
                 "autonomous_fixed_final_head": np.array([[1., 2.]])}
    return parent, arrays, witnesses


def test_reuse_receipt_preserves_conditional_and_autonomous_initial_only(tmp_path):
    parent, old, old_w = reuse_fixture(tmp_path)
    arrays = {key: value.copy() for key, value in old.items()}
    arrays["actions"][1] = 0
    witnesses = {key: value.copy() for key, value in old_w.items()}
    witnesses["autonomous_fixed_final_head"][:] = 9.
    receipt = runner.reuse_receipt(parent, arrays, witnesses, old, old_w, "invented-csv-digest", 1)
    assert receipt["new_conditional_predictions"] == 0
    assert receipt["conditional_witness_fields"] == receipt["unchanged_autonomous_initial_witness_fields"] == 1
    assert receipt["conditional_half_sha256"]["actions"] == hashlib.sha256(old["actions"][0].tobytes()).hexdigest()


@pytest.mark.parametrize("kind", ["array_dtype", "array_bytes", "array_missing_key", "conditional_witness", "initial_witness", "witness_dtype"])
def test_reuse_receipt_rejects_preserved_data_changes(tmp_path, kind):
    parent, old, old_w = reuse_fixture(tmp_path)
    arrays, witnesses = copy.deepcopy(old), copy.deepcopy(old_w)
    if kind == "array_dtype": arrays["probabilities"] = arrays["probabilities"].astype(np.float32)
    elif kind == "array_bytes": arrays["actions"][0, 0, 0] = 1
    elif kind == "array_missing_key": del arrays["actions"]
    elif kind == "conditional_witness": witnesses["conditional_fixed_initial_head"][0, 0] = 1
    elif kind == "initial_witness": witnesses["autonomous_fixed_initial_head"][0, 0] = 1
    elif kind == "witness_dtype": witnesses["autonomous_fixed_initial_head"] = witnesses["autonomous_fixed_initial_head"].astype(np.float32)
    with pytest.raises(ValueError):
        runner.reuse_receipt(parent, arrays, witnesses, old, old_w, "digest", 1)


def gate_fixture(tmp_path):
    synthesis = tmp_path / "results/phase4_synthesis_v1/synthesis.json"
    put_json(synthesis, {"direct_level4_test_completed": False})
    put_json(synthesis.with_name("verification.json"), {"status": "passed", "synthesis_sha256": runner.sha(synthesis)})
    put_json(tmp_path / "results/time_mixture_v1/verification.json", {"status": "passed"})
    put_json(tmp_path / "results/human_task_structure_v1/audit.json", {"gate": {"direct_human_model_similarity_allowed": False}})
    original_cfg = runner.read(ROOT / "configs/human_shell_v1.json")
    for name in original_cfg["source_files"] + original_cfg["graph_files"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"invented frozen file: " + name.encode())
    put_json(tmp_path / "configs/human_shell_v1.json", original_cfg)
    external = {name: {"bytes": len(name), "sha256": hashlib.sha256(name.encode()).hexdigest()}
                for name in ("Dataset_TNU.zip", "metadata.json", "readme.pdf")}
    put_json(tmp_path / original_cfg["manifest"], {"files": external})
    parent = tmp_path / "results/human_shell_v1"
    put_json(parent / "metadata.json", {"status": "completed"})
    put_json(parent / "summary.json", {"invented_summary": True})
    put_json(parent / "registration.json", {"study_id": "human_shell_v1", "config": original_cfg,
                                             "config_sha256": runner.sha(tmp_path / "configs/human_shell_v1.json"),
                                             "source_files": runner.hashes(tmp_path, original_cfg["source_files"]),
                                             "graph_files": runner.hashes(tmp_path, original_cfg["graph_files"]),
                                             "external_files": external})
    put_json(parent / "verification.json", {"status": "passed", "registration_sha256": runner.sha(parent / "registration.json"),
                                             "summary_sha256": runner.sha(parent / "summary.json")})
    (parent / "payload.bin").write_bytes(b"invented frozen computation")
    put_json(parent / "artifact_hashes.json", {p.name: runner.sha(p) for p in parent.iterdir()
                                               if p.name not in ("verification.json", "artifact_hashes.json")})
    return runner.read(ROOT / "configs/human_shell_rng_v2.json"), parent


def test_gate_requires_completed_verified_parent_and_exact_artifact_bytes(tmp_path):
    cfg, parent = gate_fixture(tmp_path)
    result = runner.gate(tmp_path, cfg)
    assert result["no_refitting"] is True and result["parent_arithmetic_verification_passed"] is True
    (parent / "payload.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        runner.gate(tmp_path, cfg)


def test_gate_rejects_unverified_parent(tmp_path):
    cfg, parent = gate_fixture(tmp_path)
    put_json(parent / "verification.json", {"status": "failed"})
    with pytest.raises(ValueError):
        runner.gate(tmp_path, cfg)


@pytest.mark.parametrize("relative", ["src/drosolearn/human_shell.py", "src/drosolearn/human_shell_models.py",
                                      "data/flywire_783_mb_left_kc512_s701/edges.csv"])
def test_gate_rejects_mutated_original_source_or_graph(tmp_path, relative):
    cfg, _ = gate_fixture(tmp_path)
    (tmp_path / relative).write_bytes(b"changed after original registration")
    with pytest.raises(ValueError):
        runner.gate(tmp_path, cfg)


@pytest.mark.parametrize("field", ["registration_sha256", "summary_sha256"])
def test_gate_rejects_verification_bound_to_wrong_parent_bytes(tmp_path, field):
    cfg, parent = gate_fixture(tmp_path)
    verified = runner.read(parent / "verification.json")
    verified[field] = "0" * 64
    put_json(parent / "verification.json", verified)
    with pytest.raises(ValueError):
        runner.gate(tmp_path, cfg)
