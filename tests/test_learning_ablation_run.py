"""Exclusive tiny matched-prefix runs; fake parent proofs never claim science."""
from datetime import datetime, timedelta, timezone
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from drosolearn import learning_ablation as study
from test_learning_ablation import synthetic_config


REPO = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def read_csv(path):
    with Path(path).open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_parent_proof(root):
    """Invented metadata only, just to exercise the engineering gate in temp."""
    base = root / "results/stochastic_feedback_network_v1"
    regpath = root / "results/registration_stochastic_feedback_network_v1.json"
    configpath = root / "configs/stochastic_feedback_network_v1.json"
    save_json(regpath, {"synthetic_registration": True, "not_a_scientific_result": True})
    save_json(base / "summary.json", {"study_id": "stochastic_feedback_network_v1", "stage": "network",
                                      "status": "completed", "not_a_scientific_result": True})
    save_json(base / "verification.json", {"status": "passed", "summary_sha256": study.digest(base / "summary.json"),
             "registration_sha256": study.digest(regpath), "config_sha256": study.digest(configpath),
             "verified_at_utc": "2026-01-01T00:00:00+00:00", "not_a_scientific_result": True})
    save_json(base / "artifact_hashes.json", {"summary.json": study.digest(base / "summary.json")})
    return {"published_commit": "c" * 40, "summary_sha256": study.digest(base / "summary.json"),
            "verification_sha256": study.digest(base / "verification.json"), "registration_sha256": study.digest(regpath),
            "config_sha256": study.digest(configpath), "artifact_hashes_sha256": study.digest(base / "artifact_hashes.json")}


def make_fixture(tmp_path, monkeypatch):
    root = tmp_path / "synthetic_repo"
    root.mkdir()
    cfg = synthetic_config()
    for name in cfg["frozen_files"]:
        if name == "configs/learning_ablation_v1.json":
            continue
        source = REPO / name
        assert source.is_file(), f"Wait for all new source files before integration tests: {name}"
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    cfg["parent_gate_identity"] = make_parent_proof(root)
    configpath = root / "configs/learning_ablation_v1.json"
    save_json(configpath, cfg)
    original_git = study.subprocess.check_output
    def synthetic_git(command, **kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            assert Path(kwargs["cwd"]) == root
            return "d" * 40 + "\n"
        return original_git(command, **kwargs)
    monkeypatch.setattr(study.subprocess, "check_output", synthetic_git)
    original_splits = study.make_splits
    def guarded_splits(config, seed):
        assert config["scope"] == "integration_test" and seed in (17, 18)
        assert config["trials_per_phase"] == config["eval_trials"] == 8
        return original_splits(config, seed)
    monkeypatch.setattr(study, "make_splits", guarded_splits)
    return {"root": root, "cfg": cfg, "config": configpath, "registration": root / "registration.json", "out": root / "fresh_run"}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    return make_fixture(tmp_path, monkeypatch)


def register(fixture):
    return study.register(fixture["root"], fixture["config"], fixture["registration"])


def run(fixture):
    return study.run(fixture["root"], fixture["config"], fixture["registration"], fixture["out"])


@pytest.fixture
def completed(fixture):
    register(fixture)
    assert run(fixture)["status"] == "completed"
    return fixture


def test_registration_checks_bytes_without_generating_inputs_or_updating_any_model(fixture, monkeypatch):
    monkeypatch.setattr(study, "make_splits", lambda *a, **k: pytest.fail("Registration generated inputs"))
    monkeypatch.setattr(study, "selected_update", lambda *a, **k: pytest.fail("Registration trained"))
    record = register(fixture)
    assert record["config"] == fixture["cfg"]
    assert record["source_git_commit"] == "d" * 40
    assert len(record["frozen_files"]) == len(fixture["cfg"]["frozen_files"])
    assert record["gate"]["passed"] is True
    assert not fixture["out"].exists()


@pytest.mark.parametrize("target", ["source", "config", "graph", "parent"])
def test_frozen_input_changes_block_run_before_creating_output(fixture, target):
    register(fixture)
    path = {"source": fixture["root"] / "src/drosolearn/learning_ablation.py", "config": fixture["config"],
            "graph": fixture["root"] / fixture["cfg"]["graph"] / "provenance.json",
            "parent": fixture["root"] / "results/stochastic_feedback_network_v1/summary.json"}[target]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        run(fixture)
    assert not fixture["out"].exists()


@pytest.mark.parametrize("target", ["summary_sha256", "verification_sha256", "registration_sha256", "config_sha256", "artifact_hashes_sha256"])
def test_parent_gate_requires_every_exact_completed_parent_byte_identity(fixture, target):
    fixture["cfg"]["parent_gate_identity"][target] = "0" * 64
    save_json(fixture["config"], fixture["cfg"])
    with pytest.raises(ValueError):
        register(fixture)
    assert not fixture["registration"].exists()


def test_rehashing_replaced_parent_summary_cannot_reuse_its_old_verification(fixture):
    base = fixture["root"] / "results/stochastic_feedback_network_v1"
    summary = read_json(base / "summary.json")
    summary["unverified_replacement"] = True
    save_json(base / "summary.json", summary)
    save_json(base / "artifact_hashes.json", {"summary.json": study.digest(base / "summary.json")})
    fixture["cfg"]["parent_gate_identity"]["summary_sha256"] = study.digest(base / "summary.json")
    fixture["cfg"]["parent_gate_identity"]["artifact_hashes_sha256"] = study.digest(base / "artifact_hashes.json")
    save_json(fixture["config"], fixture["cfg"])
    with pytest.raises(ValueError):
        register(fixture)


def test_registration_and_previous_results_cannot_be_overwritten(fixture):
    register(fixture)
    before = fixture["registration"].read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        register(fixture)
    assert fixture["registration"].read_bytes() == before
    fixture["out"].mkdir()
    sentinel = fixture["out"] / "keep.bin"
    sentinel.write_bytes(b"previous result")
    with pytest.raises((ValueError, FileExistsError)):
        run(fixture)
    assert sentinel.read_bytes() == b"previous result"
    assert list(fixture["out"].iterdir()) == [sentinel]


@pytest.mark.parametrize("target", ["future_timestamp", "naive_timestamp", "source_version"])
def test_invalid_registration_time_or_git_identity_blocks_execution(fixture, target):
    record = register(fixture)
    if target == "source_version":
        record["source_git_commit"] = "unavailable"
    else:
        stamp = datetime.now(timezone.utc) + timedelta(days=1) if target == "future_timestamp" else datetime(2000, 1, 1)
        record["registered_at_utc"] = stamp.isoformat()
    save_json(fixture["registration"], record)
    with pytest.raises(ValueError):
        run(fixture)
    assert not fixture["out"].exists()


def test_prefix_is_trained_and_counted_once_and_masked_updates_are_separate(fixture, monkeypatch):
    calls = []
    original = study.selected_update
    def audited(model, inputs, action, reward, *, head=True, internal=True):
        assert np.asarray(inputs).shape == (2, 7)
        assert type(action) is type(reward) is int
        calls.append((head, internal, bool(model.weights.size)))
        return original(model, inputs, action, reward, head=head, internal=internal)
    monkeypatch.setattr(study, "selected_update", audited)
    register(fixture)
    metadata = run(fixture)
    assert metadata["actual_choice_feedback_trials"] == 320
    assert metadata["prefix_training_trials"] == 64
    assert metadata["branch_training_trials"] == 256
    assert metadata["head_update_calls"] == 192
    assert metadata["internal_update_calls"] == 144
    assert metadata["privileged_update_calls"] == 0
    assert len(calls) == 256  # Frozen branches have 64 actual trials but skip update.
    assert sum(head for head, internal, nonempty in calls) == 192
    assert sum(internal and nonempty for head, internal, nonempty in calls) == 144


def test_every_trial_and_shared_uniform_position_obeys_actual_policy_and_reward(completed):
    rows = read_csv(completed["out"] / "seed_17/trials.csv")
    assert len(rows) == 160
    assert sum(row["branch"] == "prefix" for row in rows) == 32
    with np.load(completed["out"] / "seed_17/stimuli.npz", allow_pickle=False) as stimuli:
        for row in rows:
            phase, index = row["phase"], int(row["trial"]) - 1
            p, au, ru = float(row["p_right"]), float(row["action_uniform"]), float(row["reward_uniform"])
            action, reward = int(row["action"]), int(row["reward"])
            assert au == stimuli[f"action_{phase}_uniforms"][index]
            assert ru == stimuli[f"reward_{phase}_uniforms"][index]
            assert action == int(au < p)
            target = int(np.argmax(stimuli[f"train_{phase}_counts"][index]) if phase == "phase1"
                         else np.argmin(stimuli[f"train_{phase}_counts"][index]))
            q = .8 if action == target else .2
            assert float(row["q_reward"]) == q
            assert reward == int(ru < q)
            assert int(row["chosen_count"]) == stimuli[f"train_{phase}_counts"][index, action]
            assert int(row["global_trial"]) == int(row["trial"]) + (8 if phase == "phase2" else 0)
            if row["head_enabled"] == "0":
                assert row["head_update_called"] == row["head_updates"] == "0"
                assert float(row["readout_step_l2"]) == 0.
            if row["internal_enabled"] == "0" or row["model"] == "linear":
                assert row["internal_update_called"] == row["internal_updates"] == "0"
                assert float(row["internal_step_l2"]) == 0.


def test_all_branch_initial_states_equal_learned_prefix_without_resetting_scale_bounds_or_native_counters(completed):
    base = completed["out"] / "seed_17"
    for model in study.MODELS:
        with np.load(base / model / "prefix/phase1_0008.npz", allow_pickle=False) as prefix:
            assert len(prefix.files) == 21
            assert int(prefix["head_updates"]) == 8
            assert int(prefix["internal_updates"]) == (0 if model == "linear" else 8)
            for branch in study.BRANCHES:
                with np.load(base / model / branch / "phase2_0000.npz", allow_pickle=False) as clone:
                    assert set(clone.files) == set(prefix.files)
                    for key in set(prefix.files) - {"phase_head_updates", "phase_internal_updates"}:
                        np.testing.assert_array_equal(prefix[key], clone[key])
                    assert int(clone["phase_head_updates"]) == int(clone["phase_internal_updates"]) == 0
                    assert np.linalg.norm(clone["readout_weights"]) > 0
                with np.load(base / model / branch / "phase2_0008.npz", allow_pickle=False) as endpoint:
                    np.testing.assert_array_equal(endpoint["feature_scale"], prefix["feature_scale"])
                    np.testing.assert_array_equal(endpoint["initial_internal_weights"], prefix["initial_internal_weights"])
                    np.testing.assert_array_equal(endpoint["internal_upper_bounds"], prefix["internal_upper_bounds"])
                    if branch in ("internal_only", "frozen"):
                        np.testing.assert_array_equal(endpoint["readout_weights"], prefix["readout_weights"])
                        assert int(endpoint["phase_head_updates"]) == 0 and int(endpoint["head_updates"]) == 8
                    if branch in ("head_only", "frozen") or model == "linear":
                        np.testing.assert_array_equal(endpoint["internal_weights"], prefix["internal_weights"])
                        assert int(endpoint["phase_internal_updates"]) == 0


def test_linear_equivalent_branch_pairs_have_exact_arrays_probabilities_choices_and_rewards(completed):
    base = completed["out"] / "seed_17/linear"
    rows = read_csv(completed["out"] / "seed_17/trials.csv")
    for left, right in (("full", "head_only"), ("internal_only", "frozen")):
        for t in (0, 4, 8):
            with np.load(base / left / f"phase2_{t:04d}.npz", allow_pickle=False) as a, \
                    np.load(base / right / f"phase2_{t:04d}.npz", allow_pickle=False) as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
        left_rows = [r for r in rows if r["model"] == "linear" and r["branch"] == left]
        right_rows = [r for r in rows if r["model"] == "linear" and r["branch"] == right]
        for a, b in zip(left_rows, right_rows, strict=True):
            for key in ("p_right", "action", "reward", "q_reward", "action_uniform", "reward_uniform"):
                assert a[key] == b[key]


def test_resource_totals_sum_phase_local_calls_without_duplicating_inherited_prefix(completed):
    resources = read_json(completed["out"] / "model_resources.json")
    stages = read_json(completed["out"] / "stages.json")
    assert len(resources) == len(stages) == 40
    assert sum(r["head_update_calls"] for r in resources) == sum(r["head_update_calls"] for r in stages) == 192
    assert sum(r["internal_update_calls"] for r in resources) == sum(r["internal_update_calls"] for r in stages) == 144
    assert sum(r["actual_trials"] for r in stages) == 320
    evidence = read_json(completed["out"] / "clone_evidence.json")
    assert len(evidence) == 32 and all(row["all_native_checkpoint_fields_equal"] is True for row in evidence)


def test_checkpoint_curves_keep_phase_displacement_distinct_from_total_displacement(completed):
    curves = read_csv(completed["out"] / "learning_curves.csv")
    assert len(curves) == 2 * 4 * 5 * 3 * 2
    for row in curves:
        if row["phase"] == "phase2" and row["trial"] == "0":
            assert float(row["phase_internal_change_l2"]) == float(row["phase_readout_change_l2"]) == 0.
            assert row["head_updates"] == row["internal_updates"] == "0"
        if row["branch"] in ("frozen", "internal_only"):
            assert float(row["phase_readout_change_l2"]) == 0.
        if row["branch"] in ("frozen", "head_only") or row["model"] == "linear":
            assert float(row["phase_internal_change_l2"]) == 0.


def test_complete_run_hashes_every_primary_file_and_reports_all_failed_thresholds(completed):
    output = completed["out"]
    metadata, summary = read_json(output / "metadata.json"), read_json(output / "summary.json")
    hashes = read_json(output / "artifact_hashes.json")
    assert all(study.digest(output / relative) == sha for relative, sha in hashes.items())
    assert set(hashes) == {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file() and p.name != "artifact_hashes.json"}
    assert metadata["registration_sha256"] == study.digest(completed["registration"])
    assert metadata["human_or_species_comparison"] is False and summary["prefix_failure_exclusion"] is False
    assert len(summary["outcomes"]) == 2 * 4 * 4 * 2
    assert len(summary["paired_contrasts"]) == 2 * 4 * 2 * 8 * 5
    assert set(summary["primary_hypotheses"]) == {"H_I", "H_H"}
    assert all(row["n_seeds"] == 2 for row in summary["primary_hypotheses"].values())
    assert summary["failures"]


def test_midrun_failure_preserves_first_seed_actual_rows_partial_curves_and_failed_metadata(fixture, monkeypatch):
    register(fixture)
    original = study.make_splits
    def fail_second(config, seed):
        if seed == 18:
            raise RuntimeError("deliberate second-seed failure")
        return original(config, seed)
    monkeypatch.setattr(study, "make_splits", fail_second)
    with pytest.raises(RuntimeError, match="deliberate second-seed failure"):
        run(fixture)
    output = fixture["out"]
    metadata = read_json(output / "metadata.json")
    assert metadata["status"] == "failed" and metadata["error_type"] == "RuntimeError"
    assert metadata["actual_choice_feedback_trials"] == 160
    assert len(read_csv(output / "seed_17/trials.csv")) == 160
    assert len(read_csv(output / "partial_learning_curves.csv")) == 120
    assert all(study.digest(output / relative) == sha for relative, sha in read_json(output / "artifact_hashes.json").items())
