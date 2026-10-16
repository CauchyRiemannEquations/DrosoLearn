"""End-to-end actual-reward learning uses tiny development fixtures only."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from drosolearn import stochastic_feedback as study
from drosolearn.reward_linear import LinearLearner, NormalizedLearner


REPO = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf8")


def read_csv(path):
    with Path(path).open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_config(stage="linear", seeds=(17, 18)):
    cfg = read_json(REPO / f"configs/stochastic_feedback_{stage}_v1.json")
    cfg.update(scope="integration_test", seeds=list(seeds), trials_per_phase=8,
               checkpoint_every=4, eval_trials=8, probe_rows=2)
    cfg["actual_choice_feedback_trials"] = len(seeds) * len(cfg["conditions"]) * 3 * 16
    cfg["expected_head_update_calls"] = len(seeds) * (len(cfg["conditions"]) - 1) * 3 * 16
    cfg["expected_internal_update_calls"] = len(seeds) * 3 * 3 * 16 if stage == "network" else 0
    return cfg


def make_fixture(tmp_path, monkeypatch):
    root = tmp_path / "synthetic_repo"
    root.mkdir()
    cfg = synthetic_config()
    for directory in ("src", cfg["graph"]):
        shutil.copytree(REPO / directory, root / directory)
    # Source/docs/projection only. Never copy previous human/model trial outputs.
    copied = (
        "configs/stochastic_feedback_linear_v1.json", "configs/stochastic_feedback_network_v1.json",
        "docs/stochastic-feedback-protocol.md", "docs/stochastic-feedback-design.md",
        "data/source-manifest.json", "data/LICENSE.md", "requirements-lock.txt", "pyproject.toml",
        "scripts/run_stochastic_feedback.py", "scripts/verify_stochastic_feedback.py",
        "tests/test_stochastic_feedback.py", "tests/test_stochastic_feedback_run.py",
        "tests/test_stochastic_feedback_verify.py",
    )
    for name in copied:
        source = REPO / name
        assert source.is_file(), f"Wait until new implementation is stable before tests: {name}"
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    cfg["frozen_files"] = list(dict.fromkeys(cfg.get("frozen_files", []) + [
        *copied, "src/drosolearn/stochastic_feedback.py", "src/drosolearn/reward_linear.py",
        "src/drosolearn/model.py", "src/drosolearn/data.py", f"{cfg['graph']}/provenance.json",
    ]))
    for relative in cfg["frozen_files"]:
        assert not relative.startswith("results/"), "Test source copying cannot read old behavioral results"
        destination = root / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / relative, destination)
    cfg["calibration_gate_identity"] = make_calibration_gate(root)
    config = root / "configs/stochastic_feedback_linear_v1.json"
    save_json(config, cfg)
    actual_git = subprocess.check_output

    def synthetic_git(command, **kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            return "c" * 40 + "\n"
        return actual_git(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", synthetic_git)
    original_splits = study.make_splits

    def guarded_splits(cfg, seed):
        assert cfg["scope"] == "integration_test"
        assert seed in (17, 18) and seed not in range(10001, 10011) and seed not in range(11001, 11011)
        assert cfg["trials_per_phase"] == 8
        return original_splits(cfg, seed)

    monkeypatch.setattr(study, "make_splits", guarded_splits)
    return {"root": root, "cfg": cfg, "config": config, "registration": root / "registration.json",
            "out": root / "fresh_run"}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    return make_fixture(tmp_path, monkeypatch)


def make_calibration_gate(root):
    """Synthetic byte-bound gate, unrelated to any production claim/result."""
    config = root / "configs/metric_calibration_v1.json"
    registration = root / "results/registration_metric_calibration_v1.json"
    base = root / "results/metric_calibration_v1"
    save_json(config, {"synthetic_gate_fixture": True})
    save_json(registration, {"synthetic_gate_fixture": True})
    summary = {"study_id": "metric_calibration_v1", "synthetic_gate_fixture": True, "analytic_claims": {
        name: {"criterion_met": True} for name in (
            "H1_no_learning_feedback_difference", "H2_no_learning_high_persistence", "H3_pooling_masks_phase_difference")}}
    save_json(base / "summary.json", summary)
    save_json(base / "artifact_hashes.json", {"summary.json": digest(base / "summary.json")})
    save_json(base / "verification.json", {"status": "passed", "registration_sha256": digest(registration),
        "config_sha256": digest(config), "verified_at_utc": "2026-01-01T00:00:00+00:00", "synthetic_gate_fixture": True})
    return {"summary_sha256": digest(base / "summary.json"),
            "verification_sha256": digest(base / "verification.json"),
            "registration_sha256": digest(registration),
            "artifact_hashes_sha256": digest(base / "artifact_hashes.json"), "published_commit": "d" * 40}


def make_linear_gate(root):
    """Handmade passing linear proof exists solely inside a temporary test root."""
    registration = root / "results/registration_stochastic_feedback_linear_v1.json"
    base = root / "results/stochastic_feedback_linear_v1"
    save_json(registration, {"synthetic_gate_fixture": True})
    save_json(base / "summary.json", {"study_id": "stochastic_feedback_linear_v1",
        "primary_hypothesis": {"criterion_met": True}, "synthetic_gate_fixture": True})
    save_json(base / "verification.json", {"status": "passed", "summary_sha256": digest(base / "summary.json"),
        "registration_sha256": digest(registration),
        "config_sha256": digest(root / "configs/stochastic_feedback_linear_v1.json"),
        "verified_at_utc": "2026-01-01T00:00:00+00:00", "synthetic_gate_fixture": True})


def register(fixture):
    return study.register(fixture["root"], fixture["config"], fixture["registration"])


def run(fixture):
    return study.run(fixture["root"], fixture["config"], fixture["registration"], fixture["out"])


def test_registration_hashes_sources_without_generating_or_training_on_trials(fixture, monkeypatch):
    monkeypatch.setattr(study, "make_splits", lambda *a, **k: pytest.fail("Registration generated task trials"))
    registration = register(fixture)
    assert registration == read_json(fixture["registration"])
    assert not fixture["out"].exists()
    assert digest(fixture["config"]) in json.dumps(registration)


@pytest.mark.parametrize("target", ["config", "source", "graph"])
def test_registered_source_config_or_projection_tampering_blocks_run(fixture, target):
    register(fixture)
    path = {"config": fixture["config"],
            "source": fixture["root"] / "src/drosolearn/stochastic_feedback.py",
            "graph": fixture["root"] / fixture["cfg"]["graph"] / "provenance.json"}[target]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        run(fixture)
    assert not fixture["out"].exists()


def test_previous_registration_and_existing_result_directory_cannot_be_overwritten(fixture):
    register(fixture)
    before = fixture["registration"].read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        register(fixture)
    assert fixture["registration"].read_bytes() == before
    fixture["out"].mkdir()
    sentinel = fixture["out"] / "earlier_result.bin"
    sentinel.write_bytes(b"preserve earlier result")
    with pytest.raises((ValueError, FileExistsError)):
        run(fixture)
    assert list(fixture["out"].iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"preserve earlier result"


def test_network_registration_is_blocked_without_linear_gate_evidence(fixture):
    cfg = synthetic_config("network")
    path = fixture["root"] / "configs/stochastic_feedback_network_v1.json"
    save_json(path, cfg)
    with pytest.raises((ValueError, FileNotFoundError)):
        study.register(fixture["root"], path, fixture["root"] / "network_registration.json")
    assert not (fixture["root"] / "network_registration.json").exists()


def test_actual_learning_update_interface_has_no_target_probability_or_evaluation_arguments(fixture, monkeypatch):
    linear_original = LinearLearner.update
    calls = []

    def linear_update(self, inputs, action, reward):
        assert np.asarray(inputs).shape == (2, 7)
        assert type(action) is int and action in (0, 1)
        assert type(reward) is int and reward in (0, 1)
        calls.append((action, reward))
        return linear_original(self, inputs, action, reward)

    def normalized_update(self, inputs, action, reward):
        pytest.fail("Linear diagnostic unexpectedly trained a network learner")

    monkeypatch.setattr(LinearLearner, "update", linear_update)
    monkeypatch.setattr(NormalizedLearner, "update", normalized_update)
    register(fixture)
    result = run(fixture)
    assert len(calls) == 2 * 3 * 2 * 8  # Frozen condition calls no update.
    assert result["status"] == "completed"


def test_complete_actual_reward_run_preserves_uniforms_targets_resources_and_frozen_control(fixture):
    register(fixture)
    result = run(fixture)
    assert result["status"] == "completed"
    assert result["actual_choice_feedback_trials"] == 192
    assert result["head_update_calls"] == 96
    assert result["internal_update_calls"] == result["privileged_update_calls"] == 0
    assert result["human_or_species_comparison"] is False
    assert result["registration_sha256"] == digest(fixture["registration"])
    all_rows = []
    for seed in (17, 18):
        base = fixture["out"] / f"seed_{seed}"
        rows = read_csv(base / "trials.csv")
        assert len(rows) == 96
        all_rows.extend(rows)
        with np.load(base / "stimuli.npz", allow_pickle=False) as stimuli:
            for phase, action_offset, reward_offset in (("phase1", 510000, 530000), ("phase2", 520000, 540000)):
                action_uniforms = np.random.default_rng(seed + action_offset).random(8)
                reward_uniforms = np.random.default_rng(seed + reward_offset).random(8)
                np.testing.assert_array_equal(stimuli[f"action_{phase}_uniforms"], action_uniforms)
                np.testing.assert_array_equal(stimuli[f"reward_{phase}_uniforms"], reward_uniforms)
                for row in (row for row in rows if row["phase"] == phase):
                    index = int(row["trial"]) - 1
                    assert row["global_trial"] == str(index + 1 + (8 if phase == "phase2" else 0))
                    assert row["family_id"] == stimuli[f"train_{phase}_family_ids"][index]
                    counts = stimuli[f"train_{phase}_counts"][index]
                    assert [int(row["left_count"]), int(row["right_count"])] == counts.tolist()
                    assert float(row["action_uniform"]) == action_uniforms[index]
                    assert float(row["reward_uniform"]) == reward_uniforms[index]
                    action = int(action_uniforms[index] < float(row["p_right"]))
                    assert int(row["action"]) == action
                    assert int(row["chosen_count"]) == counts[action]
                    target = int(np.argmax(counts) if phase == "phase1" else np.argmin(counts))
                    env = next(env for env in fixture["cfg"]["environments"] if env["id"] == row["environment"])
                    q = env["q_optimal"] if action == target else env["q_other"]
                    reward = int(reward_uniforms[index] < q)
                    assert float(row["q_reward"]) == q
                    assert int(row["reward"]) == reward
                    assert float(row["optimizer_factor"]) == pytest.approx(
                        (action - float(row["p_right"])) * (reward - .5))
                    assert int(row["head_update_called"]) == int(row["condition"] != "frozen")
                    assert int(row["internal_update_called"]) == 0
                    assert float(row["internal_step_l2"]) == 0
                    if row["condition"] == "frozen":
                        assert float(row["p_right"]) == .5
                        assert float(row["readout_step_l2"]) == 0
                        assert int(row["head_updates"]) == int(row["internal_updates"]) == 0
    assert sum(int(row["head_update_called"]) for row in all_rows) == 96
    resources = read_json(fixture["out"] / "model_resources.json")
    assert len(resources) == 12
    assert all(row["internal_parameters"] == row["trainable_internal_parameters"] == 0 for row in resources)
    assert all(row["readout_parameters"] == 7 for row in resources)
    assert sum(row["head_update_calls"] for row in resources) == 96
    assert sum(row["actual_trials"] for row in read_json(fixture["out"] / "stages.json")) == 192
    assert len(read_json(fixture["out"] / "summary.json")["outcomes"]) == 24
    hashes = read_json(fixture["out"] / "artifact_hashes.json")
    assert all(digest(fixture["out"] / relative) == sha for relative, sha in hashes.items())
    assert not list(fixture["out"].rglob("expected_updates.csv"))


def test_phase_transition_preserves_state_and_scaling_while_only_evaluation_direction_changes(fixture):
    register(fixture)
    run(fixture)
    curves = read_csv(fixture["out"] / "learning_curves.csv")
    for seed in (17, 18):
        with np.load(fixture["out"] / f"seed_{seed}/stimuli.npz", allow_pickle=False) as data:
            x = np.concatenate((data["train_phase1_counts"][..., None] / 8,
                                .1 * (data["train_phase1_nuisance"] - .5)), axis=-1)
        expected_scale = np.sqrt(np.mean(np.sum((x[:, 1] - x[:, 0]) ** 2, axis=1)))
        for condition in ("linear", "frozen"):
            for env in ("deterministic", "probabilistic", "uninformative"):
                base = fixture["out"] / f"seed_{seed}" / condition / env
                with np.load(base / "phase1_0008.npz", allow_pickle=False) as first, \
                        np.load(base / "phase2_0000.npz", allow_pickle=False) as second:
                    assert set(first.files) == set(second.files)
                    for key in first.files:
                        np.testing.assert_array_equal(first[key], second[key])
                    assert float(first["feature_scale"]) == pytest.approx(expected_scale)
                context = [row for row in curves if int(row["seed"]) == seed and row["condition"] == condition
                           and row["environment"] == env and row["split"] == "iid"]
                end = next(row for row in context if row["phase"] == "phase1" and row["trial"] == "8")
                start = next(row for row in context if row["phase"] == "phase2" and row["trial"] == "0")
                assert float(end["p_rule"]) + float(start["p_rule"]) == pytest.approx(1)
                assert end["head_updates"] == start["head_updates"]
                assert end["internal_updates"] == start["internal_updates"]


@pytest.mark.parametrize("gate_field", ["status", "config_binding", "registration_binding", "summary_binding", "hypothesis"])
def test_failed_or_unbound_calibration_gate_blocks_linear_registration(fixture, gate_field):
    base = fixture["root"] / "results/metric_calibration_v1"
    if gate_field == "summary_binding":
        path = base / "artifact_hashes.json"
        value = {"summary.json": "0" * 64}
    elif gate_field == "hypothesis":
        path = base / "summary.json"
        value = read_json(path)
        value["analytic_claims"]["H1_no_learning_feedback_difference"]["criterion_met"] = False
    else:
        path = base / "verification.json"
        value = read_json(path)
        value[{"status": "status", "config_binding": "config_sha256",
               "registration_binding": "registration_sha256"}[gate_field]] = "failed"
    save_json(path, value)
    with pytest.raises(ValueError):
        register(fixture)
    assert not fixture["registration"].exists()


def test_midrun_failure_preserves_metadata_previous_seed_trials_and_partial_curves(fixture, monkeypatch):
    register(fixture)
    original = study.make_splits

    def fail_second_seed(cfg, seed):
        if seed == 18:
            raise RuntimeError("synthetic deliberate second-seed failure")
        return original(cfg, seed)

    monkeypatch.setattr(study, "make_splits", fail_second_seed)
    with pytest.raises(RuntimeError, match="deliberate second-seed failure"):
        run(fixture)
    metadata = read_json(fixture["out"] / "metadata.json")
    assert metadata["status"] == "failed" and metadata["error_type"] == "RuntimeError"
    assert metadata["actual_choice_feedback_trials"] == 96
    assert len(read_csv(fixture["out"] / "seed_17/trials.csv")) == 96
    assert len(read_csv(fixture["out"] / "partial_learning_curves.csv")) == 72
    hashes = read_json(fixture["out"] / "artifact_hashes.json")
    assert all(digest(fixture["out"] / relative) == sha for relative, sha in hashes.items())


def test_replacing_summary_and_artifact_map_cannot_reuse_older_passing_verification(fixture):
    base = fixture["root"] / "results/metric_calibration_v1"
    summary = read_json(base / "summary.json")
    summary["unverified_replacement"] = True
    save_json(base / "summary.json", summary)
    save_json(base / "artifact_hashes.json", {"summary.json": digest(base / "summary.json")})
    # The passing verification status remains; it did not verify these new bytes.
    assert read_json(base / "verification.json")["status"] == "passed"
    with pytest.raises(ValueError):
        register(fixture)
    assert not fixture["registration"].exists()


def test_tiny_network_path_runs_all_conditions_with_separate_head_internal_and_activity_records(fixture):
    make_linear_gate(fixture["root"])
    cfg = synthetic_config("network", seeds=(17,))
    cfg["frozen_files"] = fixture["cfg"]["frozen_files"].copy()
    fixture["cfg"] = cfg
    fixture["config"] = fixture["root"] / "configs/stochastic_feedback_network_v1.json"
    save_json(fixture["config"], cfg)
    register(fixture)
    result = run(fixture)
    assert result["actual_choice_feedback_trials"] == 384
    assert result["head_update_calls"] == 336
    assert result["internal_update_calls"] == 144
    assert result["privileged_update_calls"] == 0
    rows = read_csv(fixture["out"] / "seed_17/trials.csv")
    assert set(row["condition"] for row in rows) == set(cfg["conditions"])
    resources = read_json(fixture["out"] / "model_resources.json")
    assert len(resources) == 24
    assert sum(row["head_update_calls"] for row in resources) == 336
    assert sum(row["internal_update_calls"] for row in resources) == 144
    for env in ("deterministic", "probabilistic", "uninformative"):
        for fixed, plastic in (("fixed", "plastic"), ("rewired_fixed", "rewired_plastic"), ("ann_fixed", "ann_plastic")):
            base = fixture["out"] / "seed_17"
            with np.load(base / fixed / env / "phase1_0000.npz", allow_pickle=False) as first, \
                    np.load(base / plastic / env / "phase1_0000.npz", allow_pickle=False) as second:
                for key in ("internal_weights", "encoder", "bias", "pre", "post", "feature_scale", "readout_weights"):
                    np.testing.assert_array_equal(first[key], second[key])
            condition_rows = [row for row in rows if row["condition"] == plastic and row["environment"] == env]
            first_update = condition_rows[0]
            assert first_update["head_updates"] == first_update["internal_updates"] == "1"
            assert float(first_update["internal_step_l2"]) == 0  # Old head was initially zero.
        for condition in cfg["conditions"]:
            with np.load(fixture["out"] / "seed_17" / condition / env / "phase2_0008.npz", allow_pickle=False) as snap:
                if condition in ("linear", "frozen"):
                    assert snap["internal_weights"].size == 0
                elif condition not in study.PLASTIC:
                    np.testing.assert_array_equal(snap["internal_weights"], snap["initial_internal_weights"])
                else:
                    assert np.all(snap["internal_weights"] >= 0)
                    assert np.all(snap["internal_weights"] <= snap["internal_upper_bounds"])
                if condition == "frozen":
                    assert np.count_nonzero(snap["readout_weights"]) == 0
                    np.testing.assert_array_equal(snap["iid_p_right"], np.full(8, .5))
    summary = read_json(fixture["out"] / "summary.json")
    assert summary["primary_hypothesis"]["scope"] == "descriptive_independent_replication_not_retroactive_gate"
    assert len(summary["network_contrasts"]) == 7 * 3 * 2 * 8


@pytest.mark.parametrize("field,value", [
    ("source_git_commit", "not_a_commit"),
    ("registered_at_utc", "2999-01-01T00:00:00+00:00"),
    ("registered_at_utc", "2026-01-01T00:00:00"),
    ("config_sha256", "0" * 64),
])
def test_registration_provenance_tampering_is_blocked_before_output(fixture, field, value):
    register(fixture)
    record = read_json(fixture["registration"])
    record[field] = value
    save_json(fixture["registration"], record)
    with pytest.raises(ValueError):
        run(fixture)
    assert not fixture["out"].exists()


def test_actual_action_and_reward_use_strict_less_than_at_exact_uniform_ties(fixture, monkeypatch):
    """Inject exact ties only in this temporary engineering test, no real runs."""
    original = np.random.default_rng
    action_values = np.array([.5, .0, .99, .5, .2, .8, .5, .5])
    reward_values = np.array([.5, .8, .2, .0, .5, .8, .2, .999])
    action_seeds = {seed + offset for seed in (17, 18) for offset in (510000, 520000)}
    reward_seeds = {seed + offset for seed in (17, 18) for offset in (530000, 540000)}

    class PresetUniforms:
        def __init__(self, values):
            self.values = values

        def random(self, size):
            assert size == 8
            return self.values.copy()

    def controlled_rng(seed=None):
        if seed in action_seeds:
            return PresetUniforms(action_values)
        if seed in reward_seeds:
            return PresetUniforms(reward_values)
        return original(seed)

    register(fixture)
    monkeypatch.setattr(np.random, "default_rng", controlled_rng)
    run(fixture)
    rows = [row for seed in (17, 18)
            for row in read_csv(fixture["out"] / f"seed_{seed}/trials.csv")]
    action_ties = [row for row in rows if float(row["action_uniform"]) == float(row["p_right"])]
    reward_ties = [row for row in rows if float(row["reward_uniform"]) == float(row["q_reward"])]
    assert action_ties and reward_ties
    assert all(row["action"] == "0" for row in action_ties)
    assert all(row["reward"] == "0" for row in reward_ties)
    assert all(int(row["action"]) == int(float(row["action_uniform"]) < float(row["p_right"])) for row in rows)
    assert all(int(row["reward"]) == int(float(row["reward_uniform"]) < float(row["q_reward"])) for row in rows)
