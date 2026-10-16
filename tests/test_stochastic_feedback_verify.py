"""Independent sampled-reward audit: invented engineering fixtures only."""
from copy import deepcopy
import ast
import csv
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_stochastic_feedback.py"
SPEC = importlib.util.spec_from_file_location("independent_stochastic_feedback_verifier", SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def test_runtime_imports_only_immutable_independent_auditors_and_standard_libraries():
    tree = ast.parse(SCRIPT.read_text(encoding="utf8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported == {"__future__", "datetime", "pathlib", "drosolearn.encoding_audit",
                        "drosolearn.linearity_audit", "drosolearn.bandit_reversal_audit"}
    assert {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names} == {
        "argparse", "csv", "hashlib", "json", "re", "numpy"}


@pytest.mark.parametrize("stage", ["linear", "network"])
def test_scientific_configs_are_validated_without_running_scientific_seeds(stage):
    cfg = VERIFY.read_json(ROOT / f"configs/stochastic_feedback_{stage}_v1.json")
    VERIFY.config_check(cfg)
    assert len(cfg["seeds"]) == 10


@pytest.mark.parametrize("stage,seed", [("linear", 10001), ("linear", 11001),
                                       ("network", 10010), ("network", 11010)])
def test_declared_scientific_seeds_cannot_be_used_for_engineering_training(stage, seed):
    cfg = VERIFY.read_json(ROOT / f"configs/stochastic_feedback_{stage}_v1.json")
    cfg.update(scope="integration_test", seeds=[seed])
    with pytest.raises(ValueError, match="Scientific seeds forbidden"):
        VERIFY.config_check(cfg)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1, True])
def test_nonfinite_out_of_range_or_boolean_threshold_is_rejected(value):
    cfg = VERIFY.read_json(ROOT / "configs/stochastic_feedback_linear_v1.json")
    cfg["criteria"]["acquisition_accuracy"] = value
    with pytest.raises(ValueError):
        VERIFY.config_check(cfg)


def linear_args():
    return {"stimulus": np.asarray([[.25, 0., 0., 0., 0., 0., 0.], [.5, 0., 0., 0., 0., 0., 0.]]),
            "action_uniform": .1, "reward_uniform": .1, "larger": True,
            "env": VERIFY.ENVIRONMENTS[1], "encoder": np.empty((0, 7)), "bias": np.empty(0),
            "pre": np.empty(0, dtype=int), "post": np.empty(0, dtype=int), "weights": np.empty(0),
            "head": np.zeros(7), "scale": .25, "linear": True, "plastic": False,
            "frozen": False, "head_lr": .1, "internal_lr": .01, "upper": np.empty(0)}


@pytest.mark.parametrize("environment", VERIFY.ENVIRONMENTS)
@pytest.mark.parametrize("chosen_optimal", [False, True])
def test_sampled_reward_uses_selected_action_probability_and_strict_threshold(environment, chosen_optimal):
    args = linear_args()
    args.update(env=environment, action_uniform=.1 if chosen_optimal else .9)
    q = environment["q_optimal"] if chosen_optimal else environment["q_other"]
    args["reward_uniform"] = q
    actual, _, _ = VERIFY.independent_step(**args)
    assert actual["action"] == int(chosen_optimal)
    assert actual["q_reward"] == q
    assert actual["reward"] == 0
    if q > 0:
        args["reward_uniform"] = np.nextafter(q, 0.)
        below, _, _ = VERIFY.independent_step(**args)
        assert below["reward"] == 1


@pytest.mark.parametrize("reward_uniform,reward,expected_head", [(.1, 1, .025), (.9, 0, -.025)])
def test_manual_linear_update_uses_actual_reward_not_privileged_rule(reward_uniform, reward, expected_head):
    args = linear_args()
    args["reward_uniform"] = reward_uniform
    record, weights, head = VERIFY.independent_step(**args)
    assert record["action"] == 1 and record["reward"] == reward
    assert record["optimizer_factor"] == (.25 if reward else -.25)
    assert weights.size == 0
    np.testing.assert_array_equal(head[1:], np.zeros(6))
    assert head[0] == expected_head


@pytest.mark.parametrize("environment", VERIFY.ENVIRONMENTS)
def test_frozen_step_receives_feedback_but_retains_exact_initial_state(environment):
    args = linear_args()
    args.update(env=environment, frozen=True)
    record, weights, head = VERIFY.independent_step(**args)
    np.testing.assert_array_equal(head, args["head"])
    np.testing.assert_array_equal(weights, args["weights"])
    assert record["readout_step_l2"] == record["internal_step_l2"] == 0.
    assert record["p_right"] == .5


@pytest.mark.parametrize("phase_larger", [False, True])
def test_uninformative_expected_reward_is_half_even_with_extreme_direction_preference(phase_larger):
    counts = np.asarray([[2, 4], [4, 2]])
    result = VERIFY.metrics(np.asarray([.99, .01]), counts, phase_larger, VERIFY.ENVIRONMENTS[2])
    assert result["expected_reward"] == .5
    assert result["p_rule"] == pytest.approx(.99 if phase_larger else .01)
    assert result["greedy_rule"] == float(phase_larger)


def test_unique_utc_and_json_types_are_preserved(tmp_path):
    bad = tmp_path / "duplicate.json"
    bad.write_text('{"passed": true, "passed": false}', encoding="utf8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        VERIFY.read_json(bad)
    with pytest.raises(ValueError):
        VERIFY.utc("2026-10-16T00:00:00")
    with pytest.raises(ValueError):
        VERIFY.utc("2026-10-16T00:00:00+09:00")
    with pytest.raises(ValueError):
        VERIFY.equal(1, True, "truth is not numeric one")


@pytest.mark.parametrize("internal_lr", [.01, 100.])
def test_internal_eligibility_uses_old_head_fixed_scale_and_original_upper_bound(internal_lr):
    args = linear_args()
    encoder = np.zeros((1, 7))
    encoder[0, 0] = 1.
    args.update(linear=False, plastic=True, encoder=encoder, bias=np.zeros(1),
                pre=np.asarray([0]), post=np.asarray([0]), weights=np.asarray([.7]),
                head=np.asarray([.6]), internal_lr=internal_lr, upper=np.asarray([1.4]))
    record, weights, head = VERIFY.independent_step(**args)
    probability = 1 / (1 + np.exp(-.42))
    factor = (1 - probability) * .5
    assert record["p_right"] == pytest.approx(probability)
    assert head[0] == pytest.approx(.6 + .1 * factor * .7)
    expected = min(1.4, .7 + internal_lr * factor * .6)
    assert weights[0] == pytest.approx(expected)
    if internal_lr == .01:
        assert weights[0] != pytest.approx(.7 + internal_lr * factor * head[0], abs=1e-9)


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "sub\\file", ""])
def test_hash_manifest_cannot_escape_registered_root(tmp_path, relative):
    with pytest.raises(ValueError):
        VERIFY.safe(tmp_path, relative)


@pytest.fixture(scope="module", params=["linear", "network"])
def audited_fixture(request, tmp_path_factory):
    # This helper copies source/docs/projection only and constructs completely
    # invented parent gates. It never reads historical human or model outcomes.
    from test_stochastic_feedback_run import make_fixture, make_linear_gate, synthetic_config, save_json
    from drosolearn import stochastic_feedback as producer
    monkeypatch = pytest.MonkeyPatch()
    try:
        fixture = make_fixture(tmp_path_factory.mktemp(f"independent-{request.param}"), monkeypatch)
        if request.param == "network":
            cfg = synthetic_config("network", seeds=(17,))
            cfg["frozen_files"] = fixture["cfg"]["frozen_files"]
            fixture.update(cfg=cfg, config=fixture["root"] / "configs/stochastic_feedback_network_v1.json")
            save_json(fixture["config"], cfg)
            make_linear_gate(fixture["root"])
        producer.register(fixture["root"], fixture["config"], fixture["registration"])
        producer.run(fixture["root"], fixture["config"], fixture["registration"], fixture["out"])
        report = VERIFY.verify(fixture["root"], fixture["registration"], fixture["out"])
        assert report["status"] == "passed", report["errors"]
        yield fixture
    finally:
        monkeypatch.undo()


def clone_fixture(original, tmp_path):
    root = tmp_path / "rehashed_corruption"
    shutil.copytree(original["root"], root)
    return {**original, "root": root, "registration": root / original["registration"].relative_to(original["root"]),
            "out": root / original["out"].relative_to(original["root"])}


def rehash(fixture, name):
    manifest_path = fixture["out"] / "artifact_hashes.json"
    manifest = VERIFY.read_json(manifest_path)
    manifest[name] = VERIFY.digest(fixture["out"] / name)
    manifest_path.write_text(json.dumps(manifest), encoding="utf8")


def replace_json(fixture, name, value):
    (fixture["out"] / name).write_text(json.dumps(value, allow_nan=False), encoding="utf8")
    rehash(fixture, name)


def read_csv(fixture, name):
    with (fixture["out"] / name).open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))


def replace_csv(fixture, name, rows):
    with (fixture["out"] / name).open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rehash(fixture, name)


def rejected(fixture):
    report = VERIFY.verify(fixture["root"], fixture["registration"], fixture["out"])
    assert report["status"] == "failed" and report["passed"] is False and report["errors"]


def test_complete_independent_replay_preserves_every_primary_byte(audited_fixture):
    manifest = VERIFY.read_json(audited_fixture["out"] / "artifact_hashes.json")
    report = VERIFY.verify(audited_fixture["root"], audited_fixture["registration"], audited_fixture["out"])
    assert report["passed"], report["errors"]
    assert report["summary_sha256"] == VERIFY.digest(audited_fixture["out"] / "summary.json")
    assert report["counts"]["actual_choice_feedback_trials"] == audited_fixture["cfg"]["actual_choice_feedback_trials"]
    assert report["counts"]["head_update_calls"] == audited_fixture["cfg"]["expected_head_update_calls"]
    assert report["counts"]["internal_update_calls"] == audited_fixture["cfg"]["expected_internal_update_calls"]
    assert report["counts"]["privileged_update_calls"] == 0
    assert all(VERIFY.digest(audited_fixture["out"] / p) == sha for p, sha in manifest.items())
    assert not (audited_fixture["out"] / "verification.json").exists()


@pytest.mark.parametrize("field", ["action", "reward", "q_reward", "p_right", "chosen_count", "global_trial",
                                  "action_uniform", "reward_uniform", "optimizer_factor", "readout_step_l2",
                                  "head_updates", "head_update_called"])
def test_rehashed_actual_action_reward_uniform_update_and_identity_corruption_is_rejected(audited_fixture, tmp_path, field):
    fixture = clone_fixture(audited_fixture, tmp_path)
    name = "seed_17/trials.csv"
    rows = read_csv(fixture, name)
    row = next(r for r in rows if r["condition"] == "linear" and r["environment"] == "probabilistic"
               and r["phase"] == "phase2" and r["trial"] == "2")
    if field in {"action", "reward", "head_update_called"}:
        row[field] = 1 - int(row[field])
    elif field in {"chosen_count", "global_trial", "head_updates"}:
        row[field] = int(row[field]) + 1
    else:
        row[field] = float(row[field]) + .013
    replace_csv(fixture, name, rows)
    rejected(fixture)


@pytest.mark.parametrize("field", ["readout_weights", "choice_probability", "decision_activity", "feature_scale",
                                  "iid_p_right", "head_updates", "initial_internal_weights", "internal_upper_bounds",
                                  "incoming_normalizer"])
def test_rehashed_checkpoint_state_scale_case_probability_and_original_bounds_corruption_is_rejected(audited_fixture, tmp_path, field):
    fixture = clone_fixture(audited_fixture, tmp_path)
    condition = "plastic" if fixture["cfg"]["stage"] == "network" else "linear"
    name = f"seed_17/{condition}/probabilistic/phase2_0008.npz"
    with np.load(fixture["out"] / name, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    if arrays[field].size:
        arrays[field].flat[0] += 1 if field == "head_updates" else .013
    else:
        arrays[field] = np.asarray([.013])
    np.savez_compressed(fixture["out"] / name, **arrays)
    rehash(fixture, name)
    rejected(fixture)


@pytest.mark.parametrize("field", ["action_phase1_uniforms", "reward_phase1_uniforms", "reward_phase2_uniforms",
                                  "train_phase1_nuisance", "train_phase2_nuisance", "iid_nuisance"])
def test_rehashed_shared_stream_and_train_eval_input_corruption_is_rejected(audited_fixture, tmp_path, field):
    fixture = clone_fixture(audited_fixture, tmp_path)
    name = "seed_17/stimuli.npz"
    with np.load(fixture["out"] / name, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    arrays[field].flat[0] += .013
    np.savez_compressed(fixture["out"] / name, **arrays)
    rehash(fixture, name)
    rejected(fixture)


@pytest.mark.parametrize("section,field", [
    ("outcomes", "phase1_start"), ("outcomes", "phase1_end"), ("outcomes", "phase1_gain"),
    ("outcomes", "phase1_threshold_met"), ("outcomes", "phase1_acquired"),
    ("outcomes", "phase2_start"), ("outcomes", "phase2_end"), ("outcomes", "phase2_gain"),
    ("outcomes", "phase2_raw_auc"), ("outcomes", "phase2_baseline_adjusted_auc"),
    ("outcomes", "qualified_adaptation"), ("outcomes", "first_observed_threshold_checkpoint"),
    ("environment_contrasts", "difference"), ("family_verdicts", "phase1_acquisition_count"),
    ("environment_contrast_statistics", "mean"),
])
def test_every_summary_outcome_gate_gain_area_and_all_seed_contrast_is_recomputed(audited_fixture, tmp_path, section, field):
    fixture = clone_fixture(audited_fixture, tmp_path)
    summary = VERIFY.read_json(fixture["out"] / "summary.json")
    row = summary[section][0]
    previous = row[field]
    row[field] = not previous if type(previous) is bool else 0 if previous is None else previous + .013
    replace_json(fixture, "summary.json", summary)
    rejected(fixture)


@pytest.mark.parametrize("field", ["probabilistic_acquisition_count", "mean_paired_reward_gap", "criterion_met",
                                  "fraction_criterion_met", "gap_criterion_met", "scope"])
def test_rehashed_primary_hypothesis_cannot_be_rewritten(audited_fixture, tmp_path, field):
    fixture = clone_fixture(audited_fixture, tmp_path)
    summary = VERIFY.read_json(fixture["out"] / "summary.json")
    old = summary["primary_hypothesis"][field]
    summary["primary_hypothesis"][field] = not old if type(old) is bool else old + "changed" if type(old) is str else old + .013
    replace_json(fixture, "summary.json", summary)
    rejected(fixture)


@pytest.mark.parametrize("filename,field", [("stages.json", "actual_reward_count"),
                                           ("stages.json", "actual_designated_choice_count"),
                                           ("model_resources.json", "trainable_readout_parameters"),
                                           ("model_resources.json", "head_update_calls")])
def test_actual_feedback_counts_and_storage_vs_trainable_resource_fields_are_replayed(audited_fixture, tmp_path, filename, field):
    fixture = clone_fixture(audited_fixture, tmp_path)
    rows = VERIFY.read_json(fixture["out"] / filename)
    rows[0][field] += 1
    replace_json(fixture, filename, rows)
    rejected(fixture)


def test_missing_failed_cases_are_not_silently_removed_from_rehashed_summary(audited_fixture, tmp_path):
    fixture = clone_fixture(audited_fixture, tmp_path)
    summary = VERIFY.read_json(fixture["out"] / "summary.json")
    assert summary["failures"]
    summary["failures"].pop()
    replace_json(fixture, "summary.json", summary)
    rejected(fixture)


def test_uninformative_threshold_crossing_cannot_be_promoted_to_rule_learning(audited_fixture, tmp_path):
    fixture = clone_fixture(audited_fixture, tmp_path)
    summary = VERIFY.read_json(fixture["out"] / "summary.json")
    row = next(r for r in summary["outcomes"] if r["environment"] == "uninformative")
    assert row["phase1_acquired"] is None
    row["phase1_acquired"] = True
    replace_json(fixture, "summary.json", summary)
    rejected(fixture)


def test_verifier_never_calls_producer_training_forward_data_reward_or_summary(audited_fixture, monkeypatch):
    from drosolearn import stochastic_feedback as producer
    def forbidden(*args, **kwargs):
        raise AssertionError("Producer computation called by independent verifier")
    for name in ("make_splits", "reward_probability", "state_pack", "evaluate", "summarize", "gate"):
        monkeypatch.setattr(producer, name, forbidden)
    for learner in (producer.LinearLearner, producer.NormalizedLearner):
        for name in ("features", "probabilities", "update", "snapshot", "raw_activities"):
            monkeypatch.setattr(learner, name, forbidden)
    report = VERIFY.verify(audited_fixture["root"], audited_fixture["registration"], audited_fixture["out"])
    assert report["passed"], report["errors"]
