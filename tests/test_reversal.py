"""Counterfactual histories, paired streams and prospective freeze invariants.

Tiny registered fixtures test the design rather than scientific task success.
"""
from pathlib import Path
import copy
import csv
import json
import shutil

import numpy as np
import pytest

from drosolearn.data import analytical_accuracy, assert_disjoint, make_dataset
from drosolearn.graph import load_projection
from drosolearn.model import Learner
from drosolearn.reversal import (
    HISTORIES, make_splits, normalized_auc, probe_metrics, register, run,
    summarize, validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


def fixture_config():
    cfg = json.loads((ROOT / "configs/reversal_diagnostic.json").read_text(encoding="utf-8"))
    cfg.update(scope="integration_test", seeds=[31], conditions=["fixed", "plastic"],
               trials_per_phase=8, checkpoint_every=4, eval_trials=8, probe_rows=4)
    return cfg


def test_external_config_paths_are_recorded_without_relative_path_failure(tmp_path):
    root, original, _ = prepare_fixture(tmp_path)
    external = tmp_path / "outside-config.json"
    external.write_bytes(original.read_bytes())
    registration = tmp_path / "outside-registration.json"
    register(root, external, registration)
    output = tmp_path / "external-run"
    run(root, external, registration, output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    args = metadata["reproduce_argv"]
    assert args[args.index("--config") + 1] == external.resolve().as_posix()


def prepare_fixture(path):
    root = path / "study"
    root.mkdir(parents=True)
    shutil.copytree(ROOT / "src", root / "src")
    graph = fixture_config()["graph"]
    shutil.copytree(ROOT / graph, root / graph)
    for name in ("data/source-manifest.json", "data/LICENSE.md", "pyproject.toml",
                 "requirements-lock.txt", "docs/reversal-protocol.md"):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    config = root / "config.json"
    config.write_text(json.dumps(fixture_config()) + "\n", encoding="utf-8")
    registration = root / "registration.json"
    register(root, config, registration)
    return root, config, registration


@pytest.fixture(scope="module")
def completed_reversal_fixture(tmp_path_factory):
    root, config, registration = prepare_fixture(tmp_path_factory.mktemp("reversal"))
    output = root / "fixture-output"
    summary = run(root, config, registration, output)
    return root, config, registration, output, summary


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def snapshot(output, condition, history, phase, trial):
    with np.load(output / "seed_31" / condition / history / f"{phase}_{trial:04d}.npz") as data:
        return {name: data[name].copy() for name in data.files}


@pytest.mark.parametrize("field,value", [
    ("histories", list(reversed(HISTORIES))),
    ("seeds", [31, 31]),
    ("conditions", ["frozen"]),
    ("trials_per_phase", 10),
    ("checkpoint_every", 3),
    ("probe_rows", 9),
    ("eval_trials", 6),
    ("head_lr", float("nan")),
])
def test_invalid_diagnostic_design_is_rejected(field, value):
    cfg = fixture_config()
    cfg[field] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_preregistered_diagnostic_requires_ten_seeds():
    cfg = fixture_config()
    cfg["scope"] = "preregistered_diagnostic"
    with pytest.raises(ValueError, match="ten"):
        validate_config(cfg)


def test_dataset_streams_are_reproducible_disjoint_and_crossed():
    cfg = fixture_config()
    first, repeated = make_splits(cfg, 31), make_splits(cfg, 31)
    assert_disjoint(list(first.values()))
    assert set(first) == {"train_phase1", "train_phase2", "iid", "ood"}
    for name, data in first.items():
        np.testing.assert_array_equal(data.counts, repeated[name].counts)
        np.testing.assert_array_equal(data.nuisance, repeated[name].nuisance)
        assert analytical_accuracy(data) == 1.0
        assert analytical_accuracy(data, larger=False) == 1.0
        assert all(analytical_accuracy(data, nuisance_index=i) == 0.5 for i in range(6))
    independent = make_dataset(31 * 100 + 22, "train_phase2", 8)
    np.testing.assert_array_equal(first["train_phase2"].nuisance, independent.nuisance)


def test_auc_uses_baseline_and_normalizes_full_trial_budget():
    assert normalized_auc([0.2, 0.6, 0.8], 4, 8) == pytest.approx(0.55)
    assert normalized_auc([0.7, 0.7, 0.7], 4, 8) == pytest.approx(0.7)
    with pytest.raises(ValueError, match="baseline"):
        normalized_auc([0.2, 0.8], 4, 8)


def test_probe_targets_are_complementary_and_evaluation_never_updates():
    graph = load_projection(ROOT / fixture_config()["graph"])
    learner = Learner(graph, 731, plastic=True)
    data = make_splits(fixture_config(), 31)["iid"]
    before = learner.snapshot(data.inputs()[:4])
    large, p_large, _ = probe_metrics(learner, data, True)
    small, p_small, _ = probe_metrics(learner, data, False)
    assert large["p_target"] + small["p_target"] == pytest.approx(1.0)
    assert large["mean_policy_sensitivity"] == pytest.approx(np.mean(p_large * (1 - p_large)))
    np.testing.assert_array_equal(p_large, p_small)
    after = learner.snapshot(data.inputs()[:4])
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    assert learner.head_updates == learner.internal_updates == 0


@pytest.mark.parametrize("target", [0, 1])
@pytest.mark.parametrize("probability", [0.01, 0.2, 0.5, 0.8, 0.99])
def test_binary_reward_factor_expectation_matches_analytic_prediction(target, probability):
    # Enumerate both actions and their deterministic correctness reward,
    # independently of Learner.update. This supports the stated algebra.
    expected = sum((probability if action else 1 - probability)
                   * (action - probability) * (int(action == target) - 0.5)
                   for action in (0, 1))
    assert expected == pytest.approx((2 * target - 1) * probability * (1 - probability))


@pytest.mark.parametrize("condition", ["fixed", "plastic"])
def test_high_first_histories_share_exactly_the_same_trained_prefix(completed_reversal_fixture, condition):
    output = completed_reversal_fixture[3]
    histories = ("high_then_low", "high_then_high", "reset_head_after_high")
    for t in (0, 4, 8):
        reference = snapshot(output, condition, histories[0], "phase1", t)
        for history in histories[1:]:
            comparison = snapshot(output, condition, history, "phase1", t)
            assert reference.keys() == comparison.keys()
            for name in reference:
                np.testing.assert_array_equal(reference[name], comparison[name])
    trials = csv_rows(output / "seed_31/trials.csv")
    for history in histories[1:]:
        reference = [{k: v for k, v in r.items() if k != "history"} for r in trials
                     if r["condition"] == condition and r["history"] == histories[0] and r["phase"] == "phase1"]
        comparison = [{k: v for k, v in r.items() if k != "history"} for r in trials
                      if r["condition"] == condition and r["history"] == history and r["phase"] == "phase1"]
        assert comparison == reference


@pytest.mark.parametrize("condition", ["fixed", "plastic"])
def test_reset_boundary_changes_only_head_and_has_separate_update_counts(completed_reversal_fixture, condition):
    output = completed_reversal_fixture[3]
    initial = snapshot(output, condition, "reset_head_after_high", "phase1", 0)
    trained = snapshot(output, condition, "reset_head_after_high", "phase1", 8)
    reset = snapshot(output, condition, "reset_head_after_high", "phase2", 0)
    np.testing.assert_array_equal(reset["readout_weights"], initial["readout_weights"])
    for name in ("internal_weights", "encoder", "bias", "pre", "post", "kc_activity", "mbon_activity"):
        np.testing.assert_array_equal(reset[name], trained[name])
    boundary = json.loads((output / "seed_31" / condition / "reset_head_after_high/boundary.json").read_text())
    assert boundary["head_reset_l2"] == pytest.approx(np.linalg.norm(trained["readout_weights"] - initial["readout_weights"]))
    assert boundary["head_reset_l2"] > 0
    assert boundary["internal_reset_l2"] == 0
    assert boundary["head_updates_before_phase2"] == 8
    assert boundary["final_head_updates"] == 16
    assert boundary["final_internal_updates"] == (16 if condition == "plastic" else 0)


def test_fixed_reset_and_fresh_are_identical_counterfactuals(completed_reversal_fixture):
    output = completed_reversal_fixture[3]
    for t in (0, 4, 8):
        fresh = snapshot(output, "fixed", "fresh_low", "phase2", t)
        reset = snapshot(output, "fixed", "reset_head_after_high", "phase2", t)
        for name in fresh:
            np.testing.assert_array_equal(fresh[name], reset[name])
    trials = csv_rows(output / "seed_31/trials.csv")
    def normalized(history):
        return [{k: v for k, v in r.items() if k != "history"} for r in trials
                if r["condition"] == "fixed" and r["history"] == history and r["phase"] == "phase2"]
    assert normalized("fresh_low") == normalized("reset_head_after_high")


def test_fresh_has_no_phase1_training_or_evaluation_rows(completed_reversal_fixture):
    output = completed_reversal_fixture[3]
    trials = csv_rows(output / "seed_31/trials.csv")
    curves = csv_rows(output / "learning_curves.csv")
    cases = csv_rows(output / "seed_31/evaluation_cases.csv")
    for rows in (trials, curves, cases):
        assert not [r for r in rows if r["history"] == "fresh_low" and r["phase"] == "phase1"]
    for condition in ("fixed", "plastic"):
        files = list((output / "seed_31" / condition / "fresh_low").glob("phase1_*.npz"))
        assert [f.name for f in files] == ["phase1_0000.npz"]
    assert len(trials) == 2 * (4 * 16 + 8)


def test_phase2_uniform_stream_is_shared_even_when_phase1_is_absent(completed_reversal_fixture):
    output = completed_reversal_fixture[3]
    with np.load(output / "seed_31/stimuli.npz") as inputs:
        uniform = inputs["phase2_uniforms"]
        np.testing.assert_array_equal(uniform, np.random.default_rng(31 + 190000).random(8))
        np.testing.assert_array_equal(inputs["phase1_uniforms"], np.random.default_rng(31 + 180000).random(8))
    for row in csv_rows(output / "seed_31/trials.csv"):
        if row["phase"] == "phase2":
            assert int(row["action"]) == int(uniform[int(row["trial"]) - 1] < float(row["p_right"]))


def test_high_continuation_is_evaluated_against_high_rule(completed_reversal_fixture):
    output = completed_reversal_fixture[3]
    for row in csv_rows(output / "learning_curves.csv"):
        if row["phase"] == "phase2":
            key = "p_larger" if row["history"] == "high_then_high" else "p_smaller"
            assert float(row["p_target"]) == float(row[key])


def test_all_seed_primary_is_not_filtered_by_acquisition_failure():
    cfg = fixture_config()
    cfg.update(seeds=[1, 2], conditions=["fixed"])
    curves, boundaries = [], []
    for seed in cfg["seeds"]:
        for history in cfg["histories"]:
            for split in ("iid", "ood"):
                if history != "fresh_low":
                    # Seed2 fails acquisition, but its history penalty remains
                    # in the primary mean and positive-seed denominator.
                    for t, value in ((0, 0.5), (4, 0.6), (8, 0.8 if seed == 1 else 0.6)):
                        curves.append(dict(seed=seed, condition="fixed", history=history,
                                           phase="phase1", trial=t, split=split, p_target=value))
                level = (0.8 if history == "fresh_low" else 0.4) if seed == 1 else (0.4 if history == "fresh_low" else 0.8)
                for t in (0, 4, 8):
                    curves.append(dict(seed=seed, condition="fixed", history=history,
                                       phase="phase2", trial=t, split=split, p_target=level))
            boundaries.append(dict(seed=seed, condition="fixed", history=history,
                                   head_reset_l2=0.0, final_head_updates=8, final_internal_updates=0))
    summary = summarize(cfg, curves, boundaries)
    verdict = summary["family_verdicts"][0]
    assert verdict["n_seeds"] == 2
    assert verdict["mean_history_auc_penalty"] == pytest.approx(0.0)
    assert verdict["positive_seed_fraction"] == 0.5
    assert verdict["qualified_seed_count"] == 1
    assert verdict["qualified_mean_auc_penalty"] == pytest.approx(0.4)
    assert verdict["history_cost_criterion_met"] is False


@pytest.mark.parametrize("changed", ["config", "protocol", "code"])
def test_registration_drift_prevents_result_creation(tmp_path, changed):
    root, config, registration = prepare_fixture(tmp_path)
    target = {"config": config, "protocol": root / "docs/reversal-protocol.md",
              "code": root / "src/drosolearn/reversal.py"}[changed]
    with target.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    output = root / "must-not-exist"
    with pytest.raises(ValueError, match="changed"):
        run(root, config, registration, output)
    assert not output.exists()


def test_existing_registration_and_result_cannot_be_overwritten(completed_reversal_fixture):
    root, config, registration, output, _ = completed_reversal_fixture
    with pytest.raises(FileExistsError):
        register(root, config, registration)
    with pytest.raises(FileExistsError):
        run(root, config, registration, output)

