"""Matched reward histories and a separately labeled analytic reference.

Only tiny engineering fixtures 91 and 92 execute. Scientific seeds are checked
as declarations and are never used to fit a model before publication.
"""
from pathlib import Path
import copy
import csv
import inspect
import json
import shutil

import numpy as np
import pytest

from drosolearn import bandit_reversal as study
from drosolearn.data import analytical_accuracy, assert_disjoint, make_dataset
from drosolearn.graph import Projection
from drosolearn.reward_linear import LinearLearner, NormalizedLearner


ROOT = Path(__file__).resolve().parents[1]
NONPLASTIC = ("linear_normalized", "fixed", "rewired_fixed", "ann_fixed")


def toy_graph():
    return Projection(pre=np.array([0, 1, 1, 2]), post=np.array([0, 0, 1, 1]),
                      weights=np.array([2., 1., 3., 2.]), n_kc=3, n_mbon=2,
                      kc_ids=["a", "b", "c"], mbon_ids=["d", "e"],
                      metadata={"scope": "test fixture"})


def fixture_config():
    cfg = json.loads((ROOT / "configs/bandit_reversal_diagnostic.json").read_text(encoding="utf-8"))
    cfg.update(scope="integration_test", seeds=[91], trials_per_phase=8,
               checkpoint_every=4, eval_trials=8, probe_rows=4)
    return cfg


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def saved_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def snapshot(output, condition, history, phase, trial):
    return saved_arrays(output / "seed_91" / condition / history / f"{phase}_{trial:04d}.npz")


def prepare_fixture(path):
    root = Path(path) / "study"
    root.mkdir(parents=True)
    shutil.copytree(ROOT / "src", root / "src")
    graph = fixture_config()["graph"]
    shutil.copytree(ROOT / graph, root / graph)
    for name in ("data/source-manifest.json", "data/LICENSE.md", "pyproject.toml",
                 "requirements-lock.txt", "docs/bandit-reversal-protocol.md", "docs/bandit-reversal-background.md"):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    config = root / "config.json"
    config.write_text(json.dumps(fixture_config()) + "\n", encoding="utf-8")
    registration = root / "registration.json"
    study.register(root, config, registration)
    return root, config, registration


@pytest.fixture(scope="module")
def completed_fixture(tmp_path_factory):
    root, config, registration = prepare_fixture(tmp_path_factory.mktemp("bandit-reversal"))
    output = root / "fixture-output"
    summary = study.run(root, config, registration, output)
    return root, config, registration, output, summary


@pytest.mark.parametrize("target", [0, 1])
@pytest.mark.parametrize("probability", [0., .01, .2, .5, .8, .99, 1.])
def test_conditional_reward_gradient_is_exact_expectation_of_both_actions(target, probability):
    enumerated = sum((probability if action else 1 - probability)
                     * (action - probability) * (int(action == target) - .5)
                     for action in (0, 1))
    expected = (2 * target - 1) * probability * (1 - probability)
    assert enumerated == pytest.approx(expected, abs=1e-15)
    assert study.expected_factor(probability, target) == pytest.approx(enumerated, abs=1e-15)


@pytest.mark.parametrize("target", [0, 1])
@pytest.mark.parametrize("initial_quantity_coefficient", [-20., -.7, 0., .6, 20.])
def test_expected_action_reference_reproduces_conditional_gradient_without_sampling_an_action(target, initial_quantity_coefficient):
    train = make_dataset(91, "expected_unit_train", 8).inputs()
    model = study.ExpectedLinear(train, head_lr=.13)
    model.head[:] = [initial_quantity_coefficient, -.2, .3, -.1, .4, -.6, .5]
    previous = model.head.copy()
    difference = (train[0, 1] - train[0, 0]) / model.feature_scale
    probability = 1 / (1 + np.exp(-np.clip(difference @ previous, -40, 40)))
    enumerated_factor = sum((probability if action else 1 - probability)
                            * (action - probability) * (int(action == target) - .5)
                            for action in (0, 1))
    model.update_expected(train[0], target)
    np.testing.assert_allclose(model.head, previous + .13 * enumerated_factor * difference, rtol=0, atol=2e-15)
    assert model.feature_scale == pytest.approx(np.sqrt(np.mean(np.sum((train[:, 1] - train[:, 0]) ** 2, axis=1))))
    assert model.internal_updates == 0


def test_expected_reference_uses_the_correct_action_explicitly_and_is_not_reward_only_learning():
    assert tuple(inspect.signature(study.ExpectedLinear.update_expected).parameters) == ("self", "inputs", "target")
    assert study.ExpectedLinear is not LinearLearner


@pytest.mark.parametrize("target", [-1, 2, float("nan")])
def test_expected_reference_rejects_a_nonbinary_correct_action_without_state_change(target):
    train = make_dataset(91, "expected_invalid_train", 8).inputs()
    model = study.ExpectedLinear(train)
    before = model.head.copy()
    with pytest.raises(ValueError):
        model.update_expected(train[0], target)
    np.testing.assert_array_equal(before, model.head)
    assert model.head_updates == model.internal_updates == 0


def test_recursive_conditional_drift_is_not_the_multi_trial_stochastic_ensemble_mean():
    # Large fixture-only step size makes the Jensen/nonlinear difference
    # visible. Every stochastic branch and its probability are enumerated.
    pair = np.zeros((2, 7))
    pair[:, 0] = [.25, .5]
    train = np.stack((pair, pair))
    reference = study.ExpectedLinear(train, head_lr=2.)
    states = [(1., np.zeros(7))]
    difference = np.array([1., 0., 0., 0., 0., 0., 0.])
    for _ in range(4):
        reference.update_expected(pair, 1)
        expanded = []
        for mass, head in states:
            probability = 1 / (1 + np.exp(-head[0]))
            for action in (0, 1):
                weight = probability if action else 1 - probability
                step = 2. * (action - probability) * (int(action == 1) - .5) * difference
                expanded.append((mass * weight, head + step))
        states = expanded
    assert sum(mass for mass, _ in states) == pytest.approx(1.)
    ensemble_head = sum(mass * head for mass, head in states)
    assert abs(ensemble_head[0] - reference.head[0]) > 1e-5
    assert np.isfinite(reference.head).all()


def test_expected_high_and_low_rules_have_complementary_linear_policies_from_zero_on_identical_inputs():
    train = make_dataset(92, "complementary_train", 8).inputs()
    high, low = study.ExpectedLinear(train), study.ExpectedLinear(train)
    targets = make_dataset(92, "complementary_train", 8).target()
    for pair, target in zip(train, targets):
        high.update_expected(pair, int(target))
        low.update_expected(pair, int(1 - target))
        np.testing.assert_allclose(high.head, -low.head, rtol=0, atol=2e-15)
        np.testing.assert_allclose(high.probabilities(train) + low.probabilities(train), np.ones(8), rtol=0, atol=2e-15)


def test_auc_includes_the_initial_probability_and_a_lower_auc_does_not_imply_a_smaller_gain():
    fresh, reversed_rule = [.5, .7, .8], [.1, .5, .8]
    assert study.normalized_auc(fresh, 4, 8) == pytest.approx(.675)
    assert study.normalized_auc(reversed_rule, 4, 8) == pytest.approx(.475)
    assert study.normalized_auc(fresh, 4, 8) > study.normalized_auc(reversed_rule, 4, 8)
    assert reversed_rule[-1] - reversed_rule[0] > fresh[-1] - fresh[0]
    with pytest.raises(ValueError):
        study.normalized_auc([.1, .8], 4, 8)


@pytest.mark.parametrize("kind", ["linear", "fixed", "plastic"])
def test_evaluation_of_complementary_rules_changes_neither_state_nor_update_counts(kind):
    ds = make_dataset(92, "unit_evaluation", 8)
    train = make_dataset(91, "unit_training", 8).inputs()
    model = (LinearLearner(train) if kind == "linear" else
             NormalizedLearner(toy_graph(), 92, train, plastic=kind == "plastic"))
    model.head[:] = np.linspace(-.3, .3, len(model.head))
    before = model.snapshot(ds.inputs())
    larger, p_large, _ = study.probe_metrics(model, ds, True)
    smaller, p_small, _ = study.probe_metrics(model, ds, False)
    assert larger["p_target"] + smaller["p_target"] == pytest.approx(1.)
    assert larger["mean_policy_sensitivity"] == pytest.approx(np.mean(p_large * (1 - p_large)))
    np.testing.assert_array_equal(p_large, p_small)
    for key, original in before.items():
        np.testing.assert_array_equal(original, model.snapshot(ds.inputs())[key])
    assert model.head_updates == model.internal_updates == 0


def test_reward_learner_update_api_never_receives_the_correct_action_or_reference_state():
    for learner in (LinearLearner, NormalizedLearner):
        assert tuple(inspect.signature(learner.update).parameters) == ("self", "inputs", "action", "reward")
        assert not {"label", "target", "teacher", "reference_head"} & set(inspect.signature(learner).parameters)


@pytest.mark.parametrize("kind", ["linear", "fixed", "plastic"])
def test_zeroing_the_head_preserves_internal_state_and_initial_scale_without_reinitializing(kind):
    train = make_dataset(91, "unit_train", 8).inputs()
    model = (LinearLearner(train) if kind == "linear" else
             NormalizedLearner(toy_graph(), 92, train, plastic=kind == "plastic"))
    for i, row in enumerate(train):
        model.update(row, i % 2, int(i % 3 != 0))
    scale, counts = model.feature_scale, (model.head_updates, model.internal_updates)
    preserved = [value.copy() for value in (model.weights, model.encoder, model.bias, model.pre, model.post)]
    assert np.linalg.norm(model.head) > 0
    model.head[:] = 0.
    np.testing.assert_array_equal(model.probabilities(train), np.full(8, .5))
    for before, after in zip(preserved, (model.weights, model.encoder, model.bias, model.pre, model.post)):
        np.testing.assert_array_equal(before, after)
    assert model.feature_scale == scale
    assert (model.head_updates, model.internal_updates) == counts


@pytest.mark.parametrize("field,value", [
    ("scope", "retuned"), ("seeds", []), ("seeds", [91, 91]),
    ("seeds", [True]), ("seeds", [-1]), ("seeds", [7001]),
    ("conditions", ["linear_normalized"]), ("conditions", ["fixed", "fixed"]),
    ("histories", ["fresh_low"]), ("trials_per_phase", 10),
    ("trials_per_phase", 0), ("checkpoint_every", 3), ("eval_trials", 6),
    ("probe_rows", 9), ("head_lr", float("nan")),
    ("internal_lr", -1),
])
def test_invalid_alias_duplicate_seed_and_resource_design_are_rejected(field, value):
    cfg = fixture_config()
    cfg[field] = value
    with pytest.raises(ValueError):
        study.validate_config(cfg)


def test_declarations_include_seven_learner_families_and_four_histories_only():
    cfg = fixture_config()
    assert tuple(cfg["conditions"]) == ("linear_normalized", "fixed", "plastic",
                                       "rewired_fixed", "rewired_plastic", "ann_fixed", "ann_plastic")
    assert tuple(cfg["histories"]) == ("fresh_low", "high_to_low", "high_to_low_reset_head", "low_to_low")
    assert "linear_expected_action" not in cfg["conditions"]


def test_new_phase_data_are_exactly_crossed_disjoint_and_reproducible():
    cfg = fixture_config()
    first, repeated = study.make_splits(cfg, 91), study.make_splits(cfg, 91)
    assert set(first) == {"train_phase1", "train_phase2", "iid", "ood"}
    assert_disjoint(list(first.values()))
    for name, ds in first.items():
        np.testing.assert_array_equal(ds.counts, repeated[name].counts)
        np.testing.assert_array_equal(ds.nuisance, repeated[name].nuisance)
        assert ds.family_ids == repeated[name].family_ids
        assert analytical_accuracy(ds, larger=True) == 1.
        assert analytical_accuracy(ds, larger=False) == 1.
        assert all(analytical_accuracy(ds, nuisance_index=i) == .5 for i in range(6))
        assert np.mean(ds.target()) == .5


@pytest.mark.parametrize("condition", ["linear_normalized", "fixed", "plastic", "rewired_fixed", "rewired_plastic", "ann_fixed", "ann_plastic"])
def test_high_then_low_and_head_reset_share_identical_old_high_learning_prefix(completed_fixture, condition):
    output = completed_fixture[3]
    for t in (0, 4, 8):
        original = snapshot(output, condition, "high_to_low", "phase1", t)
        reset = snapshot(output, condition, "high_to_low_reset_head", "phase1", t)
        assert original.keys() == reset.keys()
        for key in original:
            np.testing.assert_array_equal(original[key], reset[key])
    trials = csv_rows(output / "seed_91/trials.csv")
    def prefix(history):
        return [{key: value for key, value in row.items() if key != "history"}
                for row in trials if row["condition"] == condition and row["history"] == history and row["phase"] == "phase1"]
    assert prefix("high_to_low") == prefix("high_to_low_reset_head")
    assert len(prefix("high_to_low")) == 8


@pytest.mark.parametrize("condition", ["linear_normalized", "fixed", "plastic", "rewired_fixed", "rewired_plastic", "ann_fixed", "ann_plastic"])
def test_reset_boundary_changes_only_readout_and_preserves_train_only_normalization(completed_fixture, condition):
    output = completed_fixture[3]
    trained = snapshot(output, condition, "high_to_low_reset_head", "phase1", 8)
    reset = snapshot(output, condition, "high_to_low_reset_head", "phase2", 0)
    np.testing.assert_array_equal(reset["readout_weights"], np.zeros_like(reset["readout_weights"]))
    np.testing.assert_array_equal(reset["choice_probability"], np.full(4, .5))
    for key in ("internal_weights", "encoder", "bias", "pre", "post", "kc_activity", "mbon_activity",
                "decision_activity", "feature_scale", "initial_feature_gap_rms"):
        np.testing.assert_array_equal(reset[key], trained[key])
    boundary = json.loads((output / "seed_91" / condition / "high_to_low_reset_head/boundary.json").read_text(encoding="utf-8"))
    assert boundary["reset_head"] is True
    assert boundary["head_change_l2"] == pytest.approx(np.linalg.norm(trained["readout_weights"]))
    assert boundary["head_change_l2"] > 0
    assert boundary["internal_change_l2"] == 0
    assert boundary["head_updates_before"] == boundary["head_updates_after"] == 8
    assert boundary["internal_updates_before"] == boundary["internal_updates_after"] == (8 if condition.endswith("plastic") else 0)
    final = snapshot(output, condition, "high_to_low_reset_head", "phase2", 8)
    assert int(final["head_updates"]) == 16
    assert int(final["internal_updates"]) == (16 if condition.endswith("plastic") else 0)


@pytest.mark.parametrize("condition", NONPLASTIC)
def test_fixed_internal_state_reset_equals_fresh_on_every_second_phase_checkpoint_and_update(completed_fixture, condition):
    output = completed_fixture[3]
    for t in (0, 4, 8):
        fresh = snapshot(output, condition, "fresh_low", "phase2", t)
        reset = snapshot(output, condition, "high_to_low_reset_head", "phase2", t)
        for key in fresh:
            if key in ("head_updates", "internal_updates"):
                continue
            np.testing.assert_array_equal(fresh[key], reset[key])
    trials = csv_rows(output / "seed_91/trials.csv")
    def second_phase(history):
        return [{key: value for key, value in row.items() if key != "history"}
                for row in trials if row["condition"] == condition and row["history"] == history and row["phase"] == "phase2"]
    assert second_phase("fresh_low") == second_phase("high_to_low_reset_head")


@pytest.mark.parametrize("condition", ["linear_normalized", "fixed", "plastic", "rewired_fixed", "rewired_plastic", "ann_fixed", "ann_plastic"])
def test_all_histories_share_initial_training_scale_and_do_not_refit_at_phase_change(completed_fixture, condition):
    output = completed_fixture[3]
    original = snapshot(output, condition, "high_to_low", "phase1", 0)
    for history in fixture_config()["histories"]:
        for phase, t in (("phase1", 0), ("phase2", 0), ("phase2", 8)):
            current = snapshot(output, condition, history, phase, t)
            np.testing.assert_array_equal(current["feature_scale"], original["feature_scale"])
            np.testing.assert_array_equal(current["initial_feature_gap_rms"], original["initial_feature_gap_rms"])


def test_fresh_history_has_no_old_phase_learning_or_curve_rows(completed_fixture):
    output = completed_fixture[3]
    for rows in (csv_rows(output / "seed_91/trials.csv"), csv_rows(output / "learning_curves.csv")):
        assert not [row for row in rows if row["history"] == "fresh_low" and row["phase"] == "phase1"]
    for condition in fixture_config()["conditions"]:
        files = list((output / "seed_91" / condition / "fresh_low").glob("phase1_*.npz"))
        assert [path.name for path in files] == ["phase1_0000.npz"]
        boundary = json.loads((output / "seed_91" / condition / "fresh_low/boundary.json").read_text(encoding="utf-8"))
        assert boundary["head_updates_before"] == boundary["head_updates_after"] == 0
        assert boundary["internal_updates_before"] == boundary["internal_updates_after"] == 0
        final = snapshot(output, condition, "fresh_low", "phase2", 8)
        assert int(final["head_updates"]) == 8
        assert int(final["internal_updates"]) == (8 if condition.endswith("plastic") else 0)


def test_actual_update_budget_excludes_the_separately_labeled_deterministic_reference(completed_fixture):
    output, summary = completed_fixture[3:]
    trials = csv_rows(output / "seed_91/trials.csv")
    assert len(trials) == 7 * (8 + 3 * 16)
    assert {row["condition"] for row in trials} == set(fixture_config()["conditions"])
    assert all(row["phase"] in ("phase1", "phase2") for row in trials)
    for condition in fixture_config()["conditions"]:
        for history in fixture_config()["histories"]:
            selected = [row for row in trials if row["condition"] == condition and row["history"] == history]
            assert len(selected) == (8 if history == "fresh_low" else 16)
    assert summary["actual_training_trials"] == len(trials)
    expected_rows = csv_rows(output / "seed_91/expected_updates.csv")
    assert len(expected_rows) == summary["expected_reference_updates"] == 7 * 8
    assert all("action" not in row and "reward" not in row for row in expected_rows)


def test_phase_two_shared_uniforms_match_every_actual_record_even_when_phase_one_is_skipped(completed_fixture):
    output = completed_fixture[3]
    stimuli = saved_arrays(output / "seed_91/stimuli.npz")
    for phase, offset in (("phase1", 280000), ("phase2", 300000)):
        np.testing.assert_array_equal(stimuli[f"action_{phase}_uniforms"], np.random.default_rng(91 + offset).random(8))
    for row in csv_rows(output / "seed_91/trials.csv"):
        probability = float(row["p_right"])
        uniform = stimuli[f"action_{row['phase']}_uniforms"][int(row["trial"]) - 1]
        assert int(row["action"]) == int(uniform < probability)
        counts = [int(row["left_count"]), int(row["right_count"])]
        larger = row["phase"] == "phase1" and row["history"] in ("high_to_low", "high_to_low_reset_head")
        target = int(np.argmax(counts) if larger else np.argmin(counts))
        assert int(row["reward"]) == int(int(row["action"]) == target)


def synthetic_summary_fixture():
    cfg = fixture_config()
    cfg["seeds"] = [91, 92]
    curves = []
    for seed in cfg["seeds"]:
        for condition in (*cfg["conditions"], "expected_linear"):
            for history in cfg["histories"]:
                if history != "fresh_low":
                    prefix = [.5, .75, .9] if seed == 91 or history == "low_to_low" else [.5, .53, .55]
                    for split in ("iid", "ood"):
                        for trial, value in zip((0, 4, 8), prefix):
                            curves.append(dict(seed=seed, condition=condition, history=history,
                                               phase="phase1", trial=trial, split=split,
                                               p_target=value, greedy_accuracy=1.))
                levels = {"fresh_low": [.5, .7, .9] if seed == 91 else [.5, .65, .8],
                          "high_to_low": [.1, .2, .3] if seed == 91 else [.45, .65, .85],
                          "high_to_low_reset_head": [.5, .7, .9],
                          "low_to_low": [.9, .93, .95]}[history]
                for split in ("iid", "ood"):
                    for trial, value in zip((0, 4, 8), levels):
                        curves.append(dict(seed=seed, condition=condition, history=history,
                                           phase="phase2", trial=trial, split=split,
                                           p_target=value, greedy_accuracy=1.))
    return cfg, curves


def test_acquisition_failures_remain_in_primary_endpoint_cost_and_positive_denominators():
    cfg, curves = synthetic_summary_fixture()
    summary = study.summarize(cfg, curves)
    verdict = summary["primary_hypothesis"]
    assert verdict["n_seeds"] == 2
    assert verdict["mean_endpoint_cost"] == pytest.approx((.6 - .05) / 2)
    assert verdict["positive_seed_count"] == 1
    assert verdict["high_prefix_acquisition_count"] == 1
    assert verdict["fresh_acquisition_count"] == 2
    assert verdict["criterion_met"] is False
    reverse = next(row for row in summary["outcomes"] if row["seed"] == 92 and row["condition"] == "linear_normalized" and row["history"] == "high_to_low")
    assert reverse["phase2_target_passed"] is True
    assert reverse["phase1_acquired"] is False
    assert reverse["qualified_adaptation"] is False


def test_repeated_low_rule_can_qualify_without_an_additional_fifteen_point_gain():
    cfg, curves = synthetic_summary_fixture()
    summary = study.summarize(cfg, curves)
    repeated = [row for row in summary["outcomes"] if row["history"] == "low_to_low" and row["condition"] != "expected_linear"]
    assert repeated
    assert all(row["phase2_gain"] == pytest.approx(.05) for row in repeated)
    assert all(row["phase1_acquired"] and row["qualified_adaptation"] for row in repeated)
    assert all(row["first_observed_75_checkpoint"] == 0 for row in repeated)
    references = [row for row in summary["outcomes"] if row["condition"] == "expected_linear"]
    assert len(references) == len(cfg["seeds"]) * len(cfg["histories"])
    assert all(row["algorithm"] == "conditional_expected_action_reference" for row in references)
    assert all(row[key] is None for row in references for key in
               ("phase1_acquired", "fresh_acquired", "phase2_target_passed", "qualified_adaptation", "first_observed_75_checkpoint"))


def test_curve_auc_and_baseline_adjusted_auc_have_distinct_values_in_summary():
    cfg, curves = synthetic_summary_fixture()
    summary = study.summarize(cfg, curves)
    reverse = next(row for row in summary["outcomes"] if row["seed"] == 91 and row["condition"] == "linear_normalized" and row["history"] == "high_to_low")
    assert reverse["phase2_auc"] == pytest.approx(.2)
    assert reverse["phase2_baseline_adjusted_auc"] == pytest.approx(.1)
    assert reverse["phase2_gain"] == pytest.approx(.2)
    assert reverse["first_observed_75_checkpoint"] is None


@pytest.mark.parametrize("changed", ["config", "protocol", "background", "code"])
def test_registered_config_or_source_drift_stops_before_any_new_result_directory(tmp_path, changed):
    root, config, registration = prepare_fixture(tmp_path)
    target = {"config": config, "protocol": root / "docs/bandit-reversal-protocol.md",
              "background": root / "docs/bandit-reversal-background.md",
              "code": root / "src/drosolearn/bandit_reversal.py"}[changed]
    with target.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    output = root / "must-not-exist"
    with pytest.raises(ValueError):
        study.run(root, config, registration, output)
    assert not output.exists()


def test_registration_and_existing_output_are_exclusive(tmp_path):
    root, config, registration = prepare_fixture(tmp_path)
    before = registration.read_bytes()
    with pytest.raises(FileExistsError):
        study.register(root, config, registration)
    assert registration.read_bytes() == before
    output = root / "existing-output"
    output.mkdir()
    sentinel = output / "old_failure.txt"
    sentinel.write_text("preserve failure", encoding="utf-8")
    with pytest.raises(FileExistsError):
        study.run(root, config, registration, output)
    assert sentinel.read_text(encoding="utf-8") == "preserve failure"
