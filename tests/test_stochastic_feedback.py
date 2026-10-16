"""Small synthetic engineering tests; registered scientific seeds never train."""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from drosolearn import stochastic_feedback as study
from drosolearn.data import analytical_accuracy, assert_disjoint, make_dataset
from drosolearn.graph import Projection
from drosolearn.reward_linear import LinearLearner, NormalizedLearner


REPO = Path(__file__).resolve().parents[1]


def synthetic_config(stage="linear", trials=8, seeds=(17, 18)):
    cfg = json.loads((REPO / f"configs/stochastic_feedback_{stage}_v1.json").read_text(encoding="utf8"))
    cfg.update(scope="integration_test", seeds=list(seeds), trials_per_phase=trials,
               checkpoint_every=4, eval_trials=8, probe_rows=2)
    exposure = len(seeds) * len(cfg["conditions"]) * len(cfg["environments"]) * 2 * trials
    cfg["actual_choice_feedback_trials"] = exposure
    cfg["expected_head_update_calls"] = exposure - len(seeds) * 3 * 2 * trials
    cfg["expected_internal_update_calls"] = len(seeds) * 3 * 3 * 2 * trials if stage == "network" else 0
    return cfg


def toy_graph():
    return Projection(pre=np.array([0, 1, 1, 2]), post=np.array([0, 0, 1, 1]),
                      weights=np.array([2., 1., 3., 2.]), n_kc=3, n_mbon=2,
                      kc_ids=["a", "b", "c"], mbon_ids=["d", "e"], metadata={"scope": "synthetic"})


def environment(identifier):
    return next(row for row in synthetic_config()["environments"] if row["id"] == identifier)


def test_both_registered_configs_validate_without_fitting_a_model():
    for stage in ("linear", "network"):
        cfg = json.loads((REPO / f"configs/stochastic_feedback_{stage}_v1.json").read_text(encoding="utf8"))
        original = deepcopy(cfg)
        study.validate_config(cfg)
        study.validate_config(synthetic_config(stage))
        assert cfg == original


@pytest.mark.parametrize("field,value", [
    ("scope", "retuned"), ("seeds", []), ("seeds", [17, 17]), ("seeds", [True]),
    ("seeds", [10001]), ("seeds", [11001]), ("trials_per_phase", 6),
    ("checkpoint_every", 3), ("eval_trials", 6), ("probe_rows", 9),
    ("head_lr", float("nan")), ("internal_lr", -1), ("pair", [3, 4]),
    ("conditions", ["linear"]),
])
def test_invalid_budget_seed_or_scope_is_rejected(field, value):
    cfg = synthetic_config()
    cfg[field] = value
    with pytest.raises(ValueError):
        study.validate_config(cfg)


@pytest.mark.parametrize("identifier,q_optimal,q_other", [
    ("deterministic", 1., 0.), ("probabilistic", .8, .2), ("uninformative", .5, .5),
])
def test_reward_probability_depends_on_chosen_action_and_current_environment(identifier, q_optimal, q_other):
    env = environment(identifier)
    for target in (0, 1):
        assert study.reward_probability(env, target, target) == q_optimal
        assert study.reward_probability(env, 1 - target, target) == q_other
    assert env == environment(identifier)


def test_new_phase_input_splits_are_disjoint_balanced_and_reproducible():
    cfg = synthetic_config()
    first, repeated = study.make_splits(cfg, 17), study.make_splits(cfg, 17)
    assert set(first) == {"train_phase1", "train_phase2", "iid", "ood"}
    assert_disjoint(list(first.values()))
    for name, dataset in first.items():
        np.testing.assert_array_equal(dataset.counts, repeated[name].counts)
        np.testing.assert_array_equal(dataset.nuisance, repeated[name].nuisance)
        assert dataset.family_ids == repeated[name].family_ids
        assert len(dataset.counts) == 8
        assert set(dataset.counts.flat) == {2, 4}
        assert analytical_accuracy(dataset, larger=True) == 1
        assert analytical_accuracy(dataset, larger=False) == 1
        assert all(analytical_accuracy(dataset, nuisance_index=i) == .5 for i in range(6))
        assert np.mean(dataset.target()) == .5
    assert set(first["train_phase1"].family_ids).isdisjoint(first["train_phase2"].family_ids)


def test_each_input_split_uses_its_declared_independent_seed_offset():
    cfg = synthetic_config()
    datasets = study.make_splits(cfg, 17)
    for name, offset in (("train_phase1", 51), ("train_phase2", 54), ("iid", 52), ("ood", 53)):
        # Match values using the production dataset's exact split label so that
        # this checks RNG allocation rather than guessing a family namespace.
        dataset = datasets[name]
        expected = make_dataset(1700 + offset, dataset.split, 8, domain="ood" if name == "ood" else "iid")
        np.testing.assert_array_equal(dataset.counts, expected.counts)
        np.testing.assert_array_equal(dataset.nuisance, expected.nuisance)


@pytest.mark.parametrize("kind", ["linear", "fixed", "plastic"])
@pytest.mark.parametrize("identifier", ["deterministic", "probabilistic", "uninformative"])
def test_evaluation_changes_no_weights_scale_counter_activation_state_or_rng(kind, identifier):
    cfg = synthetic_config()
    splits = study.make_splits(cfg, 17)
    train, evaluation = splits["train_phase1"].inputs(), splits["iid"]
    model = (LinearLearner(train) if kind == "linear" else
             NormalizedLearner(toy_graph(), 18, train, plastic=kind == "plastic"))
    model.head[:] = np.linspace(-.3, .3, len(model.head))
    before = model.snapshot(evaluation.inputs())
    counters = model.head_updates, model.internal_updates
    random_state = np.random.get_state()
    high, p_high = study.evaluate(model, evaluation, True, environment(identifier))
    low, p_low = study.evaluate(model, evaluation, False, environment(identifier))
    assert high["p_rule"] + low["p_rule"] == pytest.approx(1)
    np.testing.assert_array_equal(p_high, p_low)
    env = environment(identifier)
    for metrics in (high, low):
        assert metrics["expected_reward"] == pytest.approx(
            env["q_other"] + (env["q_optimal"] - env["q_other"]) * metrics["p_rule"])
    after = model.snapshot(evaluation.inputs())
    assert set(before) == set(after)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert counters == (model.head_updates, model.internal_updates)
    current_random_state = np.random.get_state()
    assert random_state[0] == current_random_state[0] and random_state[2:] == current_random_state[2:]
    np.testing.assert_array_equal(random_state[1], current_random_state[1])


def test_reused_learning_api_receives_only_input_action_and_actual_reward():
    for learner in (LinearLearner, NormalizedLearner):
        assert tuple(inspect.signature(learner.update).parameters) == ("self", "inputs", "action", "reward")


def test_internal_counter_and_actual_first_weight_displacement_are_different_quantities():
    splits = study.make_splits(synthetic_config(), 17)
    inputs = splits["train_phase1"].inputs()
    model = NormalizedLearner(toy_graph(), 18, inputs, plastic=True)
    before = model.weights.copy()
    scale = model.feature_scale
    model.update(inputs[0], 1, 1)
    assert model.head_updates == model.internal_updates == 1
    np.testing.assert_array_equal(model.weights, before)
    assert np.linalg.norm(model.head) > 0
    assert model.feature_scale == scale


def synthetic_curves(cfg, *, acquired=8, gaps=None):
    """Handmade probability curves, never a training run or real-data summary."""
    gaps = [.15] * len(cfg["seeds"]) if gaps is None else gaps
    rows = []
    n, step = cfg["trials_per_phase"], cfg["checkpoint_every"]
    for index, seed in enumerate(cfg["seeds"]):
        probabilistic_end = .8 if index < acquired else .6
        for condition in cfg["conditions"]:
            for env in cfg["environments"]:
                if condition == "frozen":
                    phase1_end, phase2_end = .5, .5
                elif env["id"] == "probabilistic":
                    phase1_end, phase2_end = probabilistic_end, .8
                elif env["id"] == "uninformative":
                    phase1_end, phase2_end = probabilistic_end - gaps[index], .7
                else:
                    phase1_end, phase2_end = .8, .8
                for split in ("iid", "ood"):
                    for phase, start, end in (("phase1", .5, phase1_end),
                                              ("phase2", 1 - phase1_end, phase2_end)):
                        for trial in range(0, n + 1, step):
                            rows.append({"seed": seed, "condition": condition, "environment": env["id"],
                                         "phase": phase, "trial": trial, "split": split,
                                         "p_rule": start + (end - start) * trial / n})
    return rows


@pytest.mark.parametrize("acquired,gap,expected_fraction,expected_gap,expected", [
    (8, .15, True, True, True),
    (7, .15, False, True, False),
    (8, .149, True, False, False),
])
def test_primary_h_l_uses_registered_acquisition_fraction_and_all_seed_mean_gap(
        acquired, gap, expected_fraction, expected_gap, expected):
    cfg = synthetic_config(seeds=tuple(range(17, 27)))
    summary = study.summarize(cfg, synthetic_curves(cfg, acquired=acquired, gaps=[gap] * 10))
    primary = summary["primary_hypothesis"]
    assert primary["name"] == "H_L"
    assert primary["n_seeds"] == 10
    assert primary["probabilistic_acquisition_count"] == acquired
    assert len(primary["all_seed_endpoint_gaps"]) == 10
    assert {row["seed"] for row in primary["all_seed_endpoint_gaps"]} == set(cfg["seeds"])
    assert primary["mean_paired_reward_gap"] == pytest.approx(gap)
    assert primary["fraction_criterion_met"] is expected_fraction
    assert primary["gap_criterion_met"] is expected_gap
    assert primary["criterion_met"] is expected


def test_h_l_does_not_add_an_unregistered_positive_gap_seed_fraction_or_use_ood_instead_of_iid():
    cfg = synthetic_config(seeds=tuple(range(17, 27)))
    curves = synthetic_curves(cfg, gaps=[.5] * 5 + [-.1] * 5)
    for row in curves:
        if row["split"] == "ood" and row["phase"] == "phase1" and row["trial"] == 8:
            row["p_rule"] = .1
    primary = study.summarize(cfg, curves)["primary_hypothesis"]
    assert primary["probabilistic_acquisition_count"] == 8
    assert sum(row["difference"] > 0 for row in primary["all_seed_endpoint_gaps"]) == 5
    assert primary["mean_paired_reward_gap"] == pytest.approx(.2)
    assert primary["criterion_met"] is True


def test_summary_auc_observed_checkpoint_and_prior_acquisition_qualification_are_distinct():
    cfg = synthetic_config(seeds=(17,))
    curves = synthetic_curves(cfg, acquired=1)
    for row in curves:
        if row["condition"] == "linear" and row["environment"] == "probabilistic":
            if row["phase"] == "phase1" and row["trial"] == 8:
                row["p_rule"] = .9
            if row["phase"] == "phase2":
                row["p_rule"] = {0: .1, 4: .5, 8: .8}[row["trial"]]
    summary = study.summarize(cfg, curves)
    outcome = next(row for row in summary["outcomes"] if row["condition"] == "linear"
                   and row["environment"] == "probabilistic" and row["split"] == "iid")
    assert outcome["phase2_raw_auc"] == pytest.approx(.475)
    assert outcome["phase2_baseline_adjusted_auc"] == pytest.approx(.375)
    assert outcome["phase2_start"] == .1
    assert outcome["phase2_end"] == .8
    assert outcome["first_observed_threshold_checkpoint"] == 8
    assert outcome["qualified_adaptation"] is True
    uninformative = [row for row in summary["outcomes"] if row["environment"] == "uninformative"]
    assert all(row["phase1_acquired"] is None and row["qualified_adaptation"] is None
               for row in uninformative)
    for row in curves:
        if row["condition"] == "linear" and row["environment"] == "probabilistic" and row["phase"] == "phase1":
            row["p_rule"] = .6 if row["trial"] == 8 else .5
    outcome = next(row for row in study.summarize(cfg, curves)["outcomes"]
                   if row["condition"] == "linear" and row["environment"] == "probabilistic"
                   and row["split"] == "iid")
    assert outcome["phase2_threshold_met"] is True
    assert outcome["phase1_acquired"] is False
    assert outcome["qualified_adaptation"] is False
