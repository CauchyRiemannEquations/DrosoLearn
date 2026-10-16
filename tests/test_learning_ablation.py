"""Invented arithmetic and eight-trial learning fixtures; no scientific seeds."""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from drosolearn import learning_ablation as study
from drosolearn.data import analytical_accuracy, assert_disjoint, make_dataset
from drosolearn.graph import Projection
from drosolearn.reward_linear import LinearLearner, NormalizedLearner


REPO = Path(__file__).resolve().parents[1]


def synthetic_config(seeds=(17, 18)):
    cfg = json.loads((REPO / "configs/learning_ablation_v1.json").read_text(encoding="utf8"))
    cfg.update(scope="integration_test", seeds=list(seeds), trials_per_phase=8, checkpoint_every=4, eval_trials=8, probe_rows=2)
    cfg.update(actual_choice_feedback_trials=len(seeds) * 160, expected_head_update_calls=len(seeds) * 96,
               expected_internal_update_calls=len(seeds) * 72)
    return cfg


def toy_graph():
    return Projection(pre=np.array([0, 1, 1, 2]), post=np.array([0, 0, 1, 1]), weights=np.array([2., 1., 3., 2.]),
                      n_kc=3, n_mbon=2, kc_ids=["a", "b", "c"], mbon_ids=["d", "e"], metadata={"scope": "invented"})


def learner(network=True):
    train = make_dataset(17, "invented_train", 8).inputs()
    if network:
        return NormalizedLearner(toy_graph(), 18, train, True, .1, .01), train
    return LinearLearner(train, head_lr=.1), train


def test_registered_config_can_be_validated_without_generating_scientific_inputs(monkeypatch):
    cfg = json.loads((REPO / "configs/learning_ablation_v1.json").read_text(encoding="utf8"))
    original = deepcopy(cfg)
    monkeypatch.setattr(study, "make_dataset", lambda *a, **k: pytest.fail("Validation generated scientific inputs"))
    study.validate_config(cfg)
    study.validate_config(synthetic_config())
    assert cfg == original


@pytest.mark.parametrize("field,value", [
    ("seeds", [12001]), ("seeds", [11001]), ("seeds", [10001]), ("seeds", [True]),
    ("seeds", []), ("seeds", [17, 17]), ("seeds", [19]), ("trials_per_phase", 400),
    ("eval_trials", 128), ("checkpoint_every", 3), ("probe_rows", 9),
    ("head_lr", float("nan")), ("internal_lr", -1.), ("scope", "retuned"),
    ("models", ["real"]), ("branches", ["full"]), ("pair", [3, 4]),
    ("actual_choice_feedback_trials", 640),
])
def test_invalid_scope_masks_resources_and_scientific_development_seeds_are_rejected(field, value):
    cfg = synthetic_config()
    cfg[field] = value
    with pytest.raises(ValueError):
        study.validate_config(cfg)


def test_new_input_splits_match_declared_rng_streams_and_control_nuisance():
    cfg = synthetic_config()
    splits = study.make_splits(cfg, 17)
    assert_disjoint(list(splits.values()))
    for name, offset in (("train_phase1", 61), ("train_phase2", 64), ("iid", 62), ("ood", 63)):
        ds = splits[name]
        expected = make_dataset(1700 + offset, name, 8, domain="ood" if name == "ood" else "iid")
        np.testing.assert_array_equal(ds.counts, expected.counts)
        np.testing.assert_array_equal(ds.nuisance, expected.nuisance)
        assert ds.family_ids == expected.family_ids
        assert analytical_accuracy(ds) == analytical_accuracy(ds, larger=False) == 1.
        assert all(analytical_accuracy(ds, nuisance_index=i) == .5 for i in range(6))


@pytest.mark.parametrize("network", [False, True])
def test_full_selected_update_matches_existing_learner_formula(network):
    model, train = learner(network)
    reference = deepcopy(model)
    for i, inputs in enumerate(train[:4]):
        action, reward = i % 2, int(i in (0, 3))
        study.selected_update(model, inputs, action, reward, head=True, internal=True)
        reference.update(inputs, action, reward)
        np.testing.assert_allclose(model.head, reference.head, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(model.weights, reference.weights, rtol=1e-13, atol=1e-15)
        assert (model.head_updates, model.internal_updates) == (reference.head_updates, reference.internal_updates)


@pytest.mark.parametrize("head,internal", [(True, True), (True, False), (False, True), (False, False)])
def test_component_masks_prevent_both_displacement_and_disabled_counter_increment(head, internal):
    model, train = learner()
    model.head[:] = [.4, -.3]
    before_head, before_weights = model.head.copy(), model.weights.copy()
    study.selected_update(model, train[0], 1, 1, head=head, internal=internal)
    assert model.head_updates == int(head)
    assert model.internal_updates == int(internal)
    if not head:
        np.testing.assert_array_equal(model.head, before_head)
    if not internal:
        np.testing.assert_array_equal(model.weights, before_weights)


def test_internal_eligibility_uses_learned_pre_update_head_not_updated_or_zero_head():
    model, train = learner()
    model.head[:] = [.4, -.3]
    inputs = train[0]
    old_head, old_weights = model.head.copy(), model.weights.copy()
    kc, phi = model.features(inputs)
    p = float(model.probabilities(inputs))
    factor = (1 - p) * .5
    eligibility = old_head[model.post] * (kc[1, model.pre] - kc[0, model.pre]) / model.feature_scale
    expected_weights = np.clip(old_weights + model.internal_lr * factor * eligibility, 0, model.upper)
    expected_head = old_head + model.head_lr * factor * (phi[1] - phi[0])
    study.selected_update(model, inputs, 1, 1, head=True, internal=True)
    np.testing.assert_allclose(model.weights, expected_weights, atol=1e-15)
    np.testing.assert_allclose(model.head, expected_head, atol=1e-15)
    assert np.linalg.norm(model.weights - old_weights) > 0


def test_internal_only_keeps_nonzero_prefix_head_and_uses_it_for_learning():
    model, train = learner()
    model.head[:] = [.4, -.3]
    old_head, old_weights = model.head.copy(), model.weights.copy()
    study.selected_update(model, train[0], 1, 1, head=False, internal=True)
    np.testing.assert_array_equal(model.head, old_head)
    assert np.linalg.norm(model.weights - old_weights) > 0
    assert model.head_updates == 0 and model.internal_updates == 1


def test_zero_first_internal_displacement_still_has_one_enabled_update_call():
    model, train = learner()
    before = model.weights.copy()
    study.selected_update(model, train[0], 1, 1)
    np.testing.assert_array_equal(model.weights, before)
    assert model.head_updates == model.internal_updates == 1


def test_linear_internal_only_and_frozen_are_exact_negative_controls():
    model, train = learner(False)
    model.head[:] = np.arange(7) / 20
    frozen = deepcopy(model)
    for index, inputs in enumerate(train):
        p = float(model.probabilities(inputs))
        assert p == float(frozen.probabilities(inputs))
        action = int(.4 < p)
        study.selected_update(model, inputs, action, index % 2, head=False, internal=True)
    np.testing.assert_array_equal(model.head, frozen.head)
    assert model.head_updates == model.internal_updates == 0


def test_selected_update_interface_cannot_receive_reward_probabilities_or_targets():
    names = list(inspect.signature(study.selected_update).parameters)
    assert names == ["model", "inputs", "action", "reward", "head", "internal"]
    model, train = learner()
    with pytest.raises(TypeError):
        study.selected_update(model, train[0], 1, 1, target=1)


@pytest.mark.parametrize("change", [{"action": True}, {"reward": 1.}, {"head": 1}, {"internal": "False"}])
def test_selected_update_rejects_noncanonical_actions_rewards_and_masks(change):
    model, train = learner()
    values = {"action": 1, "reward": 1, "head": True, "internal": True}
    values.update(change)
    with pytest.raises(ValueError):
        study.selected_update(model, train[0], **values)


def test_deepcopy_preserves_prefix_scale_bounds_counters_and_independent_storage():
    model, train = learner()
    for inputs in train[:4]:
        study.selected_update(model, inputs, 1, 1)
    branch = deepcopy(model)
    for field in ("head", "weights", "initial_weights", "upper", "normalizer", "encoder", "bias", "pre", "post"):
        np.testing.assert_array_equal(getattr(branch, field), getattr(model, field))
        assert not np.shares_memory(getattr(branch, field), getattr(model, field))
    assert branch.feature_scale == model.feature_scale and branch.head_updates == branch.internal_updates == 4
    branch.head[:] += 1
    assert not np.array_equal(branch.head, model.head)


@pytest.mark.parametrize("differences,mean_pass,positive_count,passed", [
    ([.0625] * 8 + [0.] * 2, True, 8, True),
    ([.08] * 7 + [0.] * 3, True, 7, False),
    ([.049] * 10, False, 10, False),
    ([.1] * 8 + [-.2] * 2, False, 8, False),
])
def test_primary_component_criteria_keep_all_ten_manual_differences(differences, mean_pass, positive_count, passed):
    # Pure manual arithmetic: no datasets or actual learning for these labels.
    pairs = [{"seed": index, "difference": value} for index, value in enumerate(differences)]
    row = study.component_hypothesis("H_I", "full_minus_head_only", pairs, study.CRITERIA)
    assert row["n_seeds"] == 10 and row["all_seed_differences"] == pairs
    assert row["mean_criterion_met"] is mean_pass
    assert row["positive_seed_count"] == positive_count
    assert row["criterion_met"] is passed


def manual_curves(cfg):
    rows = []
    for seed in cfg["seeds"]:
        for model in study.MODELS:
            for split in ("iid", "ood"):
                prefix = [.5, .55, .6] if seed == 17 else [.5, .7, .9]
                for t, p in zip((0, 4, 8), prefix):
                    rows.append(dict(seed=seed, model=model, branch="prefix", phase="phase1", trial=t, split=split, p_rule=p))
                for branch, endpoint in (("full", .9), ("head_only", .8), ("internal_only", .7), ("frozen", 1 - prefix[-1])):
                    sequence = [1 - prefix[-1], .6, endpoint]
                    for t, p in zip((0, 4, 8), sequence):
                        rows.append(dict(seed=seed, model=model, branch=branch, phase="phase2", trial=t, split=split, p_rule=p))
    return rows


def test_failed_prefix_is_not_removed_from_component_differences_or_qualified_adaptation():
    cfg = synthetic_config()
    summary = study.summarize(cfg, manual_curves(cfg))
    assert all(row["n_seeds"] == 2 for row in summary["primary_hypotheses"].values())
    row = next(row for row in summary["outcomes"] if (row["seed"], row["model"], row["branch"], row["split"]) == (17, "real", "full", "iid"))
    assert row["prefix_acquired"] is False and row["phase2_threshold_met"] is True
    assert row["qualified_adaptation"] is False
    assert summary["prefix_failure_exclusion"] is False
    contrast = next(row for row in summary["paired_contrasts"] if
        (row["seed"], row["model"], row["split"], row["comparison"], row["metric"]) == (17, "real", "iid", "interaction", "phase2_end"))
    assert contrast["difference"] == pytest.approx(.9 - .8 - .7 + .4)


def test_raw_and_start_adjusted_auc_and_first_observed_checkpoint_are_separate():
    cfg = synthetic_config()
    summary = study.summarize(cfg, manual_curves(cfg))
    row = next(row for row in summary["outcomes"] if (row["seed"], row["model"], row["branch"], row["split"]) == (17, "linear", "full", "iid"))
    assert row["phase2_start"] == pytest.approx(.4)
    assert row["phase2_raw_auc"] == pytest.approx(.625)
    assert row["phase2_baseline_adjusted_auc"] == pytest.approx(.225)
    assert row["first_observed_threshold_checkpoint"] == 8


def test_ood_or_other_graph_results_cannot_replace_real_iid_primary_criteria():
    cfg = synthetic_config()
    rows = manual_curves(cfg)
    before = study.summarize(cfg, rows)["primary_hypotheses"]
    for row in rows:
        if row["split"] == "ood" or row["model"] != "real":
            row["p_rule"] = .01
    assert study.summarize(cfg, rows)["primary_hypotheses"] == before


def test_two_component_hypotheses_are_judged_separately():
    cfg, rows = synthetic_config(), manual_curves(synthetic_config())
    for row in rows:
        if row["branch"] == "head_only" and row["trial"] == 8:
            row["p_rule"] = .9
    verdict = study.summarize(cfg, rows)["primary_hypotheses"]
    assert verdict["H_I"]["criterion_met"] is False
    assert verdict["H_H"]["criterion_met"] is True


@pytest.mark.parametrize("mode", ["duplicate", "missing", "nonfinite"])
def test_incomplete_or_duplicate_checkpoint_coverage_is_not_silently_repaired(mode):
    cfg, rows = synthetic_config(), manual_curves(synthetic_config())
    if mode == "duplicate":
        rows.append(rows[0].copy())
    elif mode == "missing":
        rows.pop()
    else:
        rows[0]["p_rule"] = float("nan")
    with pytest.raises(ValueError):
        study.summarize(cfg, rows)
