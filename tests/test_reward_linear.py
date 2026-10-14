"""Independent reward gradients, train-only scale and prospective study guards.

Only engineering fixtures 81 and 82 are executed. The scientific seeds are
validated as declarations, never used to fit a model before publication.
"""
from pathlib import Path
import copy
import csv
import inspect
import json
import shutil

import numpy as np
import pytest

from drosolearn import reward_linear as study
from drosolearn.data import analytical_accuracy, assert_disjoint, make_dataset
from drosolearn.graph import Projection


ROOT = Path(__file__).resolve().parents[1]


def toy_graph():
    return Projection(
        pre=np.array([0, 1, 1, 2]), post=np.array([0, 0, 1, 1]),
        weights=np.array([2., 1., 3., 2.]), n_kc=3, n_mbon=2,
        kc_ids=["a", "b", "c"], mbon_ids=["d", "e"],
        metadata={"scope": "test fixture"},
    )


def train_inputs(seed=81, n=8):
    return make_dataset(seed, "train", n).inputs()


def independent_network_activity(model, inputs):
    kc = np.maximum(inputs @ model.encoder.T + model.bias, 0)
    raw = np.zeros((*inputs.shape[:-1], len(model.head)))
    for i, (pre, post) in enumerate(zip(model.graph.pre, model.graph.post)):
        raw[..., post] += kc[..., pre] * model.weights[i]
    return kc, raw


def independent_scale(features):
    difference = features[:, 1] - features[:, 0]
    return float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1))))


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("normalized", [False, True])
def test_linear_starts_without_quantity_prior_and_uses_train_vector_norm(normalized):
    x = train_inputs()
    model = study.LinearLearner(x, normalize=normalized)
    expected = independent_scale(x)
    assert model.initial_feature_gap_rms == pytest.approx(expected)
    assert model.feature_scale == pytest.approx(expected if normalized else 1.)
    # Reducing over all scalar entries would inject a feature-dimension factor.
    assert expected != pytest.approx(np.sqrt(np.mean((x[:, 1] - x[:, 0]) ** 2)))
    np.testing.assert_array_equal(model.head, np.zeros(7))
    np.testing.assert_array_equal(model.probabilities(x), np.full(len(x), .5))
    kc, decision = model.features(x)
    assert kc.shape[-1] == 0
    np.testing.assert_allclose(decision, x / model.feature_scale, rtol=0, atol=0)
    assert model.head_updates == model.internal_updates == 0


@pytest.mark.parametrize("plastic", [False, True])
def test_network_starts_without_policy_prior_and_scale_is_initial_train_only(plastic):
    x = train_inputs()
    model = study.NormalizedLearner(toy_graph(), 82, x, plastic=plastic)
    kc, raw = independent_network_activity(model, x)
    expected = independent_scale(raw)
    assert model.initial_feature_gap_rms == pytest.approx(expected)
    assert model.feature_scale == pytest.approx(expected)
    np.testing.assert_array_equal(model.head, np.zeros(2))
    np.testing.assert_array_equal(model.probabilities(x), np.full(len(x), .5))
    actual_kc, decision = model.features(x)
    np.testing.assert_array_equal(actual_kc, kc)
    np.testing.assert_allclose(decision, raw / expected, rtol=0, atol=1e-15)
    # Evaluation has different nuisance and quantities, and cannot refit g.
    evaluation = make_dataset(82, "evaluation", 8, pair=(5, 7), domain="ood").inputs()
    model.probabilities(evaluation)
    model.snapshot(evaluation)
    assert model.feature_scale == expected
    assert model.head_updates == model.internal_updates == 0


@pytest.mark.parametrize("action,reward", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("normalized", [False, True])
def test_linear_one_binary_feedback_update_matches_independent_policy_gradient(action, reward, normalized):
    x = train_inputs()
    model = study.LinearLearner(x, normalize=normalized, head_lr=.13)
    model.head[:] = np.array([.7, -.2, .3, .1, -.4, .6, -.5])
    old = model.head.copy()
    difference = (x[0, 1] - x[0, 0]) / model.feature_scale
    probability = 1 / (1 + np.exp(-np.clip(difference @ old, -40, 40)))
    factor = (action - probability) * (reward - .5)
    returned = model.update(x[0], action, reward)
    assert returned == pytest.approx(probability)
    np.testing.assert_allclose(model.head, old + .13 * factor * difference, rtol=0, atol=1e-15)
    assert model.head_updates == 1 and model.internal_updates == 0


@pytest.mark.parametrize("action,reward", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("plastic", [False, True])
def test_network_gradient_uses_old_head_and_fixed_scale_in_both_learning_sites(action, reward, plastic):
    x = train_inputs()
    model = study.NormalizedLearner(toy_graph(), 82, x, plastic=plastic,
                                    head_lr=.13, internal_lr=.017)
    model.head[:] = [.7, -.4]
    old_head, old_weights = model.head.copy(), model.weights.copy()
    kc, raw = independent_network_activity(model, x[0])
    delta = raw[1] - raw[0]
    probability = 1 / (1 + np.exp(-np.clip(delta @ old_head / model.feature_scale, -40, 40)))
    factor = (action - probability) * (reward - .5)
    expected_head = old_head + .13 * factor * delta / model.feature_scale
    eligibility = old_head[model.graph.post] / model.feature_scale * (
        kc[1, model.graph.pre] - kc[0, model.graph.pre])
    expected_weights = np.clip(old_weights + .017 * factor * eligibility, 0, model.upper)
    returned = model.update(x[0], action, reward)
    assert returned == pytest.approx(probability)
    np.testing.assert_allclose(model.head, expected_head, rtol=0, atol=1e-15)
    np.testing.assert_allclose(model.weights, expected_weights if plastic else old_weights,
                               rtol=0, atol=1e-15)
    assert model.head_updates == 1 and model.internal_updates == int(plastic)


def test_zero_head_prevents_first_internal_step_and_later_internal_step_uses_learned_head():
    x = train_inputs()
    model = study.NormalizedLearner(toy_graph(), 82, x, plastic=True)
    initial = model.weights.copy()
    model.update(x[0], 1, 1)
    np.testing.assert_array_equal(model.weights, initial)
    assert np.linalg.norm(model.head) > 0
    old_head, old_weights, scale = model.head.copy(), model.weights.copy(), model.feature_scale
    kc, raw = independent_network_activity(model, x[0])
    probability = 1 / (1 + np.exp(-np.clip((raw[1] - raw[0]) @ old_head / scale, -40, 40)))
    expected = np.clip(old_weights + .01 * (1 - probability) * .5 * old_head[model.graph.post] / scale
                       * (kc[1, model.graph.pre] - kc[0, model.graph.pre]), 0, model.upper)
    model.update(x[0], 1, 1)
    np.testing.assert_allclose(model.weights, expected, rtol=0, atol=1e-15)
    assert not np.array_equal(model.weights, initial)
    assert model.feature_scale == scale


def test_normalized_linear_is_invariant_to_global_input_rescaling_without_retuning():
    x = train_inputs()
    original = study.LinearLearner(x, normalize=True)
    scaled = study.LinearLearner(3 * x, normalize=True)
    assert scaled.feature_scale == pytest.approx(3 * original.feature_scale)
    for row, action, reward in zip(x, [1, 0, 1, 0, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0]):
        original.update(row, action, reward)
        scaled.update(3 * row, action, reward)
    np.testing.assert_allclose(original.head, scaled.head, rtol=0, atol=1e-15)
    np.testing.assert_allclose(original.probabilities(x), scaled.probabilities(3 * x), rtol=0, atol=1e-15)


def test_scalar_score_scale_remains_fixed_after_large_plastic_changes():
    x = train_inputs()
    model = study.NormalizedLearner(toy_graph(), 82, x, plastic=True, internal_lr=100)
    model.head[:] = [.7, -.4]
    scale = model.feature_scale
    encoder, bias, pre, post = [a.copy() for a in (model.encoder, model.bias, model.graph.pre, model.graph.post)]
    for row in x:
        model.update(row, 1, 1)
    assert model.feature_scale == scale
    assert np.all(model.weights >= 0) and np.all(model.weights <= model.upper)
    for before, after in zip((encoder, bias, pre, post),
                              (model.encoder, model.bias, model.graph.pre, model.graph.post)):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("kind", ["linear", "network"])
@pytest.mark.parametrize("action,reward", [(-1, 0), (2, 1), (0, -1), (1, 2)])
def test_optimizer_rejects_nonbinary_feedback(kind, action, reward):
    x = train_inputs()
    model = study.LinearLearner(x) if kind == "linear" else study.NormalizedLearner(toy_graph(), 82, x)
    head, weights = model.head.copy(), model.weights.copy()
    with pytest.raises(ValueError):
        model.update(x[0], action, reward)
    np.testing.assert_array_equal(model.head, head)
    np.testing.assert_array_equal(model.weights, weights)
    assert model.head_updates == model.internal_updates == 0


@pytest.mark.parametrize("kind", ["linear", "network"])
def test_snapshot_labels_decision_activity_separately_from_biological_names_and_changes_no_state(kind):
    x = train_inputs()
    model = study.LinearLearner(x) if kind == "linear" else study.NormalizedLearner(toy_graph(), 82, x)
    model.head[:] = np.linspace(-.3, .3, len(model.head))
    before = model.snapshot(x)
    required = {"internal_weights", "readout_weights", "kc_activity", "mbon_activity",
                "choice_probability", "decision_activity", "feature_scale", "initial_feature_gap_rms"}
    assert required <= set(before)
    if kind == "linear":
        assert before["kc_activity"].shape[-1] == before["mbon_activity"].shape[-1] == 0
        np.testing.assert_allclose(before["decision_activity"], x / model.feature_scale, rtol=0, atol=0)
    else:
        kc, raw = independent_network_activity(model, x)
        np.testing.assert_array_equal(before["kc_activity"], kc)
        np.testing.assert_allclose(before["mbon_activity"], raw, rtol=0, atol=1e-15)
        np.testing.assert_allclose(before["decision_activity"], raw / model.feature_scale, rtol=0, atol=1e-15)
    model.probabilities(x)
    after = model.snapshot(x)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert model.head_updates == model.internal_updates == 0


@pytest.mark.parametrize("kind", ["linear", "network"])
def test_equal_candidate_inputs_tie_and_saturation_is_clipped(kind):
    x = train_inputs()
    model = study.LinearLearner(x) if kind == "linear" else study.NormalizedLearner(toy_graph(), 82, x)
    model.head[:] = 1e9
    tied = np.stack((x[:, 0], x[:, 0]), axis=1)
    np.testing.assert_array_equal(model.probabilities(tied), np.full(len(x), .5))
    probability = model.probabilities(x)
    _, decision = model.features(x)
    gap = np.clip((decision[:, 1] - decision[:, 0]) @ model.head, -40, 40)
    np.testing.assert_allclose(probability, 1 / (1 + np.exp(-gap)), rtol=0, atol=0)


def test_constructor_and_update_have_no_label_teacher_or_target_arguments():
    for cls in (study.LinearLearner, study.NormalizedLearner):
        assert not {"label", "labels", "target", "targets", "teacher", "teacher_scores"} & set(inspect.signature(cls).parameters)
        assert tuple(inspect.signature(cls.update).parameters) == ("self", "inputs", "action", "reward")


@pytest.mark.parametrize("features", [np.zeros((8, 2, 7)), np.full((8, 2, 7), np.nan),
                                     np.full((8, 2, 7), np.inf), np.zeros((8, 7))])
def test_initial_scale_rejects_zero_nonfinite_or_unpaired_features(features):
    with pytest.raises(ValueError):
        study.gap_rms(features)


def reward_fixture_config():
    cfg = json.loads((ROOT / "configs/reward_linear_diagnostic.json").read_text(encoding="utf-8"))
    cfg.update(scope="integration_test", seeds=[81], trials=8, checkpoint_every=4,
               eval_trials=8, probe_rows=4)
    return cfg


def prepare_reward_fixture(path):
    root = Path(path) / "study"
    root.mkdir(parents=True)
    shutil.copytree(ROOT / "src", root / "src")
    graph = reward_fixture_config()["graph"]
    shutil.copytree(ROOT / graph, root / graph)
    for name in ("data/source-manifest.json", "data/LICENSE.md", "pyproject.toml",
                 "requirements-lock.txt", "docs/reward-linear-protocol.md", "docs/reward-linear-background.md"):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    config = root / "config.json"
    config.write_text(json.dumps(reward_fixture_config()) + "\n", encoding="utf-8")
    registration = root / "registration.json"
    study.register(root, config, registration)
    return root, config, registration


@pytest.fixture(scope="module")
def completed_reward_fixture(tmp_path_factory):
    root, config, registration = prepare_reward_fixture(tmp_path_factory.mktemp("reward"))
    output = root / "fixture-output"
    summary = study.run(root, config, registration, output)
    return root, config, registration, output, summary


def test_all_nine_conditions_and_new_splits_have_no_calibration_or_teacher_inputs():
    assert study.CONDITIONS_ALL == ("linear_raw", "linear_normalized", "linear_random_reward",
                                    "fixed", "plastic", "rewired_fixed", "rewired_plastic", "ann_fixed", "ann_plastic")
    cfg = reward_fixture_config()
    first, repeated = study.make_splits(cfg, 81), study.make_splits(cfg, 81)
    assert len(first) == 30 and "calibration" not in first
    assert_disjoint(list(first.values()))
    for name, ds in first.items():
        np.testing.assert_array_equal(ds.counts, repeated[name].counts)
        np.testing.assert_array_equal(ds.nuisance, repeated[name].nuisance)
        assert np.mean(ds.target()) == .5
        assert analytical_accuracy(ds) == 1.
        assert all(analytical_accuracy(ds, nuisance_index=i) == .5 for i in range(6))
    for name in ("train", "iid", "ood"):
        for codes in study.spacing_panel():
            np.testing.assert_array_equal(study.encoded_inputs(first[name], codes), first[name].inputs())


@pytest.mark.parametrize("kind", ["linear_raw", "linear_normalized", "network_fixed", "network_plastic"])
def test_frozen_transfer_cache_matches_direct_full_forward_for_each_case_and_code(kind):
    cfg = reward_fixture_config()
    datasets, panel = study.make_splits(cfg, 82), study.spacing_panel()
    train = datasets["train"].inputs()
    model = (study.LinearLearner(train, normalize=kind != "linear_raw") if kind.startswith("linear")
             else study.NormalizedLearner(toy_graph(), 82, train, plastic=kind == "network_plastic"))
    for i, inputs in enumerate(train):
        model.update(inputs, i % 2, int(i % 3 == 0))
    before = model.snapshot(datasets["iid"].inputs())
    result = study.transfer_evaluate(model, datasets, panel)
    assert result["score_cache"].shape == (27, 8, 2, 14)
    assert result["map_p_right"].shape == (4, 27, 8)
    for j, (a, b) in enumerate(study.PAIRS):
        ds = datasets[f"pair_{a}_{b}"]
        for k, value in enumerate(result["code_values"]):
            inputs = study.encoded_inputs(ds, np.full(8, value))
            if kind.startswith("linear"):
                direct_scores = inputs @ model.head / model.feature_scale
            else:
                _, raw = independent_network_activity(model, inputs)
                direct_scores = raw @ model.head / model.feature_scale
            np.testing.assert_allclose(result["score_cache"][j, ..., k], direct_scores, rtol=0, atol=2e-14)
        for map_id, codes in enumerate(panel):
            probability = model.probabilities(study.encoded_inputs(ds, codes))
            target = ds.target()
            np.testing.assert_allclose(result["map_p_right"][map_id, j], probability, rtol=0, atol=2e-14)
            expected = np.where(target == 1, probability, 1 - probability).mean()
            greedy = np.where(np.abs(probability - .5) < 1e-12, .5,
                              ((probability > .5).astype(int) == target).astype(float)).mean()
            assert result["map_p_target"][map_id, j] == pytest.approx(expected, abs=2e-14)
            assert result["map_greedy_accuracy"][map_id, j] == pytest.approx(greedy, abs=2e-14)
    for key, original in before.items():
        np.testing.assert_array_equal(original, model.snapshot(datasets["iid"].inputs())[key])
    assert model.head_updates == 8 and model.internal_updates == (8 if kind == "network_plastic" else 0)


@pytest.mark.parametrize("changed", ["config", "protocol", "background", "code"])
def test_registration_blocks_drift_before_creating_any_new_output(tmp_path, changed):
    root, config, registration = prepare_reward_fixture(tmp_path)
    target = {"config": config, "protocol": root / "docs/reward-linear-protocol.md",
              "background": root / "docs/reward-linear-background.md",
              "code": root / "src/drosolearn/reward_linear.py"}[changed]
    with target.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    output = root / "must-not-exist"
    with pytest.raises(ValueError, match="changed after registration"):
        study.run(root, config, registration, output)
    assert not output.exists()


def test_registration_and_previous_results_are_exclusive_and_preserved(tmp_path):
    root, config, registration = prepare_reward_fixture(tmp_path)
    original = registration.read_bytes()
    with pytest.raises(FileExistsError):
        study.register(root, config, registration)
    assert registration.read_bytes() == original
    output = root / "existing"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("old result", encoding="utf-8")
    with pytest.raises(FileExistsError):
        study.run(root, config, registration, output)
    assert marker.read_text(encoding="utf-8") == "old result"


@pytest.mark.parametrize("field,value", [
    ("scope", "smoke"), ("seeds", []), ("seeds", [81, 81]), ("seeds", [True]),
    ("train_pair", [1, 3]), ("primary_pairs", "all27"), ("conditions", ["linear_normalized"]),
    ("code_alphas", [.5, .25, 1., 1.5]), ("trials", 6), ("trials", 8.),
    ("checkpoint_every", 3), ("eval_trials", 6), ("probe_rows", 9),
    ("head_lr", float("nan")), ("internal_lr", -1),
])
def test_invalid_reward_design_rejected(field, value):
    cfg = reward_fixture_config()
    cfg[field] = value
    with pytest.raises(ValueError):
        study.validate_config(cfg)


def test_scientific_configuration_is_exact_without_executing_any_scientific_seed():
    cfg = json.loads((ROOT / "configs/reward_linear_diagnostic.json").read_text(encoding="utf-8"))
    study.validate_config(cfg)
    assert cfg["seeds"] == list(range(6001, 6011))
    for field, value in (("seeds", list(range(6002, 6012))), ("trials", 200),
                         ("checkpoint_every", 50), ("eval_trials", 64), ("probe_rows", 4),
                         ("head_lr", .2), ("internal_lr", .02)):
        changed = copy.deepcopy(cfg)
        changed[field] = value
        with pytest.raises(ValueError):
            study.validate_config(changed)
    for field, value in (("acquisition_accuracy", .8), ("acquisition_gain", .2),
                         ("transfer_accuracy", .8), ("reward_gap", .1), ("seed_pass_fraction", .7)):
        changed = copy.deepcopy(cfg)
        changed["criteria"][field] = value
        with pytest.raises(ValueError):
            study.validate_config(changed)
    engineering = reward_fixture_config()
    engineering["scope"] = "preregistered_diagnostic"
    with pytest.raises(ValueError):
        study.validate_config(engineering)


@pytest.mark.parametrize("field", ["experiment_id", "graph", "head_initialization",
                                    "feature_normalization", "random_reward"])
def test_study_identity_normalization_and_reward_declarations_cannot_drift(field):
    cfg = reward_fixture_config()
    cfg[field] = "different declaration"
    with pytest.raises(ValueError, match="fixed"):
        study.validate_config(cfg)


@pytest.mark.parametrize("seeds", [[6001], [81, 6010]])
def test_development_scope_cannot_alias_any_registered_scientific_seed(seeds):
    cfg = reward_fixture_config()
    cfg["seeds"] = seeds
    with pytest.raises(ValueError, match="Scientific seeds"):
        study.validate_config(cfg)


@pytest.mark.parametrize("field,value", [("acquisition_accuracy", float("nan")),
                                           ("acquisition_gain", -1), ("transfer_accuracy", 1.1),
                                           ("reward_gap", .0), ("seed_pass_fraction", 0)])
def test_invalid_or_omitted_criteria_rejected(field, value):
    cfg = reward_fixture_config()
    if field == "reward_gap":
        del cfg["criteria"][field]
    else:
        cfg["criteria"][field] = value
    with pytest.raises(ValueError):
        study.validate_config(cfg)


def synthetic_summary_inputs(cfg, cases):
    curves, transfers = [], {}
    for seed, case in zip(cfg["seeds"], cases):
        for condition in cfg["conditions"]:
            before = case.get("before", .5)
            after = (case["random"] if condition == "linear_random_reward" else case["after"])
            for split in ("iid", "ood"):
                for trial, probability in ((0, before), (cfg["trials"], after)):
                    curves.append(dict(seed=seed, condition=condition, trial=trial, split=split, p_target=probability))
            baseline = np.full((4, 27), .5)
            final = np.full((4, 27), .99)
            final[:, study.PRIMARY] = np.asarray(case.get("unseen", [.8] * 4))[:, None]
            transfers[seed, condition, "baseline"] = {"map_p_target": baseline, "map_greedy_accuracy": baseline}
            transfers[seed, condition, "final"] = {"map_p_target": final, "map_greedy_accuracy": np.ones((4, 27))}
    return curves, transfers


def synthetic_summary(cases):
    cfg = reward_fixture_config()
    cfg["seeds"] = [81, 82]
    return study.summarize(cfg, *synthetic_summary_inputs(cfg, cases))


def test_reward_and_transfer_hypotheses_require_acquisition_and_include_failed_seeds():
    result = synthetic_summary([dict(after=.8, random=.5), dict(after=.6, random=.5, unseen=[.99] * 4)])
    selected = [r for r in result["outcomes"] if r["condition"] == "linear_normalized"]
    assert selected[0]["acquired"] is True and selected[1]["acquired"] is False
    assert selected[1]["qualified_unseen_transfer"] == [False] * 4
    verdict = next(v for v in result["family_verdicts"] if v["condition"] == "linear_normalized")
    assert verdict["n_seeds"] == 2 and verdict["acquisition_count"] == 1
    assert verdict["joint_identity_transfer_count"] == 1
    assert verdict["acquisition_criterion_met"] is False
    assert result["hypotheses"]["H_R"]["mean_iid_reward_gap"] == pytest.approx(.2)
    assert result["hypotheses"]["H_R"]["criterion_met"] is False
    assert result["hypotheses"]["H_G"]["criterion_met"] is False
    assert result["actual_training_trials"] == 2 * 9 * 8


def test_primary_transfer_is_identity_unseen_mean_not_anchored_cases_or_focal_count():
    result = synthetic_summary([dict(after=.8, random=.5, unseen=[.9, .9, .74, .9])] * 2)
    assert result["hypotheses"]["H_R"]["criterion_met"] is True
    assert result["hypotheses"]["H_G"]["criterion_met"] is False
    for row in result["outcomes"]:
        assert row["anchored_after"] == pytest.approx([.99] * 4)
        if row["condition"] != "linear_random_reward":
            assert row["qualified_unseen_transfer"] == [True, True, False, True]


def test_single_focal_pair_pass_cannot_replace_unseen_fifteen_pair_gate():
    cfg = reward_fixture_config()
    cfg["seeds"] = [81, 82]
    curves, transfers = synthetic_summary_inputs(cfg, [dict(after=.8, random=.5, unseen=[.7] * 4)] * 2)
    for seed in cfg["seeds"]:
        for condition in cfg["conditions"]:
            transfers[seed, condition, "final"]["map_p_target"][:, study.FOCAL_INDEX] = .99
    result = study.summarize(cfg, curves, transfers)
    assert result["hypotheses"]["H_G"]["criterion_met"] is False
    primary = next(r for r in result["outcomes"] if r["condition"] == "linear_normalized")
    assert primary["qualified_focal_transfer"] == [True] * 4
    assert primary["qualified_unseen_transfer"] == [False] * 4
    assert primary["unseen_after"] == pytest.approx([(.7 * 14 + .99) / 15] * 4)


def test_reward_effect_gate_uses_all_paired_seeds_and_strict_positive_fraction():
    result = synthetic_summary([dict(after=.8, random=.8), dict(after=.9, random=.5)])
    reward = result["hypotheses"]["H_R"]
    assert reward["mean_iid_reward_gap"] == pytest.approx(.2)
    assert reward["positive_seed_count"] == 1
    assert reward["acquisition_count"] == 2
    assert reward["criterion_met"] is False
    assert result["hypotheses"]["H_G"]["criterion_met"] is True


def test_probability_threshold_gain_and_reward_difference_are_separate_gates():
    high_prior = synthetic_summary([dict(before=.72, after=.8, random=.5)] * 2)
    assert high_prior["hypotheses"]["H_R"]["acquisition_count"] == 0
    close_reward = synthetic_summary([dict(after=.8, random=.7)] * 2)
    assert close_reward["hypotheses"]["H_R"]["acquisition_count"] == 2
    assert close_reward["hypotheses"]["H_R"]["positive_seed_count"] == 2
    assert close_reward["hypotheses"]["H_R"]["criterion_met"] is False
    assert close_reward["hypotheses"]["H_G"]["criterion_met"] is True


def test_small_run_preserves_shared_action_randomness_and_action_independent_reward(completed_reward_fixture):
    _, config, _, output, summary = completed_reward_fixture
    cfg = json.loads(config.read_text(encoding="utf-8"))
    rows = csv_rows(output / "seed_81/trials.csv")
    assert len(rows) == summary["actual_training_trials"] == 72
    with np.load(output / "seed_81/stimuli.npz") as stimuli:
        action_uniforms, reward_uniforms = stimuli["action_uniforms"], stimuli["reward_uniforms"]
        np.testing.assert_array_equal(action_uniforms, np.random.default_rng(81 + 280000).random(8))
        np.testing.assert_array_equal(reward_uniforms, np.random.default_rng(81 + 290000).random(8))
        assert not any("calibration" in key or "teacher" in key for key in stimuli.files)
        counts = stimuli["train_counts"].copy()
    for condition in study.CONDITIONS_ALL:
        selected = [r for r in rows if r["condition"] == condition]
        assert [int(r["trial"]) for r in selected] == list(range(1, 9))
        for i, row in enumerate(selected):
            probability = float(row["p_right"])
            action, reward = int(row["action"]), int(row["reward"])
            assert action == int(action_uniforms[i] < probability)
            target = int(counts[i, 1] > counts[i, 0])
            assert reward == (int(reward_uniforms[i] < .5) if condition == "linear_random_reward" else int(action == target))
            assert float(row["optimizer_factor"]) == pytest.approx((action - probability) * (reward - .5))
            assert int(row["left_count"]) == counts[i, 0] and int(row["right_count"]) == counts[i, 1]
        assert float(selected[0]["p_right"]) == .5
        assert float(selected[0]["internal_step_l2"]) == 0
        if condition.startswith("linear_"):
            for row in selected:
                assert all(float(row[name]) == 0 for name in ("kc_left_l2", "kc_right_l2", "mbon_left_l2", "mbon_right_l2", "internal_step_l2"))
        path = output / "seed_81" / condition
        assert not list(path.glob("reference_*.npz"))
        for stage in ("baseline", "final"):
            with np.load(path / f"transfer_{stage}.npz") as transfer:
                assert set(transfer.files) == {"code_values", "score_cache", "map_p_target", "map_greedy_accuracy", "map_p_right"}
                if stage == "baseline":
                    np.testing.assert_array_equal(transfer["score_cache"], np.zeros((27, 8, 2, 14)))
                    np.testing.assert_array_equal(transfer["map_p_target"], np.full((4, 27), .5))


def test_random_reward_stays_identical_when_shared_action_uniforms_are_reversed(completed_reward_fixture, monkeypatch):
    root, config, registration, output, _ = completed_reward_fixture
    original_rng = np.random.default_rng
    class MirroredActions:
        def __init__(self, generator):
            self.generator = generator
        def random(self, n):
            return 1 - self.generator.random(n)
    def intercepted_rng(seed=None):
        generator = original_rng(seed)
        return MirroredActions(generator) if isinstance(seed, int) and seed == 81 + 280000 else generator
    monkeypatch.setattr(study.np.random, "default_rng", intercepted_rng)
    alternate = root / "alternate-actions-output"
    study.run(root, config, registration, alternate)
    old = [r for r in csv_rows(output / "seed_81/trials.csv") if r["condition"] == "linear_random_reward"]
    changed = [r for r in csv_rows(alternate / "seed_81/trials.csv") if r["condition"] == "linear_random_reward"]
    assert old[0]["action"] != changed[0]["action"]
    assert [r["reward"] for r in old] == [r["reward"] for r in changed]
    assert [r["left_count"] for r in old] == [r["left_count"] for r in changed]
    assert [r["right_count"] for r in old] == [r["right_count"] for r in changed]


def test_small_run_keeps_scopes_train_scale_and_resources_separate(completed_reward_fixture):
    root, config, _, output, summary = completed_reward_fixture
    cfg = json.loads(config.read_text(encoding="utf-8"))
    train = study.make_splits(cfg, 81)["train"].inputs()
    resources = json.loads((output / "model_resources.json").read_text(encoding="utf-8"))
    assert len(resources) == 9
    curves = csv_rows(output / "learning_curves.csv")
    assert len(curves) == 9 * 3 * 2
    assert all(float(r["p_target"]) == .5 for r in curves if int(r["trial"]) == 0)
    hashes = json.loads((output / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert {"registration.json", "metadata.json", "summary.json", "model_resources.json", "spacing_codes.npy"} <= set(hashes)
    registered = json.loads((output / "registration.json").read_text(encoding="utf-8"))
    assert registered["source_hashes"] == study.study_hashes(root, cfg)
    for condition in study.CONDITIONS_ALL:
        resource = next(r for r in resources if r["condition"] == condition)
        path = output / "seed_81" / condition
        with np.load(path / "checkpoint_0000.npz") as initial, np.load(path / "checkpoint_0008.npz") as final:
            np.testing.assert_array_equal(initial["readout_weights"], np.zeros(len(initial["readout_weights"])))
            assert not np.array_equal(initial["readout_weights"], final["readout_weights"])
            for key in ("encoder", "bias", "pre", "post", "kc_activity", "feature_scale", "initial_feature_gap_rms"):
                np.testing.assert_array_equal(initial[key], final[key])
            if condition.startswith("linear_"):
                expected_scale = independent_scale(train)
                assert initial["initial_feature_gap_rms"] == pytest.approx(expected_scale)
                assert initial["feature_scale"] == pytest.approx(1 if condition == "linear_raw" else expected_scale)
                assert resource["readout_parameters"] == 7 and resource["internal_parameters"] == 0
                assert initial["mbon_activity"].shape[-1] == 0 and initial["decision_activity"].shape[-1] == 7
            else:
                kc = np.maximum(train @ initial["encoder"].T + initial["bias"], 0)
                raw = np.zeros((*train.shape[:-1], len(initial["readout_weights"])))
                for edge, (pre, post) in enumerate(zip(initial["pre"], initial["post"])):
                    raw[..., post] += kc[..., pre] * initial["internal_weights"][edge]
                assert initial["feature_scale"] == pytest.approx(independent_scale(raw))
                assert resource["readout_parameters"] == 48
                if "plastic" in condition:
                    assert not np.array_equal(initial["internal_weights"], final["internal_weights"])
                    assert resource["internal_parameters"] == len(initial["internal_weights"])
                    assert np.all(final["internal_weights"] >= 0)
                    assert np.all(final["internal_weights"] <= 2 * initial["internal_weights"])
                else:
                    np.testing.assert_array_equal(initial["internal_weights"], final["internal_weights"])
                    assert resource["internal_parameters"] == 0
                np.testing.assert_allclose(initial["decision_activity"], initial["mbon_activity"] / initial["feature_scale"], rtol=0, atol=0)
            assert resource["feature_scale"] == float(initial["feature_scale"])
        outcome = next(r for r in summary["outcomes"] if r["condition"] == condition)
        assert outcome["head_updates"] == 8
        assert outcome["internal_updates"] == (8 if "plastic" in condition else 0)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed" and metadata["initial_head"] == "zero"
    assert metadata["rng"]["actions"] == "seed+280000" and metadata["rng"]["rewards"] == "seed+290000"


def test_external_config_path_is_recorded_as_reproducible_absolute_path(tmp_path):
    root, original, _ = prepare_reward_fixture(tmp_path)
    external = tmp_path / "outside config.json"
    external.write_bytes(original.read_bytes())
    registration = tmp_path / "outside registration.json"
    study.register(root, external, registration)
    output = tmp_path / "external-run"
    study.run(root, external, registration, output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    args = metadata["reproduce_argv"]
    assert args[args.index("--config") + 1] == external.resolve().as_posix()
