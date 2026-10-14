"""Head attenuation state, matched exposure, and prospective decision guards.

Only engineering fixtures 101 and 102 learn. Scientific seeds are checked as
declarations or in arithmetic-only synthetic curves, never fitted here.
"""
from collections import Counter
from pathlib import Path
import copy
import csv
import inspect
import json
import shutil

import numpy as np
import pytest

from drosolearn import head_attenuation as study
from drosolearn import bandit_reversal as previous
from drosolearn.data import analytical_accuracy, assert_disjoint, make_dataset
from drosolearn.graph import Projection
from drosolearn.reward_linear import LinearLearner, NormalizedLearner


ROOT = Path(__file__).resolve().parents[1]
NONPLASTIC = ("linear_normalized", "fixed", "rewired_fixed", "ann_fixed")


def toy_graph():
    return Projection(pre=np.array([0, 1, 1, 2]), post=np.array([0, 0, 1, 1]),
                      weights=np.array([2., 1., 3., 2.]), n_kc=3, n_mbon=2,
                      kc_ids=["a", "b", "c"], mbon_ids=["d", "e"],
                      metadata={"scope": "test fixture", "nested": {"ids": [101]}})


def fixture_config(seeds=None):
    cfg = json.loads((ROOT / "configs/head_attenuation_diagnostic.json").read_text(encoding="utf-8"))
    cfg.update(scope="integration_test", seeds=[101] if seeds is None else seeds, trials_per_phase=8,
               checkpoint_every=4, eval_trials=8, probe_rows=4)
    return cfg


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def saved_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def snapshot(output, condition, history, phase, trial):
    return saved_arrays(output / "seed_101" / condition / history / f"{phase}_{trial:04d}.npz")


def prepare_fixture(path, seeds=None):
    root = Path(path) / "study"
    root.mkdir(parents=True)
    shutil.copytree(ROOT / "src", root / "src")
    shutil.copytree(ROOT / "docs", root / "docs")
    graph = fixture_config()["graph"]
    shutil.copytree(ROOT / graph, root / graph)
    for name in ("data/source-manifest.json", "data/LICENSE.md", "pyproject.toml", "requirements-lock.txt"):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    config = root / "config.json"
    config.write_text(json.dumps(fixture_config(seeds)) + "\n", encoding="utf-8")
    registration = root / "registration.json"
    study.register(root, config, registration)
    return root, config, registration


@pytest.fixture(scope="module")
def completed_fixture(tmp_path_factory):
    root, config, registration = prepare_fixture(tmp_path_factory.mktemp("head-attenuation"))
    output = root / "fixture-output"
    actual_update = LinearLearner.update
    network_update = NormalizedLearner.update
    reference_update = previous.ExpectedLinear.update_expected
    observed = Counter()
    def count_actual(self, *args, **kwargs):
        observed["actual"] += 1
        return actual_update(self, *args, **kwargs)
    def count_network(self, *args, **kwargs):
        observed["actual"] += 1
        return network_update(self, *args, **kwargs)
    def count_reference(self, *args, **kwargs):
        observed["reference"] += 1
        return reference_update(self, *args, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LinearLearner, "update", count_actual)
        patch.setattr(NormalizedLearner, "update", count_network)
        patch.setattr(previous.ExpectedLinear, "update_expected", count_reference)
        summary = study.run(root, config, registration, output)
    # Check actual method calls, not just the producer's CSV accounting.
    assert observed == {"actual": 7 * 6 * 8, "reference": 6 * 8}
    return root, config, registration, output, summary


def make_trained_model(kind):
    train = make_dataset(101, "unit_train", 8).inputs()
    model = (LinearLearner(train) if kind == "linear" else
             previous.ExpectedLinear(train) if kind == "expected" else
             NormalizedLearner(toy_graph(), 102, train, plastic=kind == "plastic"))
    for i, row in enumerate(train):
        if kind == "expected":
            model.update_expected(row, i % 2)
        else:
            model.update(row, i % 2, int(i % 3 != 0))
    # The learner currently consumes pre-generated uniforms. A synthetic RNG
    # verifies deepcopy also preserves any nested state, without drawing it.
    model.diagnostic_rng = np.random.default_rng(102)
    model.diagnostic_nested = {"arrays": [np.array([1., 2.])], "notes": ["prefix"]}
    return model, train


def assert_recursive_equal(before, after):
    if isinstance(before, np.ndarray):
        np.testing.assert_array_equal(before, after)
    elif isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            assert_recursive_equal(before[key], after[key])
    elif isinstance(before, (list, tuple)):
        assert len(before) == len(after)
        for left, right in zip(before, after):
            assert_recursive_equal(left, right)
    elif isinstance(before, np.random.Generator):
        assert_recursive_equal(before.bit_generator.state, after.bit_generator.state)
    elif hasattr(before, "__dict__"):
        assert_recursive_equal(vars(before), vars(after))
    else:
        assert before == after


def independent_margin(model, inputs):
    _, phi = model.features(inputs)
    scores = phi @ model.head
    return scores[..., 1] - scores[..., 0]


def test_prior_reward_and_expected_reference_algorithms_are_reused_without_new_update_apis():
    assert study.LinearLearner is LinearLearner
    assert study.NormalizedLearner is NormalizedLearner
    assert study.ExpectedLinear is previous.ExpectedLinear
    assert study.expected_factor is previous.expected_factor
    for learner in (study.LinearLearner, study.NormalizedLearner):
        assert tuple(inspect.signature(learner.update).parameters) == ("self", "inputs", "action", "reward")
    assert tuple(inspect.signature(study.ExpectedLinear.update_expected).parameters) == ("self", "inputs", "target")


@pytest.mark.parametrize("kind", ["linear", "fixed", "plastic", "expected"])
@pytest.mark.parametrize("gamma", [0., .25, .5, 1.])
def test_boundary_is_a_genuine_deepcopy_changing_only_the_head(kind, gamma):
    prefix, train = make_trained_model(kind)
    before = copy.deepcopy(prefix)
    branch = study.clone_scaled(prefix, gamma)
    assert branch is not prefix
    np.testing.assert_array_equal(branch.head, gamma * prefix.head)
    assert_recursive_equal(before, prefix)
    for key, value in vars(prefix).items():
        if key != "head":
            assert_recursive_equal(value, vars(branch)[key])
            if isinstance(value, np.ndarray) and value.size:
                assert not np.shares_memory(value, vars(branch)[key])
    sibling = study.clone_scaled(prefix, .5)
    branch.head[:] = 3.
    branch.diagnostic_rng.random()
    branch.diagnostic_nested["arrays"][0][0] = -1.
    branch.diagnostic_nested["notes"].append("changed")
    if hasattr(branch, "graph"):
        branch.weights[:] = 0.
        branch.upper[:] = 0.
        branch.graph.weights[:] = 0.
        branch.graph.metadata["nested"]["ids"].append(102)
    assert_recursive_equal(before, prefix)
    pristine_sibling = study.clone_scaled(prefix, .5)
    assert_recursive_equal(pristine_sibling, sibling)
    assert prefix.head_updates == len(train)
    assert prefix.internal_updates == (len(train) if kind == "plastic" else 0)


@pytest.mark.parametrize("gamma", [-1., 1.01, float("nan"), float("inf"), True, "0.25"])
def test_invalid_attenuation_is_rejected_without_touching_prefix(gamma):
    prefix, _ = make_trained_model("plastic")
    before = copy.deepcopy(prefix)
    with pytest.raises(ValueError):
        study.clone_scaled(prefix, gamma)
    assert_recursive_equal(before, prefix)


@pytest.mark.parametrize("kind", ["linear", "fixed", "plastic", "expected"])
@pytest.mark.parametrize("gamma", [.25, .5, 1.])
def test_positive_gamma_multiplies_raw_margin_and_preserves_sign_order_without_refitting_features(kind, gamma):
    prefix, train = make_trained_model(kind)
    prefix.head[:] = np.linspace(-.7, .3, len(prefix.head))
    branch = study.clone_scaled(prefix, gamma)
    before = independent_margin(prefix, train)
    after = independent_margin(branch, train)
    np.testing.assert_array_equal(after, gamma * before)
    np.testing.assert_array_equal(np.sign(after), np.sign(before))
    np.testing.assert_array_equal(np.argsort(after), np.argsort(before))
    np.testing.assert_array_equal(study.raw_margin(prefix, train), before)
    np.testing.assert_array_equal(study.raw_margin(branch, train), after)
    assert branch.feature_scale == prefix.feature_scale
    np.testing.assert_array_equal(branch.features(train)[1], prefix.features(train)[1])


def test_sigmoid_and_probability_tolerance_ties_do_not_erase_nonzero_raw_margin_rank():
    pair = np.zeros((4, 2, 7))
    pair[:, 0, 0] = [.25, .5, .25, .5]
    pair[:, 1, 0] = [.5, .25, .5, .25]
    prefix = LinearLearner(pair)
    prefix.head[0] = 1e-17
    branch = study.clone_scaled(prefix, .25)
    np.testing.assert_array_equal(branch.probabilities(pair), np.full(4, .5))
    original, attenuated = study.raw_margin(prefix, pair), study.raw_margin(branch, pair)
    assert np.all(original != 0) and np.all(attenuated != 0)
    np.testing.assert_array_equal(np.sign(original), np.sign(attenuated))
    np.testing.assert_array_equal(np.argsort(original), np.argsort(attenuated))


def test_plastic_eligibility_also_scales_with_the_old_head_and_zero_prevents_first_internal_step():
    train = make_dataset(101, "eligibility_train", 8).inputs()
    prefix = NormalizedLearner(toy_graph(), 102, train, plastic=True,
                               head_lr=.13, internal_lr=.0001)
    prefix.head[:] = [.7, -.4]
    pair = train[0]
    kc, _ = prefix.raw_activities(pair)
    original_eligibility = prefix.head[prefix.post] / prefix.feature_scale * (kc[1, prefix.pre] - kc[0, prefix.pre])
    for gamma in (0., .25, .5, 1.):
        branch = study.clone_scaled(prefix, gamma)
        old_weights, old_head = branch.weights.copy(), branch.head.copy()
        probability = float(branch.probabilities(pair))
        factor = (1 - probability) * .5
        eligibility = old_head[branch.post] / branch.feature_scale * (kc[1, branch.pre] - kc[0, branch.pre])
        np.testing.assert_array_equal(eligibility, gamma * original_eligibility)
        branch.update(pair, 1, 1)
        expected_weights = np.clip(old_weights + .0001 * factor * eligibility, 0, branch.upper)
        np.testing.assert_allclose(branch.weights, expected_weights, rtol=0, atol=1e-15)
        assert branch.head_updates == branch.internal_updates == 1
        if gamma == 0:
            np.testing.assert_array_equal(branch.weights, old_weights)
            assert np.linalg.norm(branch.head) > 0


@pytest.mark.parametrize("field,value", [
    ("scope", "retuned"), ("seeds", []), ("seeds", [101, 101]),
    ("seeds", [True]), ("seeds", [-1]), ("seeds", [8001]),
    ("conditions", ["linear_normalized"]), ("conditions", ["fixed", "fixed"]),
    ("histories", ["fresh_low"]), ("trials_per_phase", 10),
    ("trials_per_phase", 0), ("checkpoint_every", 3), ("eval_trials", 6),
    ("probe_rows", 9), ("head_lr", float("nan")), ("internal_lr", -1),
])
def test_invalid_or_scientific_development_fixture_design_is_rejected(field, value):
    cfg = fixture_config()
    cfg[field] = value
    with pytest.raises(ValueError):
        study.validate_config(cfg)


@pytest.mark.parametrize("field,value", [
    ("gammas", [0., .5, .25, 1.]), ("gammas", [0., .25, 1.]),
    ("gammas", [False, .25, .5, True]), ("primary_gamma", .5),
    ("attenuation_mode", "reinitialize_internal"), ("histories", ["gamma_1", "gamma_05", "gamma_025", "gamma_0", "fresh_low"]),
])
def test_attenuation_levels_order_and_primary_contrast_cannot_be_reselected(field, value):
    cfg = fixture_config()
    cfg[field] = value
    with pytest.raises(ValueError):
        study.validate_config(cfg)


def test_scientific_declarations_fix_new_seeds_seven_families_and_actual_budget():
    cfg = json.loads((ROOT / "configs/head_attenuation_diagnostic.json").read_text(encoding="utf-8"))
    study.validate_config(cfg)
    assert cfg["seeds"] == list(range(8001, 8011))
    assert tuple(cfg["conditions"]) == ("linear_normalized", "fixed", "plastic",
                                       "rewired_fixed", "rewired_plastic", "ann_fixed", "ann_plastic")
    assert tuple(cfg[k] for k in ("trials_per_phase", "checkpoint_every", "eval_trials", "probe_rows", "head_lr", "internal_lr")) == (400, 100, 128, 8, .1, .01)
    assert cfg["gammas"] == [0., .25, .5, 1.]
    assert cfg["primary_gamma"] == .25
    assert cfg["criteria"]["endpoint_benefit"] == .15
    assert tuple(cfg["histories"]) == ("fresh_low", "gamma_0", "gamma_025", "gamma_05", "gamma_1")
    assert len(cfg["seeds"]) * len(cfg["conditions"]) * (400 + 4 * 400 + 400) == 168000
    assert len(cfg["seeds"]) * (400 + 4 * 400 + 400) == 24000


def test_new_data_are_exactly_crossed_disjoint_and_reproducible_without_scientific_seeds():
    cfg = fixture_config()
    first, repeated = study.make_splits(cfg, 102), study.make_splits(cfg, 102)
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


@pytest.mark.parametrize("changed", ["config", "protocol", "code"])
def test_registered_config_protocol_or_source_drift_stops_before_creating_any_result_directory(tmp_path, changed):
    root, config, registration = prepare_fixture(tmp_path)
    target = {"config": config, "protocol": root / "docs/head-attenuation-protocol.md",
              "code": root / "src/drosolearn/head_attenuation.py"}[changed]
    with target.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    output = root / "must-not-exist"
    with pytest.raises(ValueError):
        study.run(root, config, registration, output)
    assert not output.exists()


def test_registration_and_existing_output_are_exclusive_and_preserve_previous_failures(tmp_path):
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


@pytest.mark.parametrize("condition", ["linear_normalized", "fixed", "plastic", "rewired_fixed", "rewired_plastic", "ann_fixed", "ann_plastic"])
def test_each_family_fits_exactly_one_high_prefix_and_uses_its_state_for_all_four_branches(completed_fixture, condition):
    output = completed_fixture[3]
    prefix = snapshot(output, condition, "shared_prefix", "phase1", 8)
    trials = csv_rows(output / "seed_101/trials.csv")
    high_rows = [row for row in trials if row["condition"] == condition and row["phase"] == "phase1"]
    assert len(high_rows) == 8
    assert {row["history"] for row in high_rows} == {"shared_prefix"}
    assert int(prefix["head_updates"]) == 8
    assert int(prefix["internal_updates"]) == (8 if condition.endswith("plastic") else 0)
    for history in ("gamma_0", "gamma_025", "gamma_05", "gamma_1"):
        before = saved_arrays(output / "seed_101" / condition / history / "boundary_before.npz")
        assert before.keys() == prefix.keys()
        for key in prefix:
            np.testing.assert_array_equal(before[key], prefix[key])
        final = snapshot(output, condition, history, "phase2", 8)
        assert int(final["head_updates"]) == 16
        assert int(final["internal_updates"]) == (16 if condition.endswith("plastic") else 0)


@pytest.mark.parametrize("condition", ["linear_normalized", "fixed", "plastic", "rewired_fixed", "rewired_plastic", "ann_fixed", "ann_plastic"])
def test_saved_attenuation_boundary_preserves_internal_state_bounds_encoder_scale_and_counts(completed_fixture, condition):
    output = completed_fixture[3]
    unchanged = ("internal_weights", "initial_internal_weights", "internal_upper_bounds", "incoming_normalizer", "encoder", "bias", "pre", "post",
                 "kc_activity", "mbon_activity", "decision_activity", "feature_scale", "initial_feature_gap_rms",
                 "head_updates", "internal_updates")
    for history, gamma in (("gamma_0", 0.), ("gamma_025", .25), ("gamma_05", .5), ("gamma_1", 1.)):
        path = output / "seed_101" / condition / history
        before, after = (saved_arrays(path / f"boundary_{side}.npz") for side in ("before", "after"))
        initial_low = snapshot(output, condition, history, "phase2", 0)
        assert before.keys() == after.keys() == initial_low.keys()
        np.testing.assert_array_equal(after["readout_weights"], gamma * before["readout_weights"])
        for key in unchanged:
            np.testing.assert_array_equal(before[key], after[key])
        for split in ("iid", "ood"):
            margin_before, margin_after = before[f"{split}_raw_margin"], after[f"{split}_raw_margin"]
            np.testing.assert_array_equal(margin_after, gamma * margin_before)
            if gamma > 0:
                np.testing.assert_array_equal(np.sign(margin_after), np.sign(margin_before))
                np.testing.assert_array_equal(np.argsort(margin_after), np.argsort(margin_before))
            else:
                np.testing.assert_array_equal(margin_after, np.zeros(8))
            p_before, p_after = before[f"{split}_p_right"], after[f"{split}_p_right"]
            confidence_before, confidence_after = np.maximum(p_before, 1 - p_before), np.maximum(p_after, 1 - p_after)
            assert np.all(confidence_after <= confidence_before + 2e-15)
            np.testing.assert_array_equal(p_after, 1 / (1 + np.exp(-np.clip(margin_after, -40, 40))))
        for key in after:
            np.testing.assert_array_equal(after[key], initial_low[key])
        if gamma == 0:
            np.testing.assert_array_equal(after["choice_probability"], np.full(4, .5))
        if gamma == 1:
            for key in before:
                np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize("condition", NONPLASTIC)
def test_zero_branch_is_exactly_fresh_for_fixed_features_at_every_checkpoint_and_update(completed_fixture, condition):
    output = completed_fixture[3]
    for trial in (0, 4, 8):
        fresh = snapshot(output, condition, "fresh_low", "phase2", trial)
        zero = snapshot(output, condition, "gamma_0", "phase2", trial)
        assert fresh.keys() == zero.keys()
        for key in fresh:
            if key in ("head_updates", "internal_updates"):
                continue
            np.testing.assert_array_equal(fresh[key], zero[key])
    def low_rows(history):
        return [{key: value for key, value in row.items() if key != "history"}
                for row in csv_rows(output / "seed_101/trials.csv")
                if row["condition"] == condition and row["history"] == history]
    assert low_rows("fresh_low") == low_rows("gamma_0")


def test_fresh_control_skips_high_training_and_all_histories_keep_phase_one_train_only_scale(completed_fixture):
    output = completed_fixture[3]
    stimuli = saved_arrays(output / "seed_101/stimuli.npz")
    train = np.concatenate((stimuli["train_phase1_counts"][..., None] / 8.,
                            .1 * (stimuli["train_phase1_nuisance"] - .5)), axis=-1)
    assert not [row for row in csv_rows(output / "seed_101/trials.csv")
                if row["history"] == "fresh_low" and row["phase"] == "phase1"]
    for condition in fixture_config()["conditions"]:
        initial = snapshot(output, condition, "shared_prefix", "phase1", 0)
        if condition == "linear_normalized":
            raw = train
        else:
            kc = np.maximum(train @ initial["encoder"].T + initial["bias"], 0)
            raw = np.stack([np.bincount(initial["post"],
                                       weights=row[initial["pre"]] * initial["internal_weights"],
                                       minlength=len(initial["readout_weights"]))
                            for row in kc.reshape(-1, kc.shape[-1])]).reshape(8, 2, -1)
        independent_scale = float(np.sqrt(np.mean(np.sum((raw[:, 1] - raw[:, 0]) ** 2, axis=-1))))
        assert float(initial["feature_scale"]) == pytest.approx(independent_scale, rel=2e-14)
        np.testing.assert_array_equal(initial["readout_weights"], np.zeros_like(initial["readout_weights"]))
        np.testing.assert_array_equal(initial["internal_upper_bounds"], 2 * initial["initial_internal_weights"])
        for history in fixture_config()["histories"]:
            for trial in (0, 4, 8):
                current = snapshot(output, condition, history, "phase2", trial)
                np.testing.assert_array_equal(current["feature_scale"], initial["feature_scale"])
                np.testing.assert_array_equal(current["initial_feature_gap_rms"], initial["initial_feature_gap_rms"])
        fresh = snapshot(output, condition, "fresh_low", "phase2", 0)
        assert int(fresh["head_updates"]) == int(fresh["internal_updates"]) == 0
        np.testing.assert_array_equal(fresh["readout_weights"], np.zeros_like(fresh["readout_weights"]))
        final = snapshot(output, condition, "fresh_low", "phase2", 8)
        assert int(final["head_updates"]) == 8
        assert int(final["internal_updates"]) == (8 if condition.endswith("plastic") else 0)


def test_budget_counts_actual_prefix_once_and_labels_privileged_reference_separately(completed_fixture):
    output, summary = completed_fixture[3:]
    rows = csv_rows(output / "seed_101/trials.csv")
    assert len(rows) == summary["actual_training_trials"] == 7 * (8 + 4 * 8 + 8)
    assert {row["condition"] for row in rows} == set(fixture_config()["conditions"])
    expected = csv_rows(output / "seed_101/expected_updates.csv")
    assert len(expected) == summary["expected_reference_updates"] == 8 + 4 * 8 + 8
    assert all("action" not in row and "reward" not in row for row in expected)
    assert Counter(row["history"] for row in expected) == {history: 8 for history in ("shared_prefix", *fixture_config()["histories"])}
    for condition in fixture_config()["conditions"]:
        selected = [row for row in rows if row["condition"] == condition]
        assert Counter(row["history"] for row in selected) == {history: 8 for history in ("shared_prefix", *fixture_config()["histories"])}


def test_shared_low_data_and_uniforms_determine_every_branch_action_and_reward(completed_fixture):
    output = completed_fixture[3]
    stimuli = saved_arrays(output / "seed_101/stimuli.npz")
    for phase, offset in (("phase1", 280000), ("phase2", 300000)):
        np.testing.assert_array_equal(stimuli[f"action_{phase}_uniforms"], np.random.default_rng(101 + offset).random(8))
    for row in csv_rows(output / "seed_101/trials.csv"):
        phase, trial = row["phase"], int(row["trial"]) - 1
        probability = float(row["p_right"])
        uniform = stimuli[f"action_{phase}_uniforms"][trial]
        assert int(row["action"]) == int(uniform < probability)
        dataset = "train_phase1" if phase == "phase1" else "train_phase2"
        counts = stimuli[f"{dataset}_counts"][trial]
        assert [int(row["left_count"]), int(row["right_count"])] == list(counts)
        assert row["family_id"] == str(stimuli[f"{dataset}_family_ids"][trial])
        target = int(np.argmax(counts) if phase == "phase1" else np.argmin(counts))
        assert int(row["reward"]) == int(int(row["action"]) == target)
        if phase == "phase1":
            assert row["history"] == "shared_prefix"
        else:
            assert row["history"] in fixture_config()["histories"]
    expected = csv_rows(output / "seed_101/expected_updates.csv")
    for row in expected:
        phase = row["phase"]
        counts = [int(row["left_count"]), int(row["right_count"])]
        target = int(np.argmax(counts) if phase == "phase1" else np.argmin(counts))
        assert int(row["target"]) == target
        assert float(row["expected_factor"]) == pytest.approx(previous.expected_factor(float(row["p_right"]), target), abs=1e-15)


def test_metadata_and_boundary_links_make_the_single_prefix_and_separate_reference_reviewable(completed_fixture):
    output = completed_fixture[3]
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["config"] == fixture_config()
    assert metadata["reproduce_argv"][2] == "drosolearn.head_attenuation"
    assert metadata["rng"]["phase1_actions"] == "seed+280000"
    assert metadata["rng"]["phase2_actions"] == "seed+300000"
    assert "label-privileged" in metadata["reference_scope"]
    assert "not ensemble expectation" in metadata["reference_scope"]
    resources = json.loads((output / "model_resources.json").read_text(encoding="utf-8"))
    assert len(resources) == 8 * 5
    assert all(row["algorithm"] == "conditional_expected_action_reference" for row in resources if row["condition"] == "expected_linear")
    boundaries = json.loads((output / "boundaries.json").read_text(encoding="utf-8"))
    assert len(boundaries) == 8 * 5
    for row in boundaries:
        fresh = row["history"] == "fresh_low"
        assert row["head_updates_before"] == row["head_updates_after"] == (0 if fresh else 8)
        assert row["internal_change_l2"] == 0
        assert row["feature_scale_before"] == row["feature_scale_after"]
        if fresh:
            assert row["gamma"] is row["shared_prefix_path"] is row["shared_prefix_sha256"] is None
        else:
            prefix = output / row["shared_prefix_path"]
            assert prefix == output / "seed_101" / row["condition"] / "shared_prefix/phase1_0008.npz"
            assert row["shared_prefix_sha256"] == previous.sha256(prefix)


def synthetic_summary_fixture(benefits=None, acquired_count=8, fresh_endpoint=.55):
    """Ten-seed declarations, with arithmetic-only synthetic checkpoint rows."""
    cfg = json.loads((ROOT / "configs/head_attenuation_diagnostic.json").read_text(encoding="utf-8"))
    benefits = [.1875] * 8 + [0.] * 2 if benefits is None else benefits
    assert len(benefits) == 10
    curves = []
    checkpoints = list(range(0, 401, 100))
    for i, seed in enumerate(cfg["seeds"]):
        for condition in (*cfg["conditions"], "expected_linear"):
            high_endpoint = .75 if i < acquired_count else .6
            prefix = np.linspace(.5, high_endpoint, len(checkpoints))
            for split in ("iid", "ood"):
                for trial, value in zip(checkpoints, prefix):
                    curves.append(dict(seed=seed, condition=condition, history="shared_prefix",
                                       phase="phase1", trial=trial, split=split,
                                       p_target=float(value), greedy_accuracy=1.))
            endpoints = {"fresh_low": fresh_endpoint, "gamma_0": .95,
                         "gamma_025": .5625 + benefits[i], "gamma_05": .95,
                         "gamma_1": .5625}
            before = {"fresh_low": .5, "gamma_0": .5, "gamma_025": .4,
                      "gamma_05": .35, "gamma_1": .25}
            for history in cfg["histories"]:
                sequence = np.linspace(before[history], endpoints[history], len(checkpoints))
                for split in ("iid", "ood"):
                    for trial, value in zip(checkpoints, sequence):
                        curves.append(dict(seed=seed, condition=condition, history=history,
                                           phase="phase2", trial=trial, split=split,
                                           p_target=float(value), greedy_accuracy=1.))
    return cfg, curves


def test_h_a_requires_all_three_registered_gates_and_accepts_the_exact_eight_seed_boundary():
    cfg, curves = synthetic_summary_fixture()
    primary = study.summarize(cfg, curves)["primary_hypothesis"]
    assert primary["name"] == "H_A"
    assert primary["condition"] == "linear_normalized"
    assert primary["n_seeds"] == 10
    assert primary["mean_endpoint_benefit"] == .15
    assert primary["positive_seed_count"] == primary["high_prefix_acquisition_count"] == 8
    assert primary["fresh_acquisition_count"] == 0
    assert primary["criterion_met"] is True


@pytest.mark.parametrize("benefits,acquired_count,mean,positive,acquired", [
    ([.25] * 8 + [-.3] * 2, 8, .14, 8, 8),
    ([.25] * 7 + [0.] * 3, 8, .175, 7, 8),
    ([.25] * 10, 7, .25, 10, 7),
])
def test_every_h_a_gate_can_fail_independently_and_failed_seed_endpoints_remain_in_all_ten_denominators(benefits, acquired_count, mean, positive, acquired):
    cfg, curves = synthetic_summary_fixture(benefits, acquired_count)
    summary = study.summarize(cfg, curves)
    primary = summary["primary_hypothesis"]
    assert primary["n_seeds"] == 10
    assert primary["mean_endpoint_benefit"] == pytest.approx(mean)
    assert primary["positive_seed_count"] == positive
    assert primary["high_prefix_acquisition_count"] == acquired
    assert primary["criterion_met"] is False
    selected = [row for row in summary["paired_contrasts"] if row["condition"] == "linear_normalized"]
    assert len(selected) == 10
    np.testing.assert_allclose([row["endpoint_benefit"] for row in selected], benefits, rtol=0, atol=1e-15)
    assert len(summary["outcomes"]) == 10 * 8 * 5
    assert len(summary["prefix_outcomes"]) == 10 * 8


def test_descriptive_reset_half_and_fresh_success_cannot_replace_a_failed_quarter_primary():
    cfg, curves = synthetic_summary_fixture([0.] * 10, acquired_count=10, fresh_endpoint=.95)
    summary = study.summarize(cfg, curves)
    assert summary["primary_hypothesis"]["criterion_met"] is False
    assert summary["primary_hypothesis"]["mean_endpoint_benefit"] == 0
    for history in ("fresh_low", "gamma_0", "gamma_05"):
        selected = [row for row in summary["outcomes"]
                    if row["condition"] == "linear_normalized" and row["history"] == history]
        assert len(selected) == 10
        assert all(row["phase2_after"] == .95 and row["qualified_adaptation"] for row in selected)


def test_adaptation_qualification_is_endpoint_plus_prefix_and_has_no_extra_gain_gate():
    cfg, curves = synthetic_summary_fixture([.1875] * 10, acquired_count=8)
    for row in curves:
        if row["history"] == "gamma_025" and row["phase"] == "phase2":
            row["p_target"] = .74 + .01 * row["trial"] / 400
    summary = study.summarize(cfg, curves)
    selected = [row for row in summary["outcomes"]
                if row["condition"] == "linear_normalized" and row["history"] == "gamma_025"]
    assert all(row["phase2_after"] == .75 for row in selected)
    assert all(row["phase2_gain"] == pytest.approx(.01) for row in selected)
    assert all(row["phase2_target_passed"] for row in selected)
    assert all(row["qualified_adaptation"] for row in selected[:8])
    assert all(not row["phase1_acquired"] and not row["qualified_adaptation"] for row in selected[8:])
    assert all(row["first_observed_75_checkpoint"] == 400 for row in selected)


def test_expected_reference_has_no_scientific_success_flags_or_primary_family_verdict():
    cfg, curves = synthetic_summary_fixture()
    summary = study.summarize(cfg, curves)
    references = [row for row in summary["outcomes"] if row["condition"] == "expected_linear"]
    assert len(references) == 10 * 5
    assert all(row["algorithm"] == "conditional_expected_action_reference" for row in references)
    flags = ("phase1_acquired", "fresh_acquired", "phase2_target_passed", "qualified_adaptation", "first_observed_75_checkpoint")
    assert all(row[key] is None for row in references for key in flags)
    assert all(row["acquired"] is None for row in summary["prefix_outcomes"] if row["condition"] == "expected_linear")
    assert {row["condition"] for row in summary["family_verdicts"]} == set(cfg["conditions"])
    assert {row["condition"] for row in summary["paired_contrasts"]} == set(cfg["conditions"])
    assert summary["actual_training_trials"] == 168000
    assert summary["expected_reference_updates"] == 24000


def test_immediate_boundary_difference_recovery_and_auc_are_separate_arithmetic_quantities():
    cfg, curves = synthetic_summary_fixture([.2375] * 10, acquired_count=10)
    summary = study.summarize(cfg, curves)
    quarter = next(row for row in summary["outcomes"] if row["seed"] == 8001 and row["condition"] == "linear_normalized" and row["history"] == "gamma_025")
    full = next(row for row in summary["outcomes"] if row["seed"] == 8001 and row["condition"] == "linear_normalized" and row["history"] == "gamma_1")
    contrast = next(row for row in summary["paired_contrasts"] if row["seed"] == 8001 and row["condition"] == "linear_normalized")
    assert quarter["phase2_auc"] == pytest.approx(.6)
    assert quarter["phase2_baseline_adjusted_auc"] == pytest.approx(.2)
    assert quarter["frozen_no_learning_endpoint"] == quarter["frozen_no_learning_auc"] == .4
    assert contrast["endpoint_benefit"] == pytest.approx(.2375)
    assert contrast["initial_probability_difference"] == pytest.approx(.15)
    assert contrast["gain_difference"] == pytest.approx(.0875)
    assert contrast["endpoint_benefit"] == pytest.approx(contrast["initial_probability_difference"] + contrast["gain_difference"])
    assert contrast["raw_auc_difference"] == pytest.approx(quarter["phase2_auc"] - full["phase2_auc"])
    assert contrast["baseline_adjusted_auc_difference"] == pytest.approx(contrast["raw_auc_difference"] - contrast["initial_probability_difference"])
