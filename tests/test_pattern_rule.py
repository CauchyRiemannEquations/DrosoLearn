"""Pattern unit tests use invented graph/seeds/budgets; no scientific run."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

import drosolearn.pattern_rule as pattern
from test_human_shell_models import small_graph


def fixture_config():
    cfg = copy.deepcopy(pattern.SCIENCE)
    cfg.update(scope="integration_test", model_seeds=[90001, 90002], training_trials=8,
               checkpoints=[0, 4, 8], population_pass_count=2, graph="synthetic-unused")
    return cfg


def write_fixture_catalog(root):
    path = root / pattern.SCIENCE["input_catalog"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"study_id": "pattern_rule_v1", "set_order": list(pattern.SET_ORDER),
                                "catalog": {key: value.tolist() for key, value in pattern.catalog().items()}}), encoding="utf-8")
    return path


def test_config_validation_is_rng_free(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("configuration validation drew a random stream")
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    pattern.validate_config(copy.deepcopy(pattern.SCIENCE))
    pattern.validate_config(fixture_config())


@pytest.mark.parametrize("key,value", [
    ("conditions", list(reversed(pattern.CONDITIONS))), ("training_trials", 800),
    ("head_lr", .2), ("acquisition_threshold", .7), ("ood_max_drop", .2),
    ("population_pass_count", 1), ("model_seeds", [15101]),
    ("phase_diagnostic_is_generalization_test", True), ("rng_namespace", 202610192),
    ("normalization", "all_test_inputs"),
])
def test_scientific_parameters_cannot_change(key, value):
    cfg = copy.deepcopy(pattern.SCIENCE)
    cfg[key] = value
    with pytest.raises(ValueError, match="Fixed pattern science"):
        pattern.validate_config(cfg)


@pytest.mark.parametrize("change", [
    {"model_seeds": [15101]}, {"model_seeds": [90001, 90001]},
    {"model_seeds": [True]}, {"training_trials": 400}, {"training_trials": 7},
    {"checkpoints": [0, 8, 4]}, {"checkpoints": [0, 3, 8]}, {"population_pass_count": 1},
])
def test_integration_scope_cannot_consume_scientific_rng_or_budget(change):
    cfg = fixture_config()
    cfg.update(change)
    with pytest.raises(ValueError):
        pattern.validate_config(cfg)


def test_finite_catalog_and_shortcut_identifiability():
    data = pattern.catalog()
    report = pattern.shortcut_report(data)
    assert data["probe_inputs"].shape == (5, 4, 7)
    assert report["unique_strings"] == 16 and report["evaluation_slots"] == 20
    assert report["new_alphabet_combined_overlap"] == 2
    assert report["phase_novel_labels"] == [0, 0]
    assert [r["oracle_accuracy"] for r in report["sets"]] == [1.] * 5
    assert [r["first_last_accuracy"] for r in report["sets"]] == [1., .5, 1., .5, .5]
    assert [r["exact_lookup_expected_accuracy"] for r in report["sets"]] == [1., .75, .5, .5, .5]
    assert [r["seen_training_inputs"] for r in report["sets"]] == [4, 2, 0, 0, 0]
    for row in report["sets"]:
        assert row["copy_last_accuracy"] == .5
        assert all(cue["best_possible_accuracy"] == .5 for cue in row["cue_audits"].values())


def test_history_and_length_are_explicit_and_not_a_last_three_buffer():
    four = pattern.encode((0, 1, 0, 1))
    six = pattern.encode((0, 1, 0, 1, 0, 1))
    np.testing.assert_array_equal(four[4:6], [0., 0.])
    assert six[5] == 1 / 3 and six[6] == 1.
    assert four[6] == 2 / 3
    assert not np.array_equal(four, six)
    with pytest.raises(ValueError):
        pattern.encode((0, 1, 0))
    with pytest.raises(ValueError):
        pattern.encode((0., 1., 0., 1.))


def test_training_only_rms_never_uses_ood_probe_values():
    data = pattern.catalog()
    reference, probe = data["train_catalog"], data["probe_inputs"].reshape(20, 7)
    left = pattern.VectorActor(small_graph(), 90001, .1, True, reference, probe)
    right = pattern.VectorActor(small_graph(), 90001, .1, True, reference, probe * 100)
    assert left.feature_scale == right.feature_scale
    np.testing.assert_array_equal(left.head, right.head)
    np.testing.assert_array_equal(left.weights, right.weights)
    linear = pattern.VectorSimple("linear", .1, reference, probe)
    assert linear.feature_scale == pytest.approx(np.sqrt(2 / 3))


@pytest.mark.parametrize("kind", ["fixed", "plastic", "linear", "uniform", "lookup_q"])
def test_nogo_hides_the_target_and_changes_no_parameters(kind):
    data = pattern.catalog()
    graph = small_graph()
    actor = pattern.make_actor(kind, pattern.build_graphs(graph, 90001), 90001, .1,
                               data["train_catalog"], data["probe_inputs"].reshape(20, 7))
    before = actor.witness()
    head, internal, value = actor.observe(data["train_catalog"][0], 0, 0)
    assert head == internal == value == 0.
    after = actor.witness()
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    with pytest.raises(ValueError):
        actor.observe(data["train_catalog"][0], 0, 1)


def test_first_positive_feedback_changes_head_before_internal_eligibility():
    data = pattern.catalog()
    actor = pattern.VectorActor(small_graph(), 90001, .1, True,
                                data["train_catalog"], data["probe_inputs"].reshape(20, 7))
    initial = actor.witness()
    assert actor.probability(data["train_catalog"][0]) == .5
    head, internal, value = actor.observe(data["train_catalog"][0], 1, 1)
    assert head > 0 and internal == 0.
    assert value == 0.
    np.testing.assert_array_equal(actor.weights, initial["weights"])
    assert actor.head_updates == actor.internal_updates == 1
    assert np.all(actor.weights >= 0) and np.all(actor.weights <= actor.upper)


def test_head_step_counts_actual_floating_change_not_intended_delta():
    data = pattern.catalog()
    linear = pattern.VectorSimple("linear", .1, data["train_catalog"], data["probe_inputs"].reshape(20, 7))
    linear.head[:] = -1e30
    before = linear.head.copy()
    head, _, _ = linear.observe(data["train_catalog"][0], 1, 1)
    assert head == 0.
    np.testing.assert_array_equal(before, linear.head)
    fixed = pattern.VectorActor(small_graph(), 90001, .1, False,
                                data["train_catalog"], data["probe_inputs"].reshape(20, 7))
    _, f = fixed._parts(data["train_catalog"][0])
    fixed.head[:] = -1e30 * np.sign(f[1] - f[0])
    before = fixed.head.copy()
    head, _, _ = fixed.observe(data["train_catalog"][0], 1, 1)
    assert head == 0.
    np.testing.assert_array_equal(before, fixed.head)


def test_small_producer_records_every_trial_checkpoint_and_weight(monkeypatch, tmp_path):
    cfg = fixture_config()
    monkeypatch.setattr(pattern, "load_projection", lambda unused: small_graph())
    write_fixture_catalog(tmp_path)
    arrays, rows = pattern.produce(cfg, tmp_path)
    summary = pattern.summarize(cfg, arrays, rows)
    assert len(rows) == 9 * 2 * 8
    assert arrays["checkpoint_probabilities"].shape == (9, 2, 3, 5, 4)
    assert summary["counts"]["static_probe_cases"] == 9 * 2 * 3 * 20
    assert summary["counts"]["head_update_calls"] == 7 * 2 * 8
    assert summary["counts"]["internal_update_calls"] == 3 * 2 * 8
    assert summary["counts"]["value_update_calls"] == 2 * 8
    np.testing.assert_array_equal(arrays["checkpoint_probabilities"][:, :, 0], .5)
    np.testing.assert_array_equal(arrays["checkpoint_logits"][:, :, 0], 0.)
    for si in range(2):
        assert all(sorted(block) == [0, 1, 2, 3] for block in arrays["schedules"][si].reshape(-1, 4))
    for row in rows:
        ci = pattern.CONDITIONS.index(row["condition"])
        si = cfg["model_seeds"].index(row["seed"])
        ti = row["trial"] - 1
        assert row["action"] == int(arrays["uniforms"][si, ti] < row["probability_go"])
        assert row["reward"] == ((1 if row["target"] else -1) if row["action"] else 0)
        assert row["head_step_l2"] == arrays["trial_head_steps"][ci, si, ti]
        assert row["value_step_l2"] == arrays["trial_value_steps"][ci, si, ti]
    for condition in pattern.CONDITIONS:
        assert arrays[f"w_{condition}_probe_kc"].shape[:4] == (2, 3, 20, 2)
        weights = arrays[f"w_{condition}_weights"]
        if not condition.endswith("plastic"):
            np.testing.assert_array_equal(weights[:, 0], weights[:, -1])
    # A deterministic replay is a synthetic implementation check only.
    again, again_rows = pattern.produce(cfg, tmp_path)
    assert again_rows == rows
    for key in arrays:
        np.testing.assert_array_equal(again[key], arrays[key])


def test_reward_learned_lookup_is_distinct_from_label_given_memorizer():
    data = pattern.catalog()
    actor = pattern.VectorSimple("lookup_q", .1, data["train_catalog"], data["probe_inputs"].reshape(20, 7))
    assert actor.probability(data["train_catalog"][0]) == .5
    head, internal, value = actor.observe(data["train_catalog"][0], 1, 1)
    assert head == internal == 0 and value == .1
    assert actor.head_updates == actor.internal_updates == 0 and actor.value_updates == 1
    assert actor.probability(data["train_catalog"][0]) == pytest.approx(1 / (1 + np.exp(-.4)))
    before = actor.witness()
    for vector in data["probe_inputs"][2:].reshape(-1, 7):
        assert actor.probability(vector) == .5
    after = actor.witness()
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert actor.values.shape == (4, 2) and len(actor.lookup) == 4
    with pytest.raises(ValueError, match="unregistered exact-input"):
        actor.observe(data["probe_inputs"][2, 0], 1, 1)


def test_lookup_q_updates_selected_value_only_and_negative_feedback_lowers_go():
    data = pattern.catalog()
    actor = pattern.VectorSimple("lookup_q", .1, data["train_catalog"], data["probe_inputs"].reshape(20, 7))
    actor.observe(data["train_catalog"][2], 1, -1)
    expected = np.zeros((4, 2))
    expected[2, 1] = -.1
    np.testing.assert_array_equal(actor.values, expected)
    assert actor.probability(data["train_catalog"][2]) < .5
    assert actor.observe(data["train_catalog"][2], 0, 0) == (0., 0., 0.)
    np.testing.assert_array_equal(actor.values, expected)


def _summary_fixture(monkeypatch, tmp_path):
    cfg = fixture_config()
    monkeypatch.setattr(pattern, "load_projection", lambda unused: small_graph())
    write_fixture_catalog(tmp_path)
    arrays, rows = pattern.produce(cfg, tmp_path)
    return cfg, arrays, rows


def test_good_ood_does_not_rescue_failed_acquisition(monkeypatch, tmp_path):
    cfg, arrays, rows = _summary_fixture(monkeypatch, tmp_path)
    targets = arrays["probe_targets"]
    arrays["checkpoint_probabilities"][:, :, -1] = np.where(targets == 1, .95, .05)
    arrays["checkpoint_probabilities"][:, :, -1, 0] = .5
    summary = pattern.summarize(cfg, arrays, rows)
    assert summary["H_acquisition"]["supported"] is False
    assert all(row["status"] == "not_assessable_without_acquisition" and not row["supported"]
               for row in summary["H_generalization"])


def test_three_new_sets_are_gated_individually_and_phase_never_is(monkeypatch, tmp_path):
    cfg, arrays, rows = _summary_fixture(monkeypatch, tmp_path)
    targets = arrays["probe_targets"]
    arrays["checkpoint_probabilities"][:, :, -1] = np.where(targets == 1, .9, .1)
    arrays["checkpoint_probabilities"][:, :, -1, 3] = .5
    summary = pattern.summarize(cfg, arrays, rows)
    assert summary["H_acquisition"]["supported"] is True
    assert [cell["supported"] for cell in summary["H_generalization"]] == [True, False, True]
    assert all(len(family["seeds"]) == 2 for family in summary["families"])
    assert all(row["sets"][1]["qualified_ood_pass"] is None for row in summary["families"][0]["seeds"])


def test_registered_input_catalog_is_read_and_cannot_be_replaced(tmp_path):
    cfg = fixture_config()
    path = write_fixture_catalog(tmp_path)
    data = pattern.load_catalog(cfg, tmp_path)
    np.testing.assert_array_equal(data["probe_inputs"], pattern.catalog()["probe_inputs"])
    original = path.read_bytes()
    changed = json.loads(original)
    changed["catalog"]["probe_targets"][0][0] = 0
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="Input catalog differs"):
        pattern.load_catalog(cfg, tmp_path)
    path.write_bytes(original)
    changed = json.loads(original)
    changed["catalog"]["probe_seen_in_training"] = np.asarray(changed["catalog"]["probe_seen_in_training"], int).tolist()
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="Input catalog differs"):
        pattern.load_catalog(cfg, tmp_path)


def test_missing_input_catalog_is_not_silently_generated(tmp_path):
    with pytest.raises(FileNotFoundError):
        pattern.load_catalog(fixture_config(), tmp_path)
