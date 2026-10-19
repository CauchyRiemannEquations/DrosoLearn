"""Independent arithmetic tests: only finite catalogs and invented tiny graphs."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("pattern_independent_test", Path(__file__).resolve().parents[1] / "scripts/verify_pattern_rule.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture_config():
    return {"study_id": "pattern_rule_v1", "scope": "integration_test", "graph": "unused-synthetic",
            "conditions": list(audit.CONDITIONS), "model_seeds": [90071, 90072], "training_trials": 8,
            "checkpoints": [0, 4, 8], "head_lr": .1, "internal_lr_multiplier": .1,
            "rng_namespace": 202610191, "primary_condition": "plastic", "set_order": list(audit.SETS),
            "lookup_alpha": .1, "lookup_beta": 4.,
            "acquisition_threshold": .8, "acquisition_gain": .25, "class_threshold": .7,
            "ood_threshold": .75, "ood_max_drop": .1, "population_pass_count": 2,
            "normalization": "initial_training_four_input_feature_gap_RMS",
            "representation": "six_raw_symbol_slots_divided_by_3_then_length_divided_by_6",
            "input_catalog": audit.INPUT_CATALOG, "input_files": [audit.INPUT_CATALOG],
            "phase_diagnostic_is_generalization_test": False}


def tiny_graph():
    n = 16
    return {"pre": np.arange(n), "post": np.arange(n), "counts": np.ones(n), "n_kc": n, "n_mbon": n}


def write_input_catalog(root, document=None):
    if document is None:
        document = {"study_id": "pattern_rule_v1", "set_order": list(audit.SETS),
                    "catalog": {key: value.tolist() for key, value in audit.independent_catalog_arrays().items()}}
    path = root / audit.INPUT_CATALOG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf8")
    return path


def test_exact_history_and_padding_semantics():
    a = audit.independent_catalog_arrays()
    assert a["train_catalog"].shape == (4, 7) and a["probe_inputs"].shape == (5, 4, 7)
    assert np.array_equal(a["probe_targets"], np.tile([1, 1, 0, 0], (5, 1)))
    assert np.all(a["probe_strings"][:4, :, 4:] == -1)
    assert np.all(a["probe_inputs"][:4, :, 4:6] == 0)
    assert a["probe_inputs"][4, 0, 5] == pytest.approx(1 / 3) and a["probe_inputs"][4, 0, 6] == 1.


def test_exact_oracle_shortcut_and_overlap_combinatorics():
    report = audit.analytical_catalog()
    assert (report["slots"], report["unique_strings"], report["alphabet_combined_overlap"]) == (20, 16, 2)
    assert report["phase_novel_only_class_counts"] == {"0": 2, "1": 0}
    assert report["phase_novel_only_always_nogo_accuracy"] == 1.
    tables = report["by_set"]
    assert all(r["oracle_correct"] == 4 for r in tables.values())
    assert [tables[name]["first_last_correct"] for name in audit.SETS] == [4, 2, 4, 2, 2]
    assert [tables[name]["lookup_expected_accuracy"] for name in audit.SETS] == [1., .75, .5, .5, .5]
    assert all(c["accuracy"] == .5 for r in tables.values() for c in r["single_cue_optimal"].values())


def test_train_only_scale_and_evaluation_cannot_change_parameters():
    inputs, _ = audit.catalog()
    actor = audit.AuditActor("fixed", 90071, tiny_graph(), inputs[0])
    before = actor.witness(inputs.reshape(20, 7))
    actor.probability(inputs[2] * 1000)
    assert actor.scale == before["feature_scale"]
    assert np.array_equal(actor.head, before["head"]) and np.array_equal(actor.weights, before["weights"])
    assert np.array_equal(actor.probability(inputs[0]), np.full(4, .5))


def test_old_head_eligibility_and_hidden_nogo_observation():
    inputs, _ = audit.catalog()
    actor = audit.AuditActor("plastic", 90072, tiny_graph(), inputs[0])
    weights = actor.weights.copy()
    first_head, first_internal, first_value = actor.update(inputs[0, 0], 1, 1)
    assert first_head > 0 and first_internal == first_value == 0
    assert np.array_equal(actor.weights, weights)
    with pytest.raises(ValueError): actor.update(inputs[0, 0], 0, -1)
    head = actor.head.copy(); actor.update(inputs[0, 0], 0, 0)
    assert np.array_equal(actor.head, head) and np.array_equal(actor.weights, weights)


def test_reward_lookup_table_is_not_analytical_label_lookup():
    inputs, _ = audit.catalog()
    actor = audit.AuditActor("lookup_q", 90071, None, inputs[0])
    assert np.all(actor.probability(inputs.reshape(20, 7)) == .5)
    actor.update(inputs[0, 0], 1, 1)
    assert actor.values[0, 1] == pytest.approx(.1)
    assert actor.probability(inputs[0, 0]) == pytest.approx(1 / (1 + np.exp(-.4)))
    before = actor.values.copy()
    assert np.all(actor.probability(inputs[2]) == .5)
    assert np.array_equal(actor.values, before) and actor.values.shape == (4, 2)
    with pytest.raises(ValueError): actor.update(inputs[2, 0], 1, 1)


@pytest.mark.parametrize("condition", ["linear", "uniform", "lookup_q"])
def test_simple_dimensions_and_separate_update_counters(condition):
    inputs, _ = audit.catalog()
    actor = audit.AuditActor(condition, 90071, None, inputs[0])
    w = actor.witness(inputs.reshape(20, 7))
    assert w["probe_kc"].shape == (20, 2, 0)
    assert w["probe_decision"].shape == (20, 2, 7 if condition == "linear" else 0)
    actor.update(inputs[0, 0], 1, 1)
    assert actor.head_updates == int(condition == "linear")
    assert actor.value_updates == int(condition == "lookup_q") and actor.internal_updates == 0


@pytest.mark.parametrize("key,value", [("head_lr", .3), ("rng_namespace", 99), ("class_threshold", .1), ("lookup_alpha", .3), ("lookup_beta", 8.),
                                       ("phase_diagnostic_is_generalization_test", True), ("representation", "last_three")])
def test_independent_configuration_rejects_rule_or_threshold_change(key, value):
    cfg = fixture_config(); cfg[key] = value
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


def test_development_scope_cannot_authorize_scientific_seed():
    cfg = fixture_config(); cfg["model_seeds"] = [15101]
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


@pytest.fixture
def independently_replayed():
    cfg = fixture_config(); arrays, rows = audit.expected_artifacts(cfg, tiny_graph())
    return cfg, arrays, rows, audit.expected_summary(cfg, arrays)


def test_independent_budget_and_full_artifact_verification(independently_replayed):
    cfg, a, rows, summary = independently_replayed
    assert a["trial_actions"].shape == (9, 2, 8) and len(rows) == 144
    assert a["checkpoint_probabilities"].shape == (9, 2, 3, 5, 4)
    assert summary["counts"]["static_probe_cases"] == 1080
    assert summary["counts"]["head_update_calls"] == 112
    assert summary["counts"]["internal_update_calls"] == 48
    assert summary["counts"]["value_update_calls"] == 16
    checks = audit.verify_against_graph(cfg, tiny_graph(), a, rows, summary, allow_fixture=True)
    assert checks["independently_replayed_training_actions"] == 144


@pytest.mark.parametrize("field", ["train_catalog", "probe_lengths", "schedules", "uniforms", "trial_actions",
                                    "trial_rewards", "trial_head_steps", "trial_value_steps", "checkpoint_logits",
                                    "w_plastic_weights", "w_fixed_feature_scale", "w_linear_head", "w_lookup_q_value_table"])
def test_state_trial_or_catalog_tamper_rejected(independently_replayed, field):
    cfg, original, rows, summary = independently_replayed
    arrays = {k: value.copy() for k, value in original.items()}
    flat = arrays[field].reshape(-1); flat[-1] += 1 if flat.dtype.kind in "biu" else .2
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, tiny_graph(), arrays, rows, summary, allow_fixture=True)


def test_row_order_and_preupdate_probability_cannot_change(independently_replayed):
    cfg, arrays, original, summary = independently_replayed
    rows = copy.deepcopy(original); rows[0]["probability_go"] = .75
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, tiny_graph(), arrays, rows, summary, allow_fixture=True)
    rows = copy.deepcopy(original); rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, tiny_graph(), arrays, rows, summary, allow_fixture=True)


def test_no_acquisition_blocks_all_ood_success(independently_replayed):
    cfg, original, _, _ = independently_replayed
    arrays = copy.deepcopy(original); p = arrays["checkpoint_probabilities"]; p[:] = .5
    p[:, :, -1, 2:, :2] = .99; p[:, :, -1, 2:, 2:] = .01
    summary = audit.expected_summary(cfg, arrays)
    assert summary["H_acquisition"]["supported"] is False
    assert all(not cell["supported"] and cell["status"] == "not_assessable_without_acquisition" for cell in summary["H_generalization"])


def test_one_bad_class_blocks_acquisition(independently_replayed):
    cfg, original, _, _ = independently_replayed
    arrays = copy.deepcopy(original); p = arrays["checkpoint_probabilities"]; p[:] = .5
    p[:, :, -1, 0, :2] = 1.; p[:, :, -1, 0, 2:] = .31
    summary = audit.expected_summary(cfg, arrays)
    assert summary["families"][0]["seeds"][0]["sets"][0]["expected_accuracy"] > .8
    assert summary["families"][0]["acquisition_supported"] is False


def test_extra_claim_or_false_gate_is_rejected(independently_replayed):
    cfg, arrays, rows, original = independently_replayed
    summary = copy.deepcopy(original); summary["general_math_understanding"] = True
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, tiny_graph(), arrays, rows, summary, allow_fixture=True)
    summary = copy.deepcopy(original); summary["H_acquisition"]["supported"] = not summary["H_acquisition"]["supported"]
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, tiny_graph(), arrays, rows, summary, allow_fixture=True)


def test_full_producer_matches_independent_synthetic_replay(monkeypatch, tmp_path):
    from drosolearn import pattern_rule
    from drosolearn.graph import Projection
    raw = tiny_graph(); n = raw["n_kc"]
    graph = Projection(raw["pre"], raw["post"], raw["counts"], n, n,
                       [str(i) for i in range(n)], [str(i) for i in range(n)], {"synthetic": True})
    monkeypatch.setattr(pattern_rule, "load_projection", lambda path: graph)
    cfg = fixture_config(); write_input_catalog(tmp_path)
    audit.verify_input_catalog(cfg, tmp_path)
    arrays, rows = pattern_rule.produce(cfg, tmp_path)
    summary = pattern_rule.summarize(cfg, arrays, rows)
    checks = audit.verify_against_graph(cfg, raw, arrays, rows, summary, allow_fixture=True)
    assert checks["independently_replayed_training_actions"] == 144


def test_frozen_input_catalog_matches_exact_independent_design(tmp_path):
    path = write_input_catalog(tmp_path)
    assert audit.verify_input_catalog(fixture_config(), tmp_path) == audit.hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("kind", ["tiny_numeric_change", "wrong_label", "float_label", "missing_field", "extra_hint", "wrong_order"])
def test_frozen_input_catalog_rejects_semantic_drift(tmp_path, kind):
    path = write_input_catalog(tmp_path); document = json.loads(path.read_text())
    if kind == "tiny_numeric_change": document["catalog"]["probe_inputs"][0][0][0] += 1e-13
    elif kind == "wrong_label": document["catalog"]["probe_targets"][0][0] = 0
    elif kind == "float_label": document["catalog"]["probe_targets"] = [[float(v) for v in row] for row in document["catalog"]["probe_targets"]]
    elif kind == "missing_field": del document["catalog"]["train_catalog"]
    elif kind == "extra_hint": document["result_hint"] = True
    else: document["set_order"] = list(reversed(document["set_order"]))
    write_input_catalog(tmp_path, document)
    with pytest.raises(ValueError): audit.verify_input_catalog(fixture_config(), tmp_path)


def test_frozen_input_catalog_rejects_duplicate_json_and_wrong_path(tmp_path):
    path = write_input_catalog(tmp_path)
    path.write_text('{"study_id":"pattern_rule_v1",' + path.read_text()[1:], encoding="utf8")
    with pytest.raises(ValueError): audit.verify_input_catalog(fixture_config(), tmp_path)
    cfg = fixture_config(); cfg["input_catalog"] = "elsewhere.json"
    with pytest.raises(ValueError): audit.verify_input_catalog(cfg, tmp_path)
