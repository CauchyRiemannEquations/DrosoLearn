"""Small invented contextual curricula; actual registered seeds/budgets are not run."""
import copy
import json

import numpy as np
import pytest

import drosolearn.interference_order as order
from test_human_shell_models import small_graph


def fixture_config():
    cfg = copy.deepcopy(order.SCIENCE)
    cfg.update(scope="integration_test", model_seeds=[91001, 91002], stage_trials=6, joint_trials=12,
               checkpoint_every=6, joint_checkpoint_every=12, bootstrap_repeats=16,
               bootstrap_entropy=[202610192, 99001, 2], population_pass_count=2, graph="synthetic-unused")
    return cfg


def write_fixture_catalog(root):
    path = root / order.SCIENCE["input_catalog"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"study_id": "interference_order_v1", "set_order": list(order.SET_ORDER),
                                "catalog": {key: value.tolist() for key, value in order.catalog().items()}}), encoding="utf8")
    return path


def small_production(monkeypatch, root):
    cfg = fixture_config()
    write_fixture_catalog(root)
    monkeypatch.setattr(order, "load_projection", lambda unused: small_graph())
    arrays, rows = order.produce(cfg, root)
    return cfg, arrays, rows


def test_validation_is_rng_free_and_actual_budget_is_fixed(monkeypatch):
    monkeypatch.setattr(np.random, "default_rng", lambda *a, **k: (_ for _ in ()).throw(AssertionError("RNG draw")))
    order.validate_config(copy.deepcopy(order.SCIENCE))
    order.validate_config(fixture_config())


@pytest.mark.parametrize("key,value", [
    ("stage_trials", 720), ("joint_trials", 1440), ("head_lr", .3),
    ("context_q_alpha", .3), ("conflict_threshold", .5), ("joint_threshold", .7),
    ("model_seeds", [15201]), ("population_pass_count", 1),
    ("joint_order", "chosen_after_results"), ("idle", "decay"), ("order_margin", .01),
    ("bootstrap_repeats", 10000),
    ("graph", "unregistered-other-graph"), ("graph_files", []), ("input_files", []),
])
def test_scientific_criteria_cannot_be_relaxed(key, value):
    cfg = copy.deepcopy(order.SCIENCE)
    cfg[key] = value
    with pytest.raises(ValueError, match="Fixed order science"):
        order.validate_config(cfg)


@pytest.mark.parametrize("change", [
    {"model_seeds": [15201]}, {"model_seeds": [91001,91001]}, {"stage_trials": 360},
    {"stage_trials": 7, "joint_trials": 14}, {"bootstrap_entropy": [202610192,15301,2]},
    {"population_pass_count": 1},
])
def test_integration_scope_cannot_run_the_scientific_budget_or_seeds(change):
    cfg = fixture_config()
    cfg.update(change)
    with pytest.raises(ValueError):
        order.validate_config(cfg)


def test_twelve_contextual_inputs_have_a_joint_solution_and_two_conflicts():
    data = order.catalog()
    x, y = data["probe_inputs"], data["probe_targets"]
    assert x.shape == (2,6,7) and y.shape == (2,6)
    np.testing.assert_array_equal(x[0,:,6], -1.)
    np.testing.assert_array_equal(x[1,:,6], 1.)
    assert len({tuple(v) for v in x.reshape(12,7)}) == 12
    np.testing.assert_array_equal(y[0] != y[1], data["conflicting_ids"])
    assert y[0].sum() == y[1].sum() == 3
    oracle = {tuple(v): int(t) for v,t in zip(x.reshape(12,7), y.ravel())}
    assert all(oracle[tuple(v)] == t for v,t in zip(x.reshape(12,7), y.ravel()))
    # Dropping context makes exactly the two declared IDs ambiguous.
    assert len({tuple(v[:-1]) for v in x.reshape(12,7)}) == 6


def test_input_catalog_file_is_consumed_and_dtype_or_hint_changes_fail(tmp_path):
    path = write_fixture_catalog(tmp_path)
    cfg = fixture_config()
    actual = order.load_catalog(cfg, tmp_path)
    np.testing.assert_array_equal(actual["probe_targets"], order.catalog()["probe_targets"])
    raw = json.loads(path.read_bytes())
    raw["catalog"]["conflicting_ids"] = [1,1,0,0,0,0]
    path.write_text(json.dumps(raw), encoding="utf8")
    with pytest.raises(ValueError, match="Contextual input differs"):
        order.load_catalog(cfg, tmp_path)


def test_missing_catalog_is_not_regenerated(tmp_path):
    with pytest.raises(FileNotFoundError):
        order.load_catalog(fixture_config(), tmp_path)


def test_scientific_counts_and_checkpoint_mapping_without_rng_draws():
    cfg = order.SCIENCE
    meta = order.checkpoint_metadata(cfg)
    assert len(meta["checkpoint_stream"]) == 43
    np.testing.assert_array_equal(meta["stream_end_checkpoint"], [6,12,18,24,30,36,42])
    np.testing.assert_array_equal(meta["stream_start_checkpoint"], [0,0,6,6,12,12,0])
    assert 9*10*(6*cfg["stage_trials"]+cfg["joint_trials"]) == 259200
    assert 9*10*43*12 == 46440


def test_small_production_preserves_prefixes_and_original_event_counts(monkeypatch, tmp_path):
    cfg, arrays, rows = small_production(monkeypatch, tmp_path)
    summary = order.summarize(cfg, arrays, rows)
    assert len(rows) == 9*2*(6*6+12) == 864
    assert arrays["checkpoint_probabilities"].shape == (9,2,8,2,6)
    assert summary["counts"]["static_probe_cases"] == 9*2*8*12
    assert summary["counts"]["head_update_calls"] == 7*2*48
    assert summary["counts"]["internal_update_calls"] == 3*2*48
    assert summary["counts"]["value_update_calls"] == 2*48
    assert summary["counts"]["idle_feedback_trials"] == 0
    assert summary["counts"]["idle_new_probability_evaluations"] == 0
    assert summary["max_decomposition_residual"] <= 1e-12
    for kind in ("head","internal","value"):
        np.testing.assert_array_equal(arrays[f"stream_start_{kind}_calls"],
                                      arrays[f"checkpoint_{kind}_calls"][:,:,arrays["stream_start_checkpoint"]])
    for ci, condition in enumerate(order.CONDITIONS):
        prefix = arrays["stream_end_checkpoint"][:2]
        for field in ("head","weights","value_table","encoder","probe_kc","probe_mbon","probe_decision"):
            np.testing.assert_array_equal(arrays[f"idle_w_{condition}_{field}"], arrays[f"w_{condition}_{field}"][:,prefix])
        np.testing.assert_array_equal(arrays["idle_probabilities"][ci], arrays["checkpoint_probabilities"][ci,:,prefix].transpose(1,0,2,3))
    for row in rows:
        ci, si, stream, ti = order.CONDITIONS.index(row["condition"]), cfg["model_seeds"].index(row["seed"]), order.STREAMS.index(row["stream"]), row["local_trial"]-1
        assert row["action"] == int(arrays["stream_uniforms"][si,stream,ti] < row["probability_go"])
        target = int(arrays["probe_targets"][order.SET_ORDER.index(row["task"]),row["stimulus_index"]])
        assert row["reward"] == ((1 if target else -1) if row["action"] else 0)
        assert row["head_step_l2"] == arrays["trial_head_steps"][ci,si,stream,ti]
        assert row["total_history_trial"] == row["local_trial"]+(6 if stream in (2,3,4,5) else 0)
    # Invalid padded slots are explicitly masked and never become fake trials.
    assert arrays["trial_valid"].sum() == 48
    assert np.all(arrays["trial_actions"][:,:,~arrays["trial_valid"]] == -1)


def test_paired_task_schedules_and_uniforms_match_between_orders(monkeypatch, tmp_path):
    cfg, arrays, rows = small_production(monkeypatch, tmp_path)
    for si in range(2):
        for task in range(2):
            for repetition in range(2):
                assert sorted(arrays["schedules"][si,task,repetition]) == list(range(6))
        # A0 in prefix_A is the same task tape later in BA; B0 likewise in AB.
        np.testing.assert_array_equal(arrays["stream_stimulus_indices"][si,0,:6], arrays["stream_stimulus_indices"][si,4,:6])
        np.testing.assert_array_equal(arrays["stream_uniforms"][si,0,:6], arrays["stream_uniforms"][si,4,:6])
        np.testing.assert_array_equal(arrays["stream_stimulus_indices"][si,1,:6], arrays["stream_stimulus_indices"][si,2,:6])
        np.testing.assert_array_equal(arrays["stream_stimulus_indices"][si,6,::2], arrays["stream_stimulus_indices"][si,0,:6])
        np.testing.assert_array_equal(arrays["stream_stimulus_indices"][si,6,1::2], arrays["stream_stimulus_indices"][si,1,:6])
    # Disjoint contextual table rows commute under the same per-task tapes.
    # This is an analytical invariant of the control, independent of learning success.
    values = arrays["w_context_q_value_table"]
    ends = arrays["stream_end_checkpoint"]
    np.testing.assert_array_equal(values[:,ends[2]], values[:,ends[4]])
    np.testing.assert_array_equal(values[:,ends[2]], values[:,ends[6]])
    q = next(f for f in order.summarize(cfg, arrays, rows)["families"] if f["condition"] == "context_q")
    assert all(task["retention_loss"] == 0. for seed in q["seeds"] for task in seed["tasks"])


def test_multiple_checkpoints_per_stage_use_actual_prefix_boundaries(monkeypatch, tmp_path):
    cfg = fixture_config()
    cfg.update(model_seeds=[91003], stage_trials=12, joint_trials=24, population_pass_count=1)
    write_fixture_catalog(tmp_path)
    monkeypatch.setattr(order, "load_projection", lambda unused: small_graph())
    arrays, rows = order.produce(cfg, tmp_path)
    summary = order.summarize(cfg, arrays, rows)
    assert len(arrays["checkpoint_stream"]) == 15
    np.testing.assert_array_equal(arrays["stream_end_checkpoint"], [2,4,6,8,10,12,14])
    assert len(rows) == 9*96
    assert summary["counts"]["static_probe_cases"] == 9*15*12
    assert all(len(task["new_task_curve"]) == 3 for f in summary["families"] for s in f["seeds"] for task in s["tasks"])


def test_context_blind_five_sixths_does_not_qualify(monkeypatch, tmp_path):
    cfg, arrays, rows = small_production(monkeypatch, tmp_path)
    y = arrays["probe_targets"]
    correct = np.full(arrays["checkpoint_probabilities"].shape, 1.)
    correct[:,:,:,:, :2] = .5
    correct[:,:,0] = .5
    arrays["checkpoint_probabilities"] = np.where(y[None,None,None] == 1, correct, 1-correct)
    summary = order.summarize(cfg, arrays, rows)
    assert summary["analytic_context_blind_witness"]["overall_accuracy"] == 5/6
    assert summary["qualification_supported"] is False
    assert summary["H_interference"]["status"] == summary["H_order"]["status"] == "not_assessable"
    assert all(not seed["qualified"] for family in summary["families"] for seed in family["seeds"])


def test_qualified_case_keeps_loss_practice_cost_and_order_distinct(monkeypatch, tmp_path):
    cfg, arrays, rows = small_production(monkeypatch, tmp_path)
    y = arrays["probe_targets"]
    accuracy = np.full(arrays["checkpoint_probabilities"].shape, .9)
    accuracy[:,:,0] = .5
    ends = arrays["stream_end_checkpoint"]
    accuracy[:,:,ends[2],0] = .6  # AB loses A by .3 after having acquired it.
    accuracy[:,:,ends[4],1] = .8  # BA loses B by .1.
    accuracy[:,:,ends[3],0] = .95 # Additional A practice improves by .05.
    arrays["checkpoint_probabilities"] = np.where(y[None,None,None] == 1, accuracy, 1-accuracy)
    summary = order.summarize(cfg, arrays, rows)
    assert summary["qualification_supported"] is True
    assert summary["H_interference"]["supported"] is True
    first = summary["families"][0]["seeds"][0]
    assert first["tasks"][0]["retention_loss"] == pytest.approx(.3)
    assert first["tasks"][0]["continued_practice_gain"] == pytest.approx(.05)
    assert first["tasks"][0]["matched_budget_terminal_cost"] == pytest.approx(.35)
    assert first["order_delta"] == pytest.approx(-.1)
    assert summary["H_order"]["supported"] is True
    assert summary["max_decomposition_residual"] <= 1e-12
