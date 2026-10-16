"""Handwritten choice/state/reward records; no learning or random draws."""
from copy import deepcopy
import ast
import inspect
import json
from pathlib import Path

import pytest

from drosolearn import conditional_transition as study
from drosolearn.feedback_transition import measure_stream


REPO = Path(__file__).resolve().parents[1]


def synthetic_config(seeds=(17, 18)):
    cfg = json.loads((REPO / "configs/conditional_transition_v1.json").read_text(encoding="utf8"))
    cfg.update(scope="integration_test")
    cfg["human"].update(record_count=2, trials=8, regime_boundary_after_trial=4)
    cfg["model"].update(seeds=list(seeds), trials_per_phase=4)
    models = len(seeds) * 8 * 3
    cfg.update(registered_streams=2 + models, registered_input_trials=16 + models * 8,
               registered_candidate_pairs=14 + models * 7, registered_strata=16 + models * 4)
    return cfg


def canonical(trial, choice=0, feedback=1, kind="human", **changes):
    row = {"stream_id": "invented/human", "kind": kind, "coordinate": "source_coded_card_identity",
           "source_position": trial, "trial": trial, "local_trial": trial, "task": "markovic_card",
           "phase": "initial", "state": "1", "choice": choice, "feedback": feedback, "actual": True, "training": True}
    row.update(changes)
    return row


def model_record(**changes):
    row = {key: "0" for key in study.MODEL_FIELDS}
    row.update(stage="network", seed="17", condition="linear", environment="probabilistic", phase="phase1", trial="1", global_trial="1",
               family_id="handcrafted:0", left_count="2", right_count="4", chosen_count="4", action="1", reward="0", p_right="0.6",
               q_reward="0.8", action_uniform="0.1", reward_uniform="0.9", head_update_called="1", head_updates="1")
    row.update(changes)
    return row


def adapt(record, position=2):
    return study.adapt_model_record(record, seed=17, source_position=position, trials_per_phase=4)


def cell(summary, phase="initial", state="1", choice=0):
    return next(row for row in summary["strata"] if (row["phase"], row["state"], row["choice"]) == (phase, state, choice))


def test_registered_config_validation_does_not_read_real_data(monkeypatch):
    cfg = json.loads((REPO / "configs/conditional_transition_v1.json").read_text(encoding="utf8"))
    before = deepcopy(cfg)
    monkeypatch.setattr(study, "input_inventory", lambda *a, **k: pytest.fail("Validation read actual source data"))
    study.validate_config(cfg)
    study.validate_config(synthetic_config())
    assert cfg == before


@pytest.mark.parametrize("field,value", [
    ("scope", "retuned"), ("learning_updates", 1), ("learner_constructions", 1), ("dataset_generations", 1), ("rng_draws", 1),
    ("pooling", True), ("strata_averaging", True), ("causal_inference", True), ("registered_strata", 1),
    ("registered_input_trials", 1), ("zero_denominator", 0),
])
def test_unsupported_inference_or_changed_denominator_is_rejected(field, value):
    cfg = synthetic_config()
    cfg[field] = value
    with pytest.raises(ValueError):
        study.validate_config(cfg)


def test_predefined_human8_model4_design_keeps_possible_but_unobserved_cells():
    assert len(study.predefined_strata("human")) == 8
    design = study.predefined_strata("model")
    assert len(design) == 4
    assert {(row["phase"], row["state"]) for row in design} == {("phase1", "larger"), ("phase2", "smaller")}
    summary, _ = study.measure_conditional([canonical(1), canonical(2)])
    assert len(summary["strata"]) == 8
    empty = cell(summary, "transfer", "2", 1)
    assert empty["support_status"] == "neither" and empty["common_support"] is False
    assert empty["stay_rate_difference_1_minus_0"] is None
    assert all(v["denominator"] == 0 and v["stay_rate"] is None for v in empty["by_feedback"].values())


def test_original_whole_stream_metric_is_unchanged_and_integer_counts_partition_it():
    rows = [canonical(i + 1, choice, feedback) for i, (choice, feedback) in enumerate(
        [(0, 1), (0, 0), (1, 1), (1, 0), (0, 0), (0, 1), (1, 1), (0, 0)])]
    original, _ = measure_stream(rows)
    result, ledger = study.measure_conditional(rows)
    assert {key: result[key] for key in study.BASE_SUMMARY_FIELDS} == original
    first, second = cell(result, choice=0), cell(result, choice=1)
    assert first["by_feedback"]["1"]["denominator"] == 2
    assert first["by_feedback"]["0"]["denominator"] == 2
    assert first["stay_rate_difference_1_minus_0"] == 0.
    assert second["stay_rate_difference_1_minus_0"] == .5
    for feedback in ("0", "1"):
        for key in ("denominator", "stay", "switch"):
            assert sum(row["by_feedback"][feedback][key] for row in result["strata"]) == original["by_feedback"][feedback][key]
    assert len(ledger) == result["candidates"] == 7


def test_marginal_feedback_difference_cannot_create_conditional_common_support():
    actions = [0, 0, 0, 1, 0, 0, 1, 0]
    rows = [canonical(i + 1, action, 1 - action) for i, action in enumerate(actions)]
    summary, _ = study.measure_conditional(rows)
    assert summary["by_feedback"]["1"]["stay_rate"] != summary["by_feedback"]["0"]["stay_rate"]
    assert cell(summary, choice=0)["support_status"] == "feedback_1_only"
    assert cell(summary, choice=1)["support_status"] == "feedback_0_only"
    assert all(row["stay_rate_difference_1_minus_0"] is None for row in summary["strata"])


def test_query_and_missing_trials_are_not_removed_to_join_their_neighbors():
    rows = [canonical(1), canonical(2, "query", "information"), canonical(3, None, None), canonical(4, 1, 0)]
    summary, ledger = study.measure_conditional(rows)
    assert summary["candidates"] == summary["excluded"] == 3 and summary["eligible"] == 0
    assert ledger[0]["reasons"] == "next_query"
    assert ledger[1]["reasons"] == "current_feedback_nonbinary|current_query|next_choice_missing"
    assert ledger[2]["reasons"] == "current_choice_missing|current_feedback_missing"
    assert ledger[0]["next_choice"] == "query" and ledger[1]["next_choice"] == ""
    assert sum(row["by_feedback"]["0"]["denominator"] + row["by_feedback"]["1"]["denominator"] for row in summary["strata"]) == 0


def test_overlapping_phase_state_boundaries_exclude_one_pair_and_preserve_native_local_trials():
    rows = [canonical(4, 1, 1, kind="model", phase="phase1", state="larger", source_position=12, local_trial=4),
            canonical(5, 0, 0, kind="model", phase="phase2", state="smaller", source_position=13, local_trial=1)]
    summary, ledger = study.measure_conditional(rows)
    assert summary["excluded"] == 1 and summary["reason_counts"] == {"phase_boundary": 1, "state_boundary": 1}
    assert ledger[0]["trial"] == 4 and ledger[0]["next_trial"] == 5
    assert ledger[0]["local_trial"] == 4 and ledger[0]["next_local_trial"] == 1
    assert ledger[0]["reasons"] == "phase_boundary|state_boundary"


def test_missing_next_feedback_is_allowed_by_the_original_current_feedback_definition():
    result, _ = study.measure_conditional([canonical(1, 0, 1), canonical(2, 0, None)])
    assert cell(result)["by_feedback"]["1"]["denominator"] == 1
    assert cell(result)["by_feedback"]["1"]["stay_rate"] == 1.


def test_trial_gap_and_nonactual_exclusions_are_preserved_with_their_overlap():
    result, ledger = study.measure_conditional([canonical(1, actual=False), canonical(3)])
    assert ledger[0]["reasons"] == "not_actual|trial_gap"
    assert result["excluded"] == 1 and result["eligible"] == 0


def test_nondeterministic_reward_after_wrong_designated_choice_is_a_valid_observed_outcome():
    row = adapt(model_record(phase="phase2", trial="1", global_trial="5", q_reward="0.2", reward="1", reward_uniform="0.1"))
    assert row["choice"] == row["feedback"] == 1
    assert row["state"] == "smaller" and row["trial"] == 5 and row["local_trial"] == 1


def test_screen_side_changes_do_not_change_absolute_quantity_choice_identity():
    first = adapt(model_record(left_count="2", right_count="4", chosen_count="2", action="0", p_right="0.4", action_uniform="0.9",
                               q_reward="0.2", reward="0"))
    second = adapt(model_record(trial="2", global_trial="2", left_count="4", right_count="2", chosen_count="2", action="1", p_right="0.6",
                                action_uniform="0.1", q_reward="0.2", reward="0"), position=3)
    summary, ledger = study.measure_conditional([first, second])
    assert first["choice"] == second["choice"] == 0
    assert summary["by_feedback"]["0"]["stay"] == 1 and ledger[0]["stay"] == 1


@pytest.mark.parametrize("changes", [
    {"global_trial": "2"}, {"trial": "0"}, {"phase": "phase3"}, {"seed": "18"},
    {"action": "0"}, {"reward": "1"}, {"q_reward": "0.2"}, {"p_right": "nan"},
    {"action_uniform": "1.0"}, {"reward_uniform": "-0.1"}, {"chosen_count": "2"},
    {"condition": "expected_linear"}, {"left_count": "3"}, {"stage": "linear"},
])
def test_model_adapter_rejects_semantically_inconsistent_actual_logs(changes):
    with pytest.raises(ValueError):
        adapt(model_record(**changes))


def test_uninformative_state_is_the_fixed_phase_reference_and_not_inferred_from_reward():
    first = adapt(model_record(environment="uninformative", q_reward="0.5", reward="0"))
    second = adapt(model_record(environment="uninformative", q_reward="0.5", reward="1", reward_uniform="0.1"))
    assert first["state"] == second["state"] == "larger"
    assert first["feedback"] != second["feedback"]


def test_measurement_does_not_mutate_canonical_records_or_create_models_or_rngs():
    rows = [canonical(1), canonical(2, 1, 0)]
    before = deepcopy(rows)
    study.measure_conditional(rows)
    assert rows == before
    calls = []
    for node in ast.walk(ast.parse(inspect.getsource(study))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.append(node.func.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.append(node.func.attr)
    assert not set(calls) & {"Learner", "LinearLearner", "NormalizedLearner", "make_dataset", "default_rng", "selected_update"}


def test_exact_comparison_distinguishes_boolean_counts_and_json_rejects_duplicate_keys(tmp_path):
    assert study.exact({"count": True}, {"count": 1}) is False
    path = tmp_path / "handcrafted.json"
    for payload in ('{"a":1,"a":2}', '{"p":NaN}'):
        path.write_text(payload, encoding="utf8")
        with pytest.raises(ValueError):
            study.read_json(path)
