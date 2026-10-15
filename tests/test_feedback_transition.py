"""Synthetic actions and feedback only; never read participant/model result files."""
from copy import deepcopy

import numpy as np
import pytest

from drosolearn.feedback_transition import human_rows, measure_stream, model_row


def row(trial, choice=0, feedback=1, **updates):
    result = {
        "stream_id": "synthetic/one", "kind": "synthetic",
        "coordinate": "synthetic_fixed_options", "source_position": trial,
        "trial": trial, "task": "synthetic_task", "phase": "training",
        "state": "option_a", "choice": choice, "feedback": feedback,
        "actual": True, "training": True,
    }
    result.update(updates)
    return result


def model_record(**updates):
    result = {
        "seed": "7001", "condition": "fixed", "history": "fresh_low",
        "phase": "phase2", "trial": "1", "left_count": "2", "right_count": "4",
        "action": "0", "reward": "1", "p_right": "0.99",
    }
    result.update(updates)
    return result


def human_arrays():
    # C is correctness, not the actual feedback: the two deliberately disagree.
    return {
        "A": np.array([[1., 2., 3., np.nan, 1., 2.]]),
        "S": np.array([[1], [1], [2], [2], [2], [1]]),
        "R": np.array([[-1, 1], [1, -1], [1, -1], [-1, 1], [1, -1], [-1, 1]]),
        "C": np.array([[1., 0., np.nan, np.nan, 0., 0.]]),
    }


def test_feedback_is_current_actual_feedback_and_last_feedback_is_unused():
    rows = [row(i + 1, c, f) for i, (c, f) in enumerate(
        [(0, 0), (0, 0), (1, 1), (1, 1), (0, 0)])]
    original = deepcopy(rows)
    summary, ledger = measure_stream(rows)
    assert summary["trials"] == 5
    assert summary["candidates"] == summary["eligible"] == 4
    assert summary["excluded"] == 0
    assert summary["reason_counts"] == {}
    for feedback in ("0", "1"):
        assert summary["by_feedback"][feedback] == {
            "denominator": 2, "stay": 1, "switch": 1,
            "stay_rate": .5, "switch_rate": .5,
        }
    assert [item["feedback"] for item in ledger] == [0, 0, 1, 1]
    assert [item["stay"] for item in ledger] == [1, 0, 1, 0]
    assert [(item["trial"], item["next_trial"]) for item in ledger] == [
        (1, 2), (2, 3), (3, 4), (4, 5)]
    assert rows == original
    rows[-1]["feedback"] = None
    assert measure_stream(rows) == (summary, ledger)


@pytest.mark.parametrize("choices,expected_stay", [([0, 0, 0], 2), ([0, 1, 0], 0)])
def test_all_stay_or_all_switch_and_zero_feedback_denominator(choices, expected_stay):
    summary, _ = measure_stream([row(i + 1, c, 0) for i, c in enumerate(choices)])
    assert summary["by_feedback"]["0"]["stay"] == expected_stay
    assert summary["by_feedback"]["0"]["switch"] == 2 - expected_stay
    assert summary["by_feedback"]["1"] == {
        "denominator": 0, "stay": 0, "switch": 0, "stay_rate": None, "switch_rate": None}


def test_single_trial_has_no_candidates_and_undefined_rates():
    summary, ledger = measure_stream([row(1)])
    assert ledger == []
    assert summary["trials"] == 1
    assert summary["candidates"] == summary["eligible"] == summary["excluded"] == 0
    assert all(cell["stay_rate"] is None for cell in summary["by_feedback"].values())


@pytest.mark.parametrize("middle_choice,middle_feedback,expected_reasons", [
    ("query", "information", {"next_query", "current_query", "current_feedback_nonbinary"}),
    (None, None, {"next_choice_missing", "current_choice_missing", "current_feedback_missing"}),
])
def test_query_or_missing_trial_remains_in_original_adjacency_ledger(
        middle_choice, middle_feedback, expected_reasons):
    rows = [row(1, 0, 0), row(2, middle_choice, middle_feedback), row(3, 0, 1), row(4, 1, 0)]
    summary, ledger = measure_stream(rows)
    assert summary["candidates"] == 3
    assert summary["eligible"] == 1
    assert summary["excluded"] == 2
    assert set(summary["reason_counts"]) == expected_reasons
    assert all(not item["eligible"] for item in ledger[:2])
    assert ledger[2]["trial"] == 3 and ledger[2]["next_trial"] == 4
    assert ledger[2]["stay"] == 0
    assert not any(item["trial"] == 1 and item["next_trial"] == 3 for item in ledger)


def test_next_feedback_missing_is_allowed_but_current_feedback_missing_is_not():
    summary, ledger = measure_stream([row(1, 0, 1), row(2, 0, None), row(3, 1, 0)])
    assert summary["eligible"] == 1
    assert ledger[0]["eligible"] == 1 and ledger[0]["stay"] == 1
    assert ledger[1]["reasons"] == "current_feedback_missing"


@pytest.mark.parametrize("field,value,reason", [
    ("task", "another_task", "task_boundary"),
    ("phase", "another_phase", "phase_boundary"),
    ("state", "option_b", "state_boundary"),
    ("actual", False, "not_actual"),
    ("training", False, "not_training"),
    ("trial", 3, "trial_gap"),
])
def test_boundary_gap_or_probe_excludes_pair_without_losing_original_positions(field, value, reason):
    rows = [row(1, source_position=7), row(2, source_position=8)]
    rows[1][field] = value
    summary, ledger = measure_stream(rows)
    assert summary["eligible"] == 0
    assert summary["reason_counts"] == {reason: 1}
    assert ledger[0]["source_position"] == 7
    assert ledger[0]["next_source_position"] == 8
    assert ledger[0]["stay"] == ""


def test_multiple_exclusion_reasons_do_not_multiply_excluded_pair_count():
    rows = [row(1, None, None, actual=False),
            row(3, "query", "information", phase="test", state="option_b", training=False)]
    summary, ledger = measure_stream(rows)
    reasons = {"current_choice_missing", "current_feedback_missing", "next_query",
               "trial_gap", "phase_boundary", "state_boundary", "not_actual", "not_training"}
    assert set(ledger[0]["reasons"].split("|")) == reasons
    assert set(summary["reason_counts"]) == reasons
    assert summary["excluded"] == 1
    assert summary["candidates"] == summary["eligible"] + summary["excluded"]
    assert sum(summary["reason_counts"].values()) > summary["excluded"]


def test_state_reversal_excludes_only_the_crossing_pair():
    rows = [row(1), row(2), row(3, state="option_b"),
            row(4, state="option_b"), row(5, state="option_b")]
    summary, ledger = measure_stream(rows)
    assert [entry["eligible"] for entry in ledger] == [1, 0, 1, 1]
    assert summary["reason_counts"] == {"state_boundary": 1}
    assert ledger[1]["trial"] == 2 and ledger[1]["next_trial"] == 3


def test_option_relabeling_preserves_transition_counts():
    rows = [row(1, 0, 1), row(2, 1, 0), row(3, 1, 1)]
    swapped = deepcopy(rows)
    for entry in swapped:
        entry["choice"] = 1 - entry["choice"]
    assert measure_stream(rows) == measure_stream(swapped)


@pytest.mark.parametrize("rows", [[], (), None])
def test_empty_or_nonlist_stream_is_rejected(rows):
    with pytest.raises(ValueError):
        measure_stream(rows)


@pytest.mark.parametrize("field,value", [
    ("choice", True), ("choice", False), ("choice", 2), ("choice", 0.),
    ("choice", float("nan")), ("choice", "unknown"),
    ("feedback", True), ("feedback", False), ("feedback", 2), ("feedback", 1.),
    ("feedback", float("nan")), ("feedback", "unknown"),
    ("source_position", True), ("source_position", 1.), ("source_position", 0),
    ("trial", True), ("trial", 1.), ("trial", 0),
    ("state", None), ("state", ""), ("state", float("nan")),
    ("actual", 1), ("training", "true"),
])
def test_invalid_canonical_codes_and_unknown_required_semantics_are_rejected(field, value):
    with pytest.raises(ValueError):
        measure_stream([row(1, **{field: value})] if field != "trial" else [row(value)])


def test_missing_state_is_not_silently_replaced_with_no_boundary():
    entry = row(1)
    del entry["state"]
    with pytest.raises(ValueError, match="Canonical fields"):
        measure_stream([entry])


@pytest.mark.parametrize("field,value", [
    ("source_position", 1), ("source_position", 0),
    ("trial", 1), ("trial", 0),
    ("stream_id", "synthetic/other"), ("kind", "model"),
    ("coordinate", "other_options"),
])
def test_duplicate_reversed_positions_and_identity_changes_fail(field, value):
    second = row(2)
    second[field] = value
    with pytest.raises(ValueError):
        measure_stream([row(1), second])


@pytest.mark.parametrize("field", ["trial", "source_position"])
def test_positive_but_reversed_original_positions_are_rejected(field):
    first, following = row(2), row(3)
    following[field] = 1
    with pytest.raises(ValueError, match="Repeated or reversed"):
        measure_stream([first, following])


@pytest.mark.parametrize("left,right,action,reward,choice", [
    (2, 4, 0, 1, 0), (2, 4, 1, 0, 1),
    (4, 2, 0, 0, 1), (4, 2, 1, 1, 0),
])
def test_model_actual_side_and_counts_recover_quantity_choice(left, right, action, reward, choice):
    record = model_record(left_count=str(left), right_count=str(right),
                          action=str(action), reward=str(reward))
    original = record.copy()
    result = model_row(record, source_position=19)
    assert result["choice"] == choice
    assert result["feedback"] == reward
    assert result["coordinate"] == "smaller_larger_quantity_category"
    assert result["state"] == "smaller"
    assert result["trial"] == 1 and result["source_position"] == 19
    assert record == original


def test_model_probability_is_never_thresholded_into_an_observed_choice():
    result = model_row(model_record(action="0", reward="1", p_right="1"), source_position=1)
    assert result["choice"] == 0
    assert result["feedback"] == 1
    assert result["actual"] is True


@pytest.mark.parametrize("second_action,second_reward,expected_stay", [(1, 1, 1), (0, 0, 0)])
def test_same_quantity_and_same_screen_side_are_different_transitions(
        second_action, second_reward, expected_stay):
    first = model_row(model_record(action="0", reward="1"), source_position=1)
    following = model_row(model_record(trial="2", left_count="4", right_count="2",
                                      action=str(second_action), reward=str(second_reward)), source_position=2)
    summary, ledger = measure_stream([first, following])
    assert summary["by_feedback"]["1"]["denominator"] == 1
    assert ledger[0]["stay"] == expected_stay


@pytest.mark.parametrize("field,value", [
    ("phase", "phase1"), ("action", "2"), ("action", True), ("action", "0.0"),
    ("reward", "2"), ("reward", False), ("reward", "1.0"),
    ("left_count", "3"), ("left_count", "4"), ("right_count", "8"),
    ("trial", "0"), ("trial", " 1"), ("trial", "1.0"),
    ("p_right", "nan"), ("p_right", "inf"), ("p_right", "-inf"),
    ("p_right", "-0.01"), ("p_right", "1.01"), ("p_right", "unknown"),
])
def test_model_unknown_codes_nonfinite_policy_and_out_of_scope_inputs_fail(field, value):
    with pytest.raises(ValueError):
        model_row(model_record(**{field: value}), source_position=1)


def test_model_reward_must_match_actual_quantity_action_not_policy():
    with pytest.raises(ValueError, match="reward"):
        model_row(model_record(action="0", reward="0", p_right=".9"), source_position=1)


def test_human_adapter_preserves_query_missing_state_and_actual_not_correctness_feedback():
    arrays = human_arrays()
    original = {key: value.copy() for key, value in arrays.items()}
    rows = human_rows(arrays, stream_id="synthetic/human", trials=6)
    assert [entry["choice"] for entry in rows] == [0, 1, "query", None, 0, 1]
    assert [entry["feedback"] for entry in rows] == [0, 0, "information", None, 1, 1]
    assert [entry["state"] for entry in rows] == ["1", "1", "2", "2", "2", "1"]
    assert rows[0]["feedback"] != int(arrays["C"][0, 0])
    assert rows[4]["feedback"] != int(arrays["C"][0, 4])
    assert [entry["trial"] for entry in rows] == list(range(1, 7))
    assert [entry["source_position"] for entry in rows] == list(range(1, 7))
    assert all(entry["coordinate"] == "source_coded_card_identity" for entry in rows)
    assert all(entry["phase"] == "initial" for entry in rows)
    summary, ledger = measure_stream(rows)
    assert summary["eligible"] == 1
    assert ledger[0]["feedback"] == 0 and ledger[0]["stay"] == 0
    assert "state_boundary" in ledger[-1]["reasons"].split("|")
    for key in arrays:
        np.testing.assert_equal(arrays[key], original[key])


@pytest.mark.parametrize("trials,boundary", [(6, 3), (1000, 800)])
def test_human_regime_boundary_is_separate_from_hidden_state_reversal(trials, boundary):
    arrays = {
        "A": np.ones((1, trials)), "S": np.ones((trials, 1), dtype=int),
        "R": np.ones((trials, 2), dtype=int), "C": np.ones((1, trials)),
    }
    rows = human_rows(arrays, stream_id="synthetic/regime", trials=trials,
                      regime_boundary_after_trial=boundary)
    assert rows[boundary - 1]["trial"] == boundary
    assert rows[boundary - 1]["phase"] == "initial"
    assert rows[boundary]["trial"] == boundary + 1
    assert rows[boundary]["phase"] == "transfer"
    assert len({entry["state"] for entry in rows}) == 1
    summary, ledger = measure_stream(rows)
    assert summary["candidates"] == trials - 1
    assert summary["eligible"] == trials - 2
    assert summary["reason_counts"] == {"phase_boundary": 1}
    assert ledger[boundary - 1]["reasons"] == "phase_boundary"
    assert ledger[boundary - 1]["eligible"] == 0
    assert all(entry["eligible"] == 1 for index, entry in enumerate(ledger) if index != boundary - 1)


@pytest.mark.parametrize("boundary", [0, -1, True, 3., None])
def test_human_unknown_or_invalid_regime_boundary_is_rejected(boundary):
    with pytest.raises(ValueError, match="boundary"):
        human_rows(human_arrays(), stream_id="synthetic/regime", trials=6,
                   regime_boundary_after_trial=boundary)


@pytest.mark.parametrize("key,index,value", [
    ("A", (0, 0), 4), ("A", (0, 0), np.inf),
    ("S", (0, 0), 3), ("S", (0, 0), np.nan),
    ("R", (0, 0), 0), ("R", (0, 0), np.nan),
    ("C", (0, 0), 0), ("C", (0, 0), np.nan),
    ("C", (0, 2), 1), ("C", (0, 3), 0),
])
def test_human_unknown_codes_missing_state_and_c_structure_fail(key, index, value):
    arrays = human_arrays()
    arrays[key] = arrays[key].astype(float)
    arrays[key][index] = value
    with pytest.raises(ValueError):
        human_rows(arrays, stream_id="synthetic/human", trials=6)


@pytest.mark.parametrize("key", ["A", "R", "S", "C"])
@pytest.mark.parametrize("dtype", [bool, complex, object, str])
def test_human_nonreal_or_boolean_source_arrays_fail(key, dtype):
    arrays = human_arrays()
    arrays[key] = arrays[key].astype(dtype)
    with pytest.raises(ValueError, match="schema"):
        human_rows(arrays, stream_id="synthetic/human", trials=6)


@pytest.mark.parametrize("key", ["A", "R", "S", "C"])
def test_human_wrong_shape_fails(key):
    arrays = human_arrays()
    arrays[key] = arrays[key].reshape(-1)
    with pytest.raises(ValueError, match="schema"):
        human_rows(arrays, stream_id="synthetic/human", trials=6)


def test_human_missing_required_state_array_is_an_error():
    arrays = human_arrays()
    del arrays["S"]
    with pytest.raises(KeyError):
        human_rows(arrays, stream_id="synthetic/human", trials=6)
