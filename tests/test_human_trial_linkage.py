"""Invented-record tests for exact linkage and feedback-block diagnostics only."""
from copy import deepcopy

import pytest

from drosolearn.human_trial_linkage import feedback_blocks, link_sequences


def pair(reward=(0, 1), punishment=(1, 0)):
    return {"REW": list(reward), "PUN": list(punishment)}


def events(*markers, rows=None):
    raw = [{"onset": str(index), "duration": "0.1", "sample": str(index * 1000), "value": value}
           for index, value in enumerate(markers)]
    if rows is not None:
        for item, row in zip(raw, rows, strict=True):
            item["row"] = row
    return raw


def test_bidirectional_exact_matches_cover_complete_candidate_matrix():
    report = link_sequences({"s-b": pair((1, 0), (0, 1)), "s-a": pair()},
                            {"row-2": pair((1, 0), (0, 1)), "row-1": pair()})
    assert report["accepted"] == [{"event_label": "s-a", "mat_row": "row-1"},
                                   {"event_label": "s-b", "mat_row": "row-2"}]
    assert len(report["edges"]) == 4
    assert report["event_candidates"] == {"s-a": ["row-1"], "s-b": ["row-2"]}
    assert report["mat_candidates"] == {"row-1": ["s-a"], "row-2": ["s-b"]}
    assert sum(edge["both_equal"] for edge in report["edges"]) == 2


def test_both_conditions_must_match_the_same_row():
    report = link_sequences({"s-a": pair()},
                            {"row-1": pair((0, 1), (0, 1)), "row-2": pair((1, 0), (1, 0))})
    assert report["accepted"] == []
    assert report["event_candidates"]["s-a"] == []
    assert any(edge["REW_equal"] for edge in report["edges"])
    assert any(edge["PUN_equal"] for edge in report["edges"])
    assert not any(edge["both_equal"] for edge in report["edges"])


@pytest.mark.parametrize("different", [pair((0,), (1, 0)), pair((0, 1), (1,)),
                                         pair((0, 0), (1, 0)), pair((0, 1), (1, 1))])
def test_length_or_single_bit_difference_is_not_edited_away(different):
    report = link_sequences({"s-a": pair()}, {"row-1": different})
    assert report["accepted"] == []
    assert report["event_candidates"] == {"s-a": []}
    assert report["mat_candidates"] == {"row-1": []}


def test_duplicate_mat_rows_remain_ambiguous():
    report = link_sequences({"s-a": pair()}, {"row-1": pair(), "row-2": pair()})
    assert report["accepted"] == []
    assert report["event_candidates"] == {"s-a": ["row-1", "row-2"]}
    assert report["mat_candidates"] == {"row-1": ["s-a"], "row-2": ["s-a"]}


def test_duplicate_event_files_remain_ambiguous():
    report = link_sequences({"s-a": pair(), "s-b": pair()}, {"row-1": pair()})
    assert report["accepted"] == []
    assert report["event_candidates"] == {"s-a": ["row-1"], "s-b": ["row-1"]}
    assert report["mat_candidates"] == {"row-1": ["s-a", "s-b"]}


def test_duplicates_on_both_sides_are_not_deduplicated_or_greedily_assigned():
    report = link_sequences({"s-a": pair(), "s-b": pair()}, {"row-1": pair(), "row-2": pair()})
    assert len(report["edges"]) == 4
    assert sum(edge["both_equal"] for edge in report["edges"]) == 4
    assert report["accepted"] == []


def test_unmatched_and_ambiguous_records_remain_alongside_unique_candidate():
    report = link_sequences({"unique": pair((1,), (1,)), "duplicate-a": pair(),
                             "duplicate-b": pair(), "unmatched": pair((0,), (0,))},
                            {"unique-row": pair((1,), (1,)), "ambiguous-row": pair(),
                             "unmatched-row": pair((1, 1), (1, 1))})
    assert report["accepted"] == [{"event_label": "unique", "mat_row": "unique-row"}]
    assert len(report["event_candidates"]) == 4
    assert len(report["mat_candidates"]) == 3
    assert report["event_candidates"]["unmatched"] == []
    assert report["mat_candidates"]["unmatched-row"] == []


def test_output_does_not_depend_on_input_dictionary_insertion_order():
    event = {"s-z": pair((1,), (1,)), "s-a": pair()}
    mat = {"row-z": pair(), "row-a": pair((1,), (1,))}
    assert link_sequences(event, mat) == link_sequences(dict(reversed(list(event.items()))),
                                                       dict(reversed(list(mat.items()))))


def test_linkage_does_not_change_input_sequences_or_create_behavioral_estimates():
    event, mat = {"s-a": pair()}, {"row-1": pair()}
    original = deepcopy((event, mat))
    report = link_sequences(event, mat)
    assert (event, mat) == original
    assert not {"accuracy", "learning_curve", "effect_size", "participant_identity_verified"} & set(report)


@pytest.mark.parametrize("bad", [[], [0, 2], [0, -1], [0, None], [0, "1"], [0, True], [0, 1.0]])
@pytest.mark.parametrize("side", ["events", "mats"])
def test_sequences_require_nonempty_strict_binary_integer_values(bad, side):
    event, mat = {"s-a": pair()}, {"row-1": pair()}
    target = event if side == "events" else mat
    next(iter(target.values()))["REW"] = bad
    with pytest.raises(ValueError):
        link_sequences(event, mat)


@pytest.mark.parametrize("bad", [{"REW": [0]}, {"REW": [0], "PUN": [1], "extra": [0]}])
def test_condition_names_are_exact_not_inferred(bad):
    with pytest.raises(ValueError):
        link_sequences({"s-a": bad}, {"row-1": pair()})


def test_clean_block_records_original_rows_and_fixed_numeric_diagnostics():
    report = feedback_blocks(events("S100", "S51", "S53", "S61", "S101"))
    assert report["event_count"] == 5
    assert report["tail_event_rows"] == [6]
    assert len(report["blocks"]) == 1
    block = report["blocks"][0]
    assert block["task"] == "REW"
    assert block["feedback_ordinal"] == 1
    assert block["feedback_event_row"] == 5
    assert block["feedback_flag"] == 1
    assert block["choice_event_rows"] == [3]
    assert block["expectation_event_rows"] == [4]
    assert block["choice_side_code"] == 1
    assert block["expectation_mat_code"] == 3
    assert block["strict_comparable"] is True
    assert block["issues"] == []


@pytest.mark.parametrize("choice,expectation,feedback,start,end,side,level,flag", [
    ("S51", "S53", "S60", "S100", "S101", 1, 3, 0),
    ("S52", "S56", "S61", "S100", "S101", 2, 6, 1),
    ("S71", "S73", "S80", "S200", "S201", 1, 3, 0),
    ("S72", "S76", "S81", "S200", "S201", 2, 6, 1),
])
def test_block_action_expectation_and_feedback_codes_are_not_fitted(choice, expectation, feedback, start, end, side, level, flag):
    block = feedback_blocks(events(start, choice, expectation, feedback, end))["blocks"][0]
    assert block["choice_side_code"] == side
    assert block["expectation_mat_code"] == level
    assert block["feedback_flag"] == flag
    assert block["strict_comparable"] is True


def test_feedback_ordinals_are_condition_specific_while_blocks_are_global():
    report = feedback_blocks(events("S100", "S51", "S53", "S61", "S52", "S54", "S60", "S101",
                                    "S200", "S72", "S76", "S81", "S201"))
    assert [(block["task"], block["feedback_ordinal"]) for block in report["blocks"]] == [
        ("REW", 1), ("REW", 2), ("PUN", 1)]
    assert [block["choice_event_rows"] for block in report["blocks"]] == [[3], [6], [11]]
    assert all(block["strict_comparable"] for block in report["blocks"])
    assert report["tail_event_rows"] == [14]


def test_explicit_original_row_numbers_are_not_replaced_by_contiguous_trial_numbers():
    report = feedback_blocks(events("S100", "S52", "S55", "S61", "S101", rows=[2, 8, 11, 17, 20]))
    block = report["blocks"][0]
    assert block["choice_event_rows"] == [8]
    assert block["expectation_event_rows"] == [11]
    assert block["feedback_event_row"] == 17
    assert report["tail_event_rows"] == [20]


@pytest.mark.parametrize("markers", [
    ("S100", "S53", "S61"),
    ("S100", "S51", "S61"),
    ("S100", "S51", "S52", "S53", "S61"),
    ("S100", "S51", "S53", "S54", "S61"),
    ("S100", "S53", "S51", "S61"),
])
def test_missing_multiple_or_reversed_events_do_not_become_clean_blocks(markers):
    report = feedback_blocks(events(*markers))
    assert len(report["blocks"]) == 1
    assert report["blocks"][0]["strict_comparable"] is False
    assert report["blocks"][0]["issues"]


def test_multiple_choice_and_expectation_rows_are_preserved_without_representative_selection():
    block = feedback_blocks(events("S100", "S51", "S52", "S53", "S54", "S61"))["blocks"][0]
    assert block["choice_event_rows"] == [3, 4]
    assert block["expectation_event_rows"] == [5, 6]
    assert block["choice_side_code"] is None
    assert block["expectation_mat_code"] is None
    assert block["strict_comparable"] is False


@pytest.mark.parametrize("inserted", ["S999", "not-a-trigger", "S150", "S151", "S101", "S200", "S73"])
def test_unknown_boundary_pause_or_other_task_after_choice_prevents_strict_comparison(inserted):
    report = feedback_blocks(events("S100", "S51", inserted, "S53", "S61"))
    block = report["blocks"][0]
    assert block["strict_comparable"] is False
    assert block["issues"]
    assert block["feedback_flag"] == 1


def test_feedback_without_active_task_is_still_kept_as_raw_observation():
    report = feedback_blocks(events("S51", "S53", "S61"))
    assert len(report["blocks"]) == 1
    assert report["blocks"][0]["strict_comparable"] is False
    assert report["blocks"][0]["feedback_event_row"] == 4


def test_pause_before_choice_with_proper_resume_does_not_invent_or_drop_feedback():
    report = feedback_blocks(events("S100", "S150", "S151", "S51", "S53", "S61", "S101"))
    assert len(report["blocks"]) == 1
    assert report["blocks"][0]["strict_comparable"] is True


def test_decreasing_time_inside_choice_feedback_window_is_retained_and_flagged():
    raw = events("S100", "S51", "S53", "S61", "S101")
    raw[2]["onset"] = "0.5"
    report = feedback_blocks(raw)
    assert report["event_count"] == 5
    block = report["blocks"][0]
    assert block["strict_comparable"] is False
    assert block["issues"]
    assert block["choice_event_rows"] == [3]
    assert block["expectation_event_rows"] == [4]


def test_tail_events_and_no_feedback_are_not_synthesized_into_blocks():
    raw = events("S100", "S51", "S53", "S101")
    report = feedback_blocks(raw)
    assert report["event_count"] == len(raw)
    assert report["blocks"] == []
    assert report["tail_event_rows"] == [2, 3, 4, 5]


def test_other_task_feedback_starts_a_new_global_block_without_reordering():
    report = feedback_blocks(events("S100", "S51", "S53", "S61", "S81", "S52", "S54", "S60"))
    assert [block["feedback_event_row"] for block in report["blocks"]] == [5, 6, 9]
    assert report["blocks"][1]["strict_comparable"] is False
    assert report["blocks"][2]["choice_event_rows"] == [7]
    assert report["blocks"][2]["expectation_event_rows"] == [8]


def test_block_parser_does_not_rewrite_the_source_event_list():
    raw = events("S100", "S 51", "S_53", "S61", "S101")
    original = deepcopy(raw)
    feedback_blocks(raw)
    assert raw == original
