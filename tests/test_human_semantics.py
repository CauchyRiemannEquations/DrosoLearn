"""Synthetic-only fixed-code contingency tests; no human performance analysis."""
from copy import deepcopy

import pytest

from drosolearn.human_semantics import summarize


LINKS = [{"event_label": "s-a", "mat_row": "mat-01"}]


def row(task="REW", ordinal=1, *, event_label="s-a", mat_row="mat-01", strict=True,
        action=1, mat_action=1, expectation=3, mat_expectation=3):
    return {"event_label": event_label, "mat_row": mat_row, "task": task,
            "feedback_ordinal": str(ordinal),
            "feedback_event_row": str(ordinal * 3 + 2 + (10000 if task == "PUN" else 0)),
            "strict_comparable": str(strict),
            "event_choice_side_code": str(action) if action is not None else "",
            "mat_action_code": str(mat_action),
            "event_expectation_plus3": str(expectation) if expectation is not None else "",
            "mat_expect_code": str(mat_expectation),
            "action_numeric_agreement": str(action == mat_action) if strict else "",
            "expectation_numeric_agreement": str(expectation == mat_expectation) if strict else ""}


def simple_rows():
    return [row(), row("PUN")]


def test_fixed_tables_preserve_all_cells_and_count_only_strict_rows():
    rows = [row(), row(ordinal=2, action=2, expectation=6, mat_expectation=4),
            row("PUN", mat_action=2, expectation=4, mat_expectation=6),
            row("PUN", 2, strict=False, action=None, expectation=None)]
    result = summarize(rows, LINKS)
    assert result["total_rows"] == 4
    assert result["strict_rows"] == 3
    assert result["not_strict_rows"] == 1
    assert result["action_matrix"] == [[1, 1], [1, 0]]
    assert result["expectation_matrix"] == [[1, 0, 0, 0], [0, 0, 0, 1],
                                               [0, 0, 0, 0], [0, 1, 0, 0]]
    assert result["action_disagreements"] == 2
    assert result["expectation_disagreements"] == 2
    assert result["mapping_selected"] is False
    assert result["behavioral_performance_computed"] is False


def test_full_grid_remains_fixed_even_when_different_mapping_could_change_diagonal():
    rows = [row(task, index + 1, action=1 + ((event + mat) % 2), mat_action=1 + (mat % 2),
                expectation=event + 3, mat_expectation=mat + 3)
            for task in ("REW", "PUN")
            for index, (event, mat) in enumerate((a, b) for a in range(4) for b in range(4))]
    result = summarize(rows, LINKS)
    assert result["expectation_matrix"] == [[2, 2, 2, 2] for _ in range(4)]
    assert result["action_matrix"] == [[8, 8], [8, 8]]
    assert result["strict_rows"] == 32
    assert result["action_disagreements"] == 16
    assert result["expectation_disagreements"] == 24
    assert result["mapping_selected"] is False


def test_all_off_diagonal_rows_are_not_relabelled_to_make_a_better_mapping():
    rows = [row(task, ordinal, action=1, mat_action=2, expectation=3, mat_expectation=6)
            for task in ("REW", "PUN") for ordinal in (1, 2)]
    result = summarize(rows, LINKS)
    assert result["action_matrix"] == [[0, 4], [0, 0]]
    assert result["expectation_matrix"][0][3] == 4
    assert result["action_disagreements"] == result["expectation_disagreements"] == 4
    assert result["mapping_selected"] is False


def test_all_nonstrict_rows_remain_outside_matrices_and_are_counted_separately():
    result = summarize([row(task, strict=False, action=None, expectation=None) for task in ("REW", "PUN")], LINKS)
    assert result["total_rows"] == result["not_strict_rows"] == 2
    assert result["strict_rows"] == 0
    assert result["action_matrix"] == [[0, 0], [0, 0]]
    assert result["expectation_matrix"] == [[0, 0, 0, 0] for _ in range(4)]
    assert result["action_disagreements"] == result["expectation_disagreements"] == 0


def test_condition_groups_add_back_to_overall_tables_without_cross_task_pooling():
    rows = [row(), row("REW", 2, mat_action=2, mat_expectation=4),
            row("PUN", action=2, mat_action=2, expectation=6, mat_expectation=6)]
    result = summarize(rows, LINKS)
    groups = result["groups"]
    assert len(groups) == 2
    assert {group["task"] for group in groups} == {"REW", "PUN"}
    assert sum(group["total_rows"] for group in groups) == result["total_rows"]
    assert sum(group["strict_rows"] for group in groups) == result["strict_rows"]
    assert sum(group["not_strict_rows"] for group in groups) == result["not_strict_rows"]
    for field, size in (("action_matrix", 2), ("expectation_matrix", 4)):
        assert [[sum(group[field][a][b] for group in groups) for b in range(size)] for a in range(size)] == result[field]


def test_multiple_links_keep_each_record_pair_and_each_task_separate():
    links = LINKS + [{"event_label": "s-b", "mat_row": "mat-02"}]
    rows = simple_rows() + [row(task, event_label="s-b", mat_row="mat-02", action=2, mat_action=2)
                            for task in ("REW", "PUN")]
    result = summarize(rows, links)
    assert len(result["groups"]) == 4
    assert {(group["event_label"], group["mat_row"], group["task"]) for group in result["groups"]} == {
        (link["event_label"], link["mat_row"], task) for link in links for task in ("REW", "PUN")}


def test_input_order_does_not_select_a_mapping_or_rewrite_records():
    rows = [row(), row("PUN"), row("REW", 2, action=2, mat_action=1)]
    original = deepcopy((rows, LINKS))
    assert summarize(rows, LINKS) == summarize(list(reversed(rows)), list(reversed(LINKS)))
    assert (rows, LINKS) == original


@pytest.mark.parametrize("field,value", [
    ("event_choice_side_code", "0"), ("event_choice_side_code", "3"),
    ("event_choice_side_code", "1.0"), ("event_choice_side_code", "True"),
    ("event_choice_side_code", ""), ("mat_action_code", "0"),
    ("event_expectation_plus3", "2"), ("event_expectation_plus3", "7"),
    ("event_expectation_plus3", "3.0"), ("event_expectation_plus3", ""),
    ("mat_expect_code", "0"), ("mat_expect_code", "7"),
])
def test_strict_rows_reject_missing_unknown_or_noninteger_codes(field, value):
    rows = simple_rows()
    rows[0][field] = value
    with pytest.raises(ValueError):
        summarize(rows, LINKS)


@pytest.mark.parametrize("field", ["action_numeric_agreement", "expectation_numeric_agreement"])
def test_claimed_agreement_must_match_fixed_code_comparison(field):
    rows = simple_rows()
    rows[0][field] = "False"
    with pytest.raises(ValueError):
        summarize(rows, LINKS)


@pytest.mark.parametrize("field", ["action_numeric_agreement", "expectation_numeric_agreement"])
@pytest.mark.parametrize("bad", ["True", "False"])
def test_nonstrict_rows_cannot_contribute_agreement_flags(field, bad):
    rows = [row("REW", strict=False), row("PUN")]
    rows[0][field] = bad
    with pytest.raises(ValueError):
        summarize(rows, LINKS)


@pytest.mark.parametrize("field,value", [
    ("strict_comparable", "true"), ("strict_comparable", "1"),
    ("strict_comparable", ""), ("action_numeric_agreement", "true"),
    ("expectation_numeric_agreement", "1"),
])
def test_boolean_flags_require_literal_registered_csv_encoding(field, value):
    rows = simple_rows()
    rows[0][field] = value
    with pytest.raises(ValueError):
        summarize(rows, LINKS)


def test_duplicate_record_task_ordinal_key_is_rejected_even_if_values_identical():
    rows = simple_rows()
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError):
        summarize(rows, LINKS)


def test_same_source_event_row_cannot_be_reused_for_another_context():
    rows = simple_rows()
    rows[1]["feedback_event_row"] = rows[0]["feedback_event_row"]
    with pytest.raises(ValueError):
        summarize(rows, LINKS)


@pytest.mark.parametrize("ordinal", ["0", "-1", "1.0", "True", ""])
def test_ordinals_are_positive_integers_not_coerced(ordinal):
    rows = simple_rows()
    rows[0]["feedback_ordinal"] = ordinal
    with pytest.raises(ValueError):
        summarize(rows, LINKS)


def test_missing_ordinal_is_not_filled_or_renumbered():
    rows = [row(), row("REW", 3), row("PUN")]
    with pytest.raises(ValueError):
        summarize(rows, LINKS)


def test_missing_task_group_is_not_treated_as_zero_measurements():
    with pytest.raises(ValueError):
        summarize([row("REW")], LINKS)


def test_unknown_task_is_not_folded_into_a_known_context():
    rows = simple_rows()
    rows[0]["task"] = "reward"
    with pytest.raises(ValueError):
        summarize(rows, LINKS)


@pytest.mark.parametrize("field,value", [("event_label", "unknown"), ("mat_row", "mat-99")])
def test_rows_must_belong_to_the_registered_link_pair(field, value):
    rows = simple_rows()
    rows[0][field] = value
    with pytest.raises(ValueError):
        summarize(rows, LINKS)


@pytest.mark.parametrize("additional", [
    {"event_label": "s-a", "mat_row": "mat-02"},
    {"event_label": "s-b", "mat_row": "mat-01"},
    {"event_label": "s-a", "mat_row": "mat-01"},
])
def test_links_must_be_one_to_one_without_duplicate_pairs(additional):
    with pytest.raises(ValueError):
        summarize(simple_rows(), LINKS + [additional])


def test_an_unrepresented_link_is_not_silently_discarded():
    with pytest.raises(ValueError):
        summarize(simple_rows(), LINKS + [{"event_label": "s-b", "mat_row": "mat-02"}])


def test_empty_records_or_links_are_not_a_passed_semantics_audit():
    with pytest.raises(ValueError):
        summarize([], LINKS)
    with pytest.raises(ValueError):
        summarize(simple_rows(), [])
