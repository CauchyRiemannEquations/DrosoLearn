"""Conditional choice persistence, with an explicit original-adjacency ledger.

This module estimates no causal reward effect and performs no species comparison.
It consumes observed actions, never policy probabilities or expected updates.
"""
from __future__ import annotations

from collections import Counter
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binary(value):
    return type(value) is int and value in (0, 1)


def validate_rows(rows):
    require(isinstance(rows, list) and len(rows) > 0, "A nonempty original stream is required")
    fields = {"stream_id", "kind", "coordinate", "source_position", "trial", "task", "phase",
              "state", "choice", "feedback", "actual", "training"}
    previous = None
    for row in rows:
        require(fields <= row.keys(), "Canonical fields are missing")
        for key in ("stream_id", "kind", "coordinate", "task", "phase", "state"):
            require(type(row[key]) is str and row[key] != "", "Known nonempty semantic identifiers required")
        for key in ("source_position", "trial"):
            require(type(row[key]) is int and row[key] > 0, "Positive integer original positions required")
        require(binary(row["choice"]) or row["choice"] is None or row["choice"] == "query",
                "Unknown choice code")
        require(not isinstance(row["choice"], bool), "Boolean choices are not integer actions")
        require(binary(row["feedback"]) or row["feedback"] is None or row["feedback"] == "information",
                "Unknown actual feedback code")
        require(not isinstance(row["feedback"], bool), "Boolean feedback is not an integer outcome")
        require(type(row["actual"]) is bool and type(row["training"]) is bool,
                "Actual/training flags must be booleans")
        if previous is not None:
            require(all(row[k] == previous[k] for k in ("stream_id", "kind", "coordinate")),
                    "A stream cannot change identity or choice coordinate")
            require(row["trial"] > previous["trial"] and row["source_position"] > previous["source_position"],
                    "Repeated or reversed original positions")
        previous = row


def pair_reasons(current, following):
    reasons = []
    for prefix, row in (("current", current), ("next", following)):
        if row["choice"] == "query":
            reasons.append(prefix + "_query")
        elif row["choice"] is None:
            reasons.append(prefix + "_choice_missing")
    if current["feedback"] is None:
        reasons.append("current_feedback_missing")
    elif not binary(current["feedback"]):
        reasons.append("current_feedback_nonbinary")
    if following["trial"] != current["trial"] + 1:
        reasons.append("trial_gap")
    for key in ("task", "phase", "state"):
        if current[key] != following[key]:
            reasons.append(key + "_boundary")
    for key in ("actual", "training"):
        if not current[key] or not following[key]:
            reasons.append("not_" + key)
    return sorted(reasons)


def measure_stream(rows):
    """Return counts and one ledger item for every original neighboring pair.

    Missing/query trials stay in place. A missing *next* feedback alone does not
    exclude the present pair. A zero denominator is represented by None.
    """
    validate_rows(rows)
    by_feedback = {str(f): {"denominator": 0, "stay": 0, "switch": 0} for f in (0, 1)}
    ledger, reasons_count = [], Counter()
    for current, following in zip(rows, rows[1:]):
        reasons = pair_reasons(current, following)
        eligible = not reasons
        item = {"stream_id": current["stream_id"], "source_position": current["source_position"],
                "next_source_position": following["source_position"], "trial": current["trial"],
                "next_trial": following["trial"], "eligible": int(eligible),
                "reasons": "|".join(reasons),
                "feedback": current["feedback"] if binary(current["feedback"]) else "", "stay": ""}
        reasons_count.update(reasons)
        if eligible:
            stay = int(current["choice"] == following["choice"])
            item["stay"] = stay
            cell = by_feedback[str(current["feedback"])]
            cell["denominator"] += 1
            cell["stay"] += stay
            cell["switch"] += 1 - stay
        ledger.append(item)
    for cell in by_feedback.values():
        d = cell["denominator"]
        cell["stay_rate"] = cell["stay"] / d if d else None
        cell["switch_rate"] = cell["switch"] / d if d else None
    eligible = sum(cell["denominator"] for cell in by_feedback.values())
    summary = {k: rows[0][k] for k in ("stream_id", "kind", "coordinate")}
    summary.update(trials=len(rows), candidates=len(ledger), eligible=eligible,
                   excluded=len(ledger) - eligible, reason_counts=dict(sorted(reasons_count.items())),
                   by_feedback=by_feedback)
    return summary, ledger


def human_rows(raw, *, stream_id, source_kind="human", trials=1000, regime_boundary_after_trial=800):
    """Decode the separately verified source-coded card identity and actual result."""
    import numpy as np
    arrays = {key: np.asarray(raw[key]) for key in ("A", "R", "S", "C")}
    shapes = {"A": (1, trials), "R": (trials, 2), "S": (trials, 1), "C": (1, trials)}
    require(all(arrays[k].shape == shape and arrays[k].dtype.kind in "iuf" for k, shape in shapes.items()),
            "Human source schema differs")
    require(all(not np.isinf(a).any() for a in arrays.values()), "Infinite human values")
    a, s, r, c = arrays["A"][0], arrays["S"][:, 0], arrays["R"], arrays["C"][0]
    require(np.isin(a[~np.isnan(a)], [1, 2, 3]).all() and np.isin(s, [1, 2]).all()
            and np.isin(r, [-1, 1]).all(), "Unknown human source codes")
    binary_mask = np.isin(a, [1, 2])
    require(np.isfinite(c[binary_mask]).all() and
            np.array_equal(c[binary_mask], (a[binary_mask] == s[binary_mask]).astype(int)),
            "Human C/action/state structural gate differs")
    require(np.isnan(c[~binary_mask]).all(), "Query/missing C must remain missing")
    require(type(regime_boundary_after_trial) is int and regime_boundary_after_trial > 0,
            "Known positive regime boundary required")
    rows = []
    for i in range(trials):
        if math.isnan(float(a[i])):
            choice, feedback = None, None
        elif a[i] == 3:
            choice, feedback = "query", "information"
        else:
            choice = int(a[i]) - 1
            feedback = int((r[i, choice] + 1) / 2)
        rows.append({"stream_id": stream_id, "kind": source_kind,
                     "coordinate": "source_coded_card_identity", "source_position": i + 1,
                     "trial": i + 1, "task": "markovic_card",
                     "phase": "initial" if i < regime_boundary_after_trial else "transfer", "state": str(int(s[i])),
                     "choice": choice, "feedback": feedback, "actual": True, "training": True})
    return rows


def model_row(record, *, source_position):
    """Map actual screen-side action to smaller/larger choice in the registered phase."""
    def integer(key):
        token = record[key]
        require(isinstance(token, str) and token.strip() == token and token.isdecimal(), "Invalid model integer")
        return int(token)

    seed, trial, action = (integer(k) for k in ("seed", "trial", "action"))
    left, right, reward = (integer(k) for k in ("left_count", "right_count", "reward"))
    require(record["phase"] == "phase2" and {left, right} == {2, 4} and action in (0, 1)
            and reward in (0, 1) and trial > 0, "Registered model phase/choices differ")
    p = float(record["p_right"])
    require(math.isfinite(p) and 0 <= p <= 1, "Nonfinite or invalid model policy probability")
    choice = int((left, right)[action] == max(left, right))
    require(reward == int(choice == 0), "Deterministic phase2 reward does not match actual action")
    stream = f"model/{seed}/{record['condition']}/{record['history']}/phase2"
    return {"stream_id": stream, "kind": "model", "coordinate": "smaller_larger_quantity_category",
            "source_position": source_position, "trial": trial, "task": "abstract_quantity", "phase": "phase2",
            "state": "smaller", "choice": choice, "feedback": reward, "actual": True, "training": True}
